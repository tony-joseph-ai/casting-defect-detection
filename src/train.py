"""Training entry point.

    python -m src.train --config configs/config.yaml
    python -m src.train --config configs/config.yaml --arch simple_cnn --tag baseline

Model selection is on defect RECALL, not accuracy. On this dataset a model that
predicts "ok" for everything scores respectably on accuracy and is worthless.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data import build_dataloaders
from src.metrics import binary_metrics
from src.model import build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, threshold: float = 0.5) -> dict:
    model.eval()
    probs_all, targets_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(defect)
        probs_all.append(probs.cpu())
        targets_all.append(y)
    probs = torch.cat(probs_all).numpy()
    targets = torch.cat(targets_all).numpy()
    return binary_metrics(targets, probs, threshold)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
        n += x.size(0)
    return running / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--arch", default=None, help="override model.arch")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default=None, help="suffix for output files")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.arch:
        cfg["model"]["arch"] = args.arch
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs

    arch = cfg["model"]["arch"]
    tag = args.tag or arch
    set_seed(cfg["data"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  arch={arch}")

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"best_{tag}.pt"

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cfg)

    model = build_model(cfg).to(device)

    weights = class_weights.to(device) if cfg["train"]["use_class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )

    monitor = cfg["train"]["monitor"]
    patience = cfg["train"]["early_stopping_patience"]
    best_score, best_epoch, history = -1.0, -1, []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, device, cfg["eval"]["threshold"])
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, **val})
        print(
            f"epoch {epoch:02d}  loss={train_loss:.4f}  "
            f"val_acc={val['accuracy']:.4f}  val_recall_defect={val['recall_defect']:.4f}  "
            f"val_precision_defect={val['precision_defect']:.4f}  "
            f"val_f1={val['f1_defect']:.4f}  ({time.time() - t0:.1f}s)"
        )

        if val[monitor] > best_score:
            best_score, best_epoch = val[monitor], epoch
            torch.save(
                {"state_dict": model.state_dict(), "arch": arch, "config": cfg},
                ckpt_path,
            )
            print(f"  -> new best ({monitor}={best_score:.4f}), saved {ckpt_path}")

        if epoch - best_epoch >= patience:
            print(f"early stopping: no improvement in {patience} epochs")
            break

    # Test set is touched exactly once, here, with the selected checkpoint.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    test = evaluate(model, test_loader, device, cfg["eval"]["threshold"])
    print("\n=== TEST (threshold=%.2f) ===" % cfg["eval"]["threshold"])
    for k, v in test.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(out_dir / f"results_{tag}.json", "w") as f:
        json.dump(
            {"arch": arch, "best_epoch": best_epoch, "history": history, "test": test},
            f,
            indent=2,
        )
    print(f"\nwrote {out_dir / f'results_{tag}.json'}")


if __name__ == "__main__":
    main()
