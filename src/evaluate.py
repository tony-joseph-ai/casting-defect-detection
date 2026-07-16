"""Evaluate a checkpoint and produce every figure the README needs.

    python -m src.evaluate --checkpoint outputs/best_resnet18.pt

Produces, in outputs/:
    confusion_matrix_<tag>.png
    threshold_sweep_<tag>.png      <- the cost argument, visualised
    gradcam_correct_<tag>.png      <- model looks at the defect
    gradcam_errors_<tag>.png       <- where and how it fails
    report_<tag>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import CLASS_NAMES, build_dataloaders
from src.gradcam import GradCAM, denormalize, overlay_heatmap
from src.metrics import binary_metrics, sweep_thresholds
from src.model import build_model, get_target_layer


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        probs.append(p.cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def plot_confusion(m: dict, path: Path, title: str) -> None:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    labels = ["ok", "defect"]
    ax.set_xticks([0, 1], labels=[f"pred {l}" for l in labels])
    ax.set_yticks([0, 1], labels=[f"true {l}" for l in labels])
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=15, fontweight="bold",
            )
    ax.text(0, 1, "\nMISSED DEFECTS", ha="center", va="top", color="crimson", fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(ths, costs, best, cost_fn, cost_fp, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(ths, costs, lw=2)
    ax.axvline(best, ls="--", color="crimson", label=f"cost-optimal = {best:.2f}")
    ax.axvline(0.5, ls=":", color="grey", label="default = 0.50")
    ax.set_xlabel("decision threshold on P(defect)")
    ax.set_ylabel(f"expected cost  ({cost_fn:g}×FN + {cost_fp:g}×FP)")
    ax.set_title("Operating threshold: 0.5 is an assumption, not a decision", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gradcam_grid(model, arch, dataset, indices, device, path: Path, title: str, threshold: float = 0.5) -> None:
    if not indices:
        print(f"  (no samples for {title} — skipping)")
        return
    cam_engine = GradCAM(model, get_target_layer(model, arch))
    n = min(len(indices), 6)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.0))
    axes = np.atleast_2d(axes)
    for col, idx in enumerate(indices[:n]):
        x, y = dataset[idx]
        cam, pred, p = cam_engine(x.unsqueeze(0).to(device))
        img = denormalize(x)
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"true: {CLASS_NAMES[y]}", fontsize=8)
        axes[1, col].imshow(overlay_heatmap(img, cam))
        pred_at_th = int(p >= threshold)
        axes[1, col].set_title(f"pred: {CLASS_NAMES[pred_at_th]}  p={p:.2f}", fontsize=8)
        for r in (0, 1):
            axes[r, col].axis("off")
    cam_engine.remove()
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/best_resnet18.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg, arch = ckpt["config"], ckpt["arch"]
    tag = Path(args.checkpoint).stem.replace("best_", "")

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, val_loader, test_loader, _ = build_dataloaders(cfg)
    out = Path(cfg["paths"]["output_dir"])

    # 1. Tune the threshold on VALIDATION, apply it to TEST. Never the reverse.
    cost_fn = cfg["eval"]["cost_false_negative"]
    cost_fp = cfg["eval"]["cost_false_positive"]
    val_probs, val_y = collect_probs(model, val_loader, device)
    ths, costs, best_th = sweep_thresholds(val_y, val_probs, cost_fn, cost_fp)
    plot_sweep(ths, costs, best_th, cost_fn, cost_fp, out / f"threshold_sweep_{tag}.png")
    print(f"cost-optimal threshold (from val) = {best_th:.3f}")

    # 2. Test-set metrics at both the default and the tuned threshold.
    test_probs, test_y = collect_probs(model, test_loader, device)
    m_default = binary_metrics(test_y, test_probs, 0.5)
    m_tuned = binary_metrics(test_y, test_probs, best_th)

    for name, m in (("threshold 0.50 (default)", m_default), (f"threshold {best_th:.2f} (tuned)", m_tuned)):
        print(f"\n=== TEST @ {name} ===")
        print(f"  accuracy        {m['accuracy']:.4f}")
        print(f"  precision(def)  {m['precision_defect']:.4f}")
        print(f"  recall(def)     {m['recall_defect']:.4f}")
        print(f"  f1(def)         {m['f1_defect']:.4f}")
        print(f"  ROC-AUC         {m['roc_auc']:.4f}")
        print(f"  missed defects  {m['fn']}   false alarms {m['fp']}")

    plot_confusion(m_tuned, out / f"confusion_matrix_{tag}.png",
                   f"{arch} — test set @ threshold {best_th:.2f}")

    # 3. Grad-CAM: successes and, more importantly, failures.
    test_ds = test_loader.dataset
    pred_tuned = (test_probs >= best_th).astype(int)
    correct_defects = [i for i in range(len(test_y)) if test_y[i] == 1 and pred_tuned[i] == 1]
    errors = [i for i in range(len(test_y)) if test_y[i] != pred_tuned[i]]

   gradcam_grid(model, arch, test_ds, correct_defects, device,
                 out / f"gradcam_correct_{tag}.png",
                 "Grad-CAM — correctly detected defects (is it looking at the casting?)",
                 threshold=best_th)
    gradcam_grid(model, arch, test_ds, errors, device,
                 out / f"gradcam_errors_{tag}.png",
                 "Grad-CAM — misclassifications (where does the evidence come from?)",
                 threshold=best_th)

    with open(out / f"report_{tag}.json", "w") as f:
        json.dump({"arch": arch, "tuned_threshold": best_th,
                   "test_default": m_default, "test_tuned": m_tuned}, f, indent=2)
    print(f"\nfigures + report written to {out}/")


if __name__ == "__main__":
    main()
