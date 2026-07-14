"""Grad-CAM, implemented from scratch with forward/backward hooks.

Written out rather than imported from a library because you will be asked, in
an interview, how it works. The mechanism:

  1. Forward pass; capture the activations A^k of the last conv layer.
  2. Backward pass from the score for class c; capture the gradients dY^c/dA^k.
  3. Global-average-pool those gradients -> one weight a_k per channel.
     A channel whose activation strongly raises the class score gets a high
     weight.
  4. Weighted sum of the activation maps, then ReLU: we keep only the evidence
     FOR class c, discarding evidence against it.
  5. Upsample to input size.

Same tooling as the occlusion-sensitivity work in the thesis; the point here is
to show the model is looking at the casting surface and not at the background,
the label, or a lighting artefact.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, out) -> None:
        self.activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, float]:
        """x: (1, C, H, W). Returns (heatmap HxW in [0,1], predicted class, P(defect))."""
        if x.dim() != 4 or x.size(0) != 1:
            raise ValueError("GradCAM expects a single image with shape (1, C, H, W)")

        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(logits.argmax(dim=1).item())
        p_defect = float(probs[0, 1].item())

        target = pred if class_idx is None else class_idx
        logits[0, target].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks captured nothing — is target_layer inside the model?")

        # weights: global average pool of gradients over spatial dims -> (C,)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        denom = cam.max()
        if denom > 1e-8:
            cam /= denom
        return cam, pred, p_defect


def overlay_heatmap(image_hw3: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap over a [0,1] RGB image using a jet-like colormap."""
    import matplotlib

    # matplotlib.cm.get_cmap() was removed in matplotlib 3.9; use the registry.
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * image_hw3 + alpha * heat, 0, 1)


def denormalize(tensor_chw: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation so the image can be displayed."""
    from src.data import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor_chw.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()
