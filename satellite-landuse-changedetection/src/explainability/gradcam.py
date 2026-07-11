"""
GradCAM explainability -- implemented directly in plain PyTorch, with NO
dependency on the pytorch_grad_cam package or OpenCV. The pytorch_grad_cam
library has OpenCV hard-baked into its own internals (not something we
can remove by fixing our code), and OpenCV itself was unreliable on the
deployment environment -- so this avoids that entire dependency chain.

Hooks `layer4`'s last conv block (before global pooling) -- hooking the
final FC layer instead gives blurry, uninformative maps since spatial
information is already collapsed by then.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class SimpleGradCAM:
    """Minimal GradCAM: forward hook captures activations, backward hook
    captures gradients, then activations are weighted by the mean
    gradient per channel (the standard GradCAM formulation)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, target_class_idx]
        score.backward()

        # global-average-pool the gradients per channel -> per-channel weight
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # only positive influence counts, standard GradCAM step

        cam = cam.squeeze().cpu().numpy()
        # normalize to [0, 1] for consistent visualization
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam


def build_gradcam(model):
    target_layer = model.layer4[-1]
    return SimpleGradCAM(model, target_layer)


def denormalize(img_tensor, mean, std):
    """img_tensor: (C, H, W) normalized tensor -> (H, W, C) float [0,1] numpy."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = img_tensor.cpu() * std_t + mean_t
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def _apply_colormap_red(cam_2d):
    """Simple red-heat colormap without matplotlib/cv2 dependency --
    low activation -> transparent-ish, high activation -> red."""
    h, w = cam_2d.shape
    heat_rgb = np.zeros((h, w, 3), dtype=np.float32)
    heat_rgb[..., 0] = cam_2d  # red channel scales with activation
    return heat_rgb


def generate_gradcam_overlay(gradcam, img_tensor, pred_class_idx, mean, std):
    """img_tensor: (C, H, W) normalized tensor for ONE image (no batch dim).
    Returns an (H, W, 3) uint8 overlay image ready to display or save."""
    input_tensor = img_tensor.unsqueeze(0)
    input_tensor.requires_grad_(False)

    cam_2d = gradcam(input_tensor, pred_class_idx)  # (h', w'), smaller than input due to conv stride

    # resize CAM up to input resolution using PIL (no cv2 needed)
    target_h, target_w = img_tensor.shape[1], img_tensor.shape[2]
    cam_resized = np.array(
        Image.fromarray(cam_2d.astype(np.float32), mode="F").resize((target_w, target_h), Image.BILINEAR)
    )

    rgb_img = denormalize(img_tensor, mean, std)
    heat_rgb = _apply_colormap_red(cam_resized)

    alpha = 0.5
    overlay = (rgb_img * (1 - alpha) + heat_rgb * alpha)
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)
