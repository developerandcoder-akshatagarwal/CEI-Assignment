"""
GradCAM explainability, using the pytorch-grad-cam package. Hooks
layer4's last conv block (before global pooling) — hooking the final
FC layer instead gives blurry, uninformative maps since spatial
information is already collapsed by then.
"""
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def build_gradcam(model):
    target_layers = [model.layer4[-1]]
    return GradCAM(model=model, target_layers=target_layers)


def denormalize(img_tensor, mean, std):
    """img_tensor: (C, H, W) normalized tensor -> (H, W, C) float [0,1] numpy."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = img_tensor.cpu() * std_t + mean_t
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def generate_gradcam_overlay(gradcam, img_tensor, pred_class_idx, mean, std):
    """img_tensor: (C, H, W) normalized tensor for ONE image (no batch dim).
    Returns an (H, W, 3) uint8 overlay image ready to display or save."""
    input_tensor = img_tensor.unsqueeze(0)
    targets = [ClassifierOutputTarget(pred_class_idx)]
    grayscale_cam = gradcam(input_tensor=input_tensor, targets=targets)[0]  # (H, W)

    rgb_img = denormalize(img_tensor, mean, std)
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return overlay
