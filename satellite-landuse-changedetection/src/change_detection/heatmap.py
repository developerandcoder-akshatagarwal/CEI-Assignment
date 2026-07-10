"""
Change heatmap generation. Two modes:
  1. Patch-grid mode (used for the ≥5 region-pair heatmaps in the
     evaluation notebook): splits each image into an NxN grid, embeds
     each patch, computes per-patch similarity, renders as a heat grid.
  2. Dashboard mode (single T1/T2 pair): same patch-grid logic, plus an
     overlay blended onto the T1 image for a more visual dashboard result.
"""
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _split_into_patches(img_tensor, grid_size=8):
    """img_tensor: (C, H, W). Returns list of patch tensors, row-major."""
    _, h, w = img_tensor.shape
    ph, pw = h // grid_size, w // grid_size
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = img_tensor[:, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
            patches.append(patch)
    return patches


@torch.no_grad()
def compute_patch_similarity_grid(embedding_model, img_t1, img_t2, device, grid_size=8):
    """img_t1, img_t2: normalized (C, H, W) tensors, same size.
    Returns a (grid_size, grid_size) numpy array of cosine similarities.
    """
    patches_t1 = _split_into_patches(img_t1, grid_size)
    patches_t2 = _split_into_patches(img_t2, grid_size)

    batch_t1 = torch.stack(patches_t1).to(device)
    batch_t2 = torch.stack(patches_t2).to(device)

    emb_t1 = embedding_model(batch_t1)
    emb_t2 = embedding_model(batch_t2)

    sims = torch.nn.functional.cosine_similarity(emb_t1, emb_t2, dim=1).cpu().numpy()
    return sims.reshape(grid_size, grid_size)


def plot_heatmap(sim_grid, save_path=None, title="Change Heatmap (red = low similarity / changed)"):
    fig, ax = plt.subplots(figsize=(5, 5))
    # invert so red = change (low similarity), blue/green = stable
    im = ax.imshow(1 - sim_grid, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Change intensity (1 - similarity)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def overlay_heatmap_on_image(base_image_np, sim_grid, alpha=0.45):
    """base_image_np: (H, W, 3) uint8 RGB image (denormalized).
    sim_grid: (grid_size, grid_size) similarity array.
    Returns an (H, W, 3) uint8 blended overlay for dashboard display."""
    h, w = base_image_np.shape[:2]
    change_intensity = 1 - sim_grid
    heat_resized = cv2.resize(change_intensity.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heat_colored = (plt.get_cmap("RdYlGn_r")(heat_resized)[:, :, :3] * 255).astype(np.uint8)
    blended = cv2.addWeighted(base_image_np, 1 - alpha, heat_colored, alpha, 0)
    return blended
