"""Visualization utilities for colorization results."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, List


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) or (B, C, H, W) tensor to numpy (H, W, C) in [0, 1]."""
    if tensor.dim() == 4:
        tensor = tensor[0]  # take first image
    img = tensor.detach().cpu().float().clamp(0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()
    # If single channel, squeeze
    if img.shape[-1] == 1:
        img = img[:, :, 0]
    return img


def plot_colorization(
    grayscale: torch.Tensor,
    colorized: torch.Tensor,
    ground_truth: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> None:
    """Display grayscale, colorized, and optionally ground truth side by side.

    Args:
        grayscale: (C, H, W) or (1, H, W) grayscale input.
        colorized: (3, H, W) generated RGB.
        ground_truth: (3, H, W) real RGB (optional).
        save_path: If provided, save figure to this path.
    """
    n_cols = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    gray_np = tensor_to_numpy(grayscale)
    axes[0].imshow(gray_np, cmap="gray")
    axes[0].set_title("Grayscale Input")
    axes[0].axis("off")

    color_np = tensor_to_numpy(colorized)
    axes[1].imshow(color_np)
    axes[1].set_title("Colorized Output")
    axes[1].axis("off")

    if ground_truth is not None:
        gt_np = tensor_to_numpy(ground_truth)
        axes[2].imshow(gt_np)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_video_frame_comparison(
    gray_frames: List[torch.Tensor],
    colorized_frames: List[torch.Tensor],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a grid of video frames showing before/after colorization.

    Args:
        gray_frames: List of grayscale frame tensors.
        colorized_frames: List of corresponding colorized frame tensors.
        titles: Optional frame titles.
        save_path: Optional save path.
    """
    n = len(gray_frames)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for i in range(n):
        gray_np = tensor_to_numpy(gray_frames[i])
        axes[0, i].imshow(gray_np, cmap="gray")
        axes[0, i].axis("off")
        if titles and i < len(titles):
            axes[0, i].set_title(titles[i])

        color_np = tensor_to_numpy(colorized_frames[i])
        axes[1, i].imshow(color_np)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Grayscale", fontsize=12)
    axes[1, 0].set_ylabel("Colorized", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
