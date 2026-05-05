"""Image colorization inference.

Given a trained Multi-GAN ensemble, this module colorizes a single
grayscale image by:
1. Segmenting it into regions
2. Clustering by SMV
3. Routing each region to its cluster's GAN for colorization
4. Compositing the colorized regions into a final RGB image
"""

import numpy as np
import torch
import cv2
from typing import List, Optional, Tuple

from preprocessing.edge_detection import get_edge_map
from preprocessing.segmentation import segment_regions, compute_smv, Region
from preprocessing.clustering import cluster_regions, compute_dynamic_k
from models.multi_gan import MultiGAN


class ImageColorizer:
    """Inference engine for single-image colorization using Multi-GAN."""

    def __init__(
        self,
        multi_gan: MultiGAN,
        config: dict,
        device: torch.device = torch.device("cpu"),
    ):
        self.multi_gan = multi_gan
        self.device = device

        pre_cfg = config.get("preprocessing", {})
        self.edge_method = pre_cfg.get("edge_method", "canny")
        self.canny_low = pre_cfg.get("canny_low", 50)
        self.canny_high = pre_cfg.get("canny_high", 150)
        self.min_region_size = pre_cfg.get("min_region_size", 64)

        clust_cfg = config.get("clustering", {})
        self.gamma = clust_cfg.get("gamma", 0.05)
        self.k_min = clust_cfg.get("k_min", 10)
        self.k_max = clust_cfg.get("k_max", 64)

    def _map_cluster_to_gan(self, cluster_id: int) -> int:
        """Map a runtime cluster ID to a GAN index.

        Since runtime clustering may produce fewer than k_max clusters,
        we simply wrap around using modulo.
        """
        if cluster_id < self.multi_gan.num_clusters:
            return cluster_id
        return cluster_id % self.multi_gan.num_clusters

    def _generate_region(
        self,
        gray: torch.Tensor,      # (1, H, W)
        region: Region,
        gan_id: int,
        target_size: Tuple[int, int] = (128, 128),
    ) -> np.ndarray:
        """Generate color for a single region using its assigned GAN.

        Args:
            gray:     Full grayscale image tensor (1, H, W).
            region:   Region with mask and bbox.
            gan_id:   Cluster GAN index to use.
            target_size: Resize target for GAN input.

        Returns:
            RGB array (H, W, 3) in [0, 255], colorized for this region.
        """
        H, W = gray.shape[1:]
        r1, c1, r2, c2 = region.bbox

        # Extract patch with padding
        h = r2 - r1 + 1
        w = c2 - c1 + 1
        side = max(h, w)
        side = int(side * 1.2)

        cr = (r1 + r2) // 2
        cc = (c1 + c2) // 2
        r1_new = max(0, cr - side // 2)
        r2_new = min(H, r1_new + side)
        c1_new = max(0, cc - side // 2)
        c2_new = min(W, c1_new + side)

        patch = gray[:, r1_new:r2_new, c1_new:c2_new]  # (1, pH, pW)

        # Resize to GAN input size and normalize to [-1, 1]
        patch_resized = torch.nn.functional.interpolate(
            patch.unsqueeze(0), size=target_size, mode="bilinear", align_corners=True
        )
        patch_norm = patch_resized * 2.0 - 1.0

        # Run GAN
        colorized = self.multi_gan.generate_for_cluster(gan_id, patch_norm)  # (1, 3, S, S)
        colorized = (colorized + 1.0) / 2.0  # -> [0, 1]

        # Resize back to original patch size
        colorized_resized = torch.nn.functional.interpolate(
            colorized, size=(r2_new - r1_new, c2_new - c1_new),
            mode="bilinear", align_corners=True
        ).squeeze(0)  # (3, pH, pW)

        colorized_np = (colorized_resized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return colorized_np, (r1_new, c1_new, r2_new, c2_new)

    def colorize(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:
        """Colorize a grayscale image.

        Args:
            gray: (H, W) or (H, W, 1) grayscale image, uint8 or float in [0, 255].

        Returns:
            (H, W, 3) colorized RGB image, uint8.
        """
        if gray.ndim == 3:
            gray = gray[:, :, 0]
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)

        H, W = gray.shape
        gray_float = gray.astype(np.float32) / 255.0

        # --- Preprocessing ---
        edges = get_edge_map(gray, method=self.edge_method)
        regions = segment_regions(edges, min_region_size=self.min_region_size)

        if len(regions) == 0:
            # Fallback: colorize whole image using first GAN
            gray_tensor = torch.from_numpy(gray_float).unsqueeze(0).unsqueeze(0).float()
            gray_norm = gray_tensor * 2.0 - 1.0
            result = self.multi_gan.generate_for_cluster(0, gray_norm)
            result = (result.squeeze(0) + 1.0) / 2.0
            return (result.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        compute_smv(gray_float, regions)

        smv_values = np.array([r.smv for r in regions], dtype=np.float32)
        k = compute_dynamic_k(gray, gamma=self.gamma, k_min=self.k_min, k_max=self.k_max)
        labels, _ = cluster_regions(smv_values, k=k, gamma=self.gamma, k_min=self.k_min, k_max=self.k_max)

        for i, region in enumerate(regions):
            region.cluster_id = int(labels[i])

        # --- Generate per region ---
        gray_tensor = torch.from_numpy(gray_float).unsqueeze(0).float()  # (1, H, W)
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)

        for region in regions:
            gan_id = self._map_cluster_to_gan(region.cluster_id)

            color_patch, (pr1, pc1, pr2, pc2) = self._generate_region(
                gray_tensor, region, gan_id
            )

            # Place patch onto canvas masked by region
            region_crop = region.mask[pr1:pr2, pc1:pc2]
            canvas[pr1:pr2, pc1:pc2][region_crop] += color_patch.astype(np.float32)[region_crop]
            weight[pr1:pr2, pc1:pc2][region_crop] += 1.0

        # Normalize overlapping regions
        weight[weight == 0] = 1.0
        canvas /= weight[:, :, np.newaxis]

        return np.clip(canvas, 0, 255).astype(np.uint8)


def colorize_image(
    multi_gan: MultiGAN,
    gray_image: np.ndarray,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Convenience function to colorize a single image.

    Args:
        multi_gan:  Trained MultiGAN ensemble.
        gray_image: (H, W) or (H, W, 1) grayscale, uint8.
        config:     Configuration dictionary.
        device:     Torch device.

    Returns:
        (H, W, 3) colorized RGB, uint8.
    """
    colorizer = ImageColorizer(multi_gan, config, device)
    return colorizer.colorize(gray_image)
