"""Multi-GAN trainer for image colorization.

Orchestrates the full training pipeline: preprocessing (segmentation +
clustering) followed by per-cluster GAN training.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple

from preprocessing.edge_detection import get_edge_map
from preprocessing.segmentation import segment_regions, compute_smv, Region
from preprocessing.clustering import cluster_regions, compute_dynamic_k
from models.multi_gan import MultiGAN
from utils.metrics import compute_psnr, compute_ssim


class MultiGANTrainer:
    """Handles the full Multi-GAN training cycle."""

    def __init__(
        self,
        multi_gan: MultiGAN,
        config: dict,
        device: torch.device = torch.device("cpu"),
    ):
        self.multi_gan = multi_gan
        self.config = config
        self.device = device

        # Preprocessing params
        pre_cfg = config.get("preprocessing", {})
        self.edge_method = pre_cfg.get("edge_method", "canny")
        self.canny_low = pre_cfg.get("canny_low", 50)
        self.canny_high = pre_cfg.get("canny_high", 150)
        self.min_region_size = pre_cfg.get("min_region_size", 64)

        # Clustering params
        clust_cfg = config.get("clustering", {})
        self.gamma = clust_cfg.get("gamma", 0.05)
        self.k_min = clust_cfg.get("k_min", 10)
        self.k_max = clust_cfg.get("k_max", 64)

        # Training params
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 100)
        self.save_interval = train_cfg.get("save_interval", 10)
        self.log_interval = train_cfg.get("log_interval", 100)

        # Output
        out_cfg = config.get("output", {})
        self.model_dir = out_cfg.get("model_dir", "./checkpoints")
        self.result_dir = out_cfg.get("result_dir", "./results")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        # Stats
        self.global_step = 0

    def _preprocess_image(
        self, gray: np.ndarray
    ) -> Tuple[List[Region], np.ndarray, int]:
        """Preprocess a single grayscale image: edge detect, segment, cluster.

        Args:
            gray: (H, W) float or uint8 image in [0, 1] or [0, 255].

        Returns:
            regions:  List of Region objects with cluster_id assigned.
            labels:   Cluster labels (N,).
            k:        Number of clusters used.
        """
        # Edge detection
        edges = get_edge_map(gray, method=self.edge_method)

        # Region segmentation
        regions = segment_regions(edges, min_region_size=self.min_region_size)
        if len(regions) == 0:
            return [], np.array([]), 0

        # Compute SMV
        compute_smv(gray, regions)

        # Clustering
        smv_values = np.array([r.smv for r in regions], dtype=np.float32)

        k = compute_dynamic_k(
            gray if gray.dtype == np.uint8 else (gray * 255).astype(np.uint8),
            gamma=self.gamma,
            k_min=self.k_min,
            k_max=self.k_max,
        )

        labels, _ = cluster_regions(
            smv_values, k=k, gamma=self.gamma, k_min=self.k_min, k_max=self.k_max
        )

        # Assign cluster IDs to regions
        for i, region in enumerate(regions):
            region.cluster_id = int(labels[i])

        return regions, labels, k

    def _extract_patches(
        self,
        gray: torch.Tensor,    # (1, H, W)
        color: torch.Tensor,   # (3, H, W)
        regions: List[Region],
        target_size: Tuple[int, int] = (128, 128),
    ) -> Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Extract square patches for each region, grouped by cluster.

        Each region is cropped via its bounding box, padded to square,
        and resized to target_size.

        Returns:
            Dict mapping cluster_id -> (gray_patches, color_patches) lists.
        """
        clusters: Dict[int, Tuple[List, List]] = {}

        H, W = gray.shape[1:]

        for region in regions:
            cid = region.cluster_id
            r1, c1, r2, c2 = region.bbox

            # Expand bbox slightly and make square
            h = r2 - r1 + 1
            w = c2 - c1 + 1
            side = max(h, w)
            # Add 20% padding
            side = int(side * 1.2)

            # Center the crop
            cr = (r1 + r2) // 2
            cc = (c1 + c2) // 2
            r1_new = max(0, cr - side // 2)
            r2_new = min(H, r1_new + side)
            c1_new = max(0, cc - side // 2)
            c2_new = min(W, c1_new + side)

            gray_patch = gray[:, r1_new:r2_new, c1_new:c2_new]
            color_patch = color[:, r1_new:r2_new, c1_new:c2_new]

            # Resize to fixed size
            gray_patch = torch.nn.functional.interpolate(
                gray_patch.unsqueeze(0), size=target_size, mode="bilinear", align_corners=True
            ).squeeze(0)
            color_patch = torch.nn.functional.interpolate(
                color_patch.unsqueeze(0), size=target_size, mode="bilinear", align_corners=True
            ).squeeze(0)

            if cid not in clusters:
                clusters[cid] = ([], [])
            clusters[cid][0].append(gray_patch)
            clusters[cid][1].append(color_patch)

        return clusters

    def train_epoch(
        self, dataloader, epoch: int
    ) -> Dict[str, float]:
        """Run one training epoch.

        For each image in the batch:
        1. Preprocess (segment + cluster)
        2. Extract patches per cluster
        3. Train each cluster GAN on its patches

        Args:
            dataloader: Training dataloader yielding (gray, color) pairs.
            epoch:      Current epoch number.

        Returns:
            Average loss dict for the epoch.
        """
        epoch_losses = {"loss_d": 0.0, "loss_g": 0.0, "loss_g_l1": 0.0, "loss_g_gan": 0.0}
        n_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for gray_batch, color_batch in pbar:
            # gray_batch: (B, 1, H, W), color_batch: (B, 3, H, W)

            batch_losses = {"loss_d": 0.0, "loss_g": 0.0, "loss_g_l1": 0.0, "loss_g_gan": 0.0}
            batch_steps = 0

            for i in range(gray_batch.size(0)):
                gray = gray_batch[i]    # (1, H, W)
                color = color_batch[i]  # (3, H, W)

                # Convert to numpy for preprocessing
                gray_np = (gray.squeeze(0).numpy() * 255).astype(np.uint8)

                # Preprocess: segment and cluster
                regions, labels, k = self._preprocess_image(gray_np)

                if k == 0 or len(regions) == 0:
                    continue

                # Extract patches grouped by cluster
                clusters = self._extract_patches(gray, color, regions)

                # Train each cluster GAN
                for cid, (gray_patches, color_patches) in clusters.items():
                    if len(gray_patches) == 0 or cid >= self.multi_gan.num_clusters:
                        continue

                    # Stack patches into a minibatch
                    g_stack = torch.stack(gray_patches)   # (N, 1, S, S)
                    c_stack = torch.stack(color_patches)  # (N, 3, S, S)

                    losses = self.multi_gan.train_on_cluster(cid, g_stack, c_stack)

                    for kk in batch_losses:
                        batch_losses[kk] += losses.get(kk, 0.0)
                    batch_steps += 1

            if batch_steps > 0:
                for kk in epoch_losses:
                    epoch_losses[kk] += batch_losses[kk] / batch_steps
                n_steps += 1

            # Update progress bar
            if batch_steps > 0:
                pbar.set_postfix({
                    "d": f"{batch_losses['loss_d'] / max(batch_steps, 1):.3f}",
                    "g": f"{batch_losses['loss_g'] / max(batch_steps, 1):.3f}",
                })

        # Average over batches
        if n_steps > 0:
            for kk in epoch_losses:
                epoch_losses[kk] /= n_steps

        return epoch_losses

    def validate(self, dataloader) -> Dict[str, float]:
        """Validate: compute PSNR and SSIM on validation set.

        For efficiency, uses a single forward pass through an arbitrary GAN
        rather than full region-based routing.
        """
        psnr_total = 0.0
        ssim_total = 0.0
        n = 0

        # Use the first available GAN for quick validation
        if self.multi_gan.num_clusters == 0:
            return {"psnr": 0.0, "ssim": 0.0}

        cid = self.multi_gan.cluster_ids[0]

        for gray_batch, color_batch in tqdm(dataloader, desc="Validating"):
            for i in range(gray_batch.size(0)):
                gray = gray_batch[i:i+1]  # (1, 1, H, W)
                color = color_batch[i:i+1]  # (1, 3, H, W)

                pred = self.multi_gan.generate_for_cluster(cid, gray)
                # Normalize to [0, 1]
                pred = (pred + 1.0) / 2.0
                color_norm = color

                psnr_total += compute_psnr(pred, color_norm, max_val=1.0)
                ssim_total += compute_ssim(pred, color_norm, max_val=1.0)
                n += 1

        return {
            "psnr": psnr_total / max(n, 1),
            "ssim": ssim_total / max(n, 1),
        }

    def train(self, train_loader, val_loader=None) -> None:
        """Full training loop.

        Args:
            train_loader: Training dataloader.
            val_loader:   Optional validation dataloader.
        """
        for epoch in range(1, self.epochs + 1):
            # Training
            train_losses = self.train_epoch(train_loader, epoch)

            print(f"Epoch {epoch}/{self.epochs} | "
                  f"D: {train_losses['loss_d']:.4f} | "
                  f"G: {train_losses['loss_g']:.4f} | "
                  f"L1: {train_losses['loss_g_l1']:.4f}")

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                print(f"  Val -> PSNR: {val_metrics['psnr']:.2f} dB | "
                      f"SSIM: {val_metrics['ssim']:.4f}")

            # Save checkpoint
            if epoch % self.save_interval == 0 or epoch == self.epochs:
                ckpt_dir = os.path.join(self.model_dir, f"epoch_{epoch:04d}")
                self.multi_gan.save_all(ckpt_dir)
                print(f"  Saved checkpoint to {ckpt_dir}")

        print("Training complete.")
