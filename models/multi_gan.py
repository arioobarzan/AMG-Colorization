"""Multi-GAN manager: maintains one independent GAN per cluster.

Each cluster gets its own GAN (generator + discriminator) with no parameter
sharing, as specified in the Multi-GAN paper.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np

from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.gan import GAN


class MultiGAN(nn.Module):
    """Container for multiple cluster-specific GANs.

    Provides methods for training individual cluster GANs, saving/loading
    the full ensemble, and generating colorized images by routing regions
    to their assigned GAN.
    """

    def __init__(
        self,
        num_clusters: int,
        device: torch.device = torch.device("cpu"),
        gen_kwargs: Optional[dict] = None,
        disc_kwargs: Optional[dict] = None,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        l1_lambda: float = 100.0,
        gan_lambda: float = 1.0,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.device = device
        self.gan_dict: Dict[int, GAN] = {}

        gen_kwargs = gen_kwargs or {}
        disc_kwargs = disc_kwargs or {}

        for cid in range(num_clusters):
            gen = UNetGenerator(**gen_kwargs)
            disc = PatchGANDiscriminator(**disc_kwargs)
            self.gan_dict[cid] = GAN(
                generator=gen,
                discriminator=disc,
                device=device,
                lr_g=lr_g,
                lr_d=lr_d,
                l1_lambda=l1_lambda,
                gan_lambda=gan_lambda,
            )

    def train_on_cluster(
        self,
        cluster_id: int,
        gray_batch: torch.Tensor,
        color_batch: torch.Tensor,
    ) -> dict:
        """Run one training step on the GAN for a specific cluster.

        Args:
            cluster_id:  Integer cluster index.
            gray_batch:  (B, 1, H, W) grayscale patches.
            color_batch: (B, 3, H, W) RGB ground-truth patches.

        Returns:
            Loss dictionary.
        """
        if cluster_id not in self.gan_dict:
            raise KeyError(f"Cluster ID {cluster_id} not found in MultiGAN.")
        return self.gan_dict[cluster_id].train_step(gray_batch, color_batch)

    def generate_for_cluster(
        self,
        cluster_id: int,
        gray: torch.Tensor,
    ) -> torch.Tensor:
        """Generate colorized output from a specific cluster GAN.

        Args:
            cluster_id: Cluster index.
            gray:       (B, 1, H, W) grayscale input.

        Returns:
            (B, 3, H, W) RGB output in [-1, 1].
        """
        return self.gan_dict[cluster_id].generate(gray)

    def get_cluster_generator(self, cluster_id: int) -> UNetGenerator:
        """Return the generator for a specific cluster."""
        return self.gan_dict[cluster_id].generator

    def save_all(self, directory: str, prefix: str = "gan_cluster") -> None:
        """Save all cluster GAN checkpoints.

        Args:
            directory: Target directory.
            prefix:    Filename prefix.
        """
        os.makedirs(directory, exist_ok=True)
        for cid, gan in self.gan_dict.items():
            path = os.path.join(directory, f"{prefix}_{cid:03d}.pt")
            gan.save(path)

    def load_all(self, directory: str, prefix: str = "gan_cluster") -> None:
        """Load all cluster GAN checkpoints.

        Args:
            directory: Source directory.
            prefix:    Filename prefix.
        """
        for cid in self.gan_dict:
            path = os.path.join(directory, f"{prefix}_{cid:03d}.pt")
            if os.path.exists(path):
                self.gan_dict[cid].load(path)
            else:
                print(f"[WARNING] Checkpoint not found: {path}")

    @property
    def cluster_ids(self) -> List[int]:
        """Return sorted list of cluster IDs."""
        return sorted(self.gan_dict.keys())
