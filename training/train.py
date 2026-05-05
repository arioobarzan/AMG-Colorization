"""Main training entry point for Multi-GAN colorization.

Usage:
    python training/train.py --config configs/default.yaml
"""

import os
import sys
import argparse
import yaml

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.device import get_device
from data.dataset_loader import create_dataloader
from models.multi_gan import MultiGAN
from training.trainer import MultiGANTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-GAN for Image Colorization")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cuda, cpu."
    )
    parser.add_argument(
        "--num_clusters", type=int, default=None,
        help="Override number of clusters (GANs). If None, determined dynamically."
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    device = get_device(args.device or config.get("device", "auto"))
    print(f"[INFO] Using device: {device}")

    # Data
    data_cfg = config.get("data", {})
    train_loader, val_loader = create_dataloader(
        dataset_type=data_cfg.get("dataset", "cifar10"),
        train_path=data_cfg.get("train_path", "./data"),
        val_path=data_cfg.get("val_path", "./data"),
        image_size=tuple(data_cfg.get("image_size", [256, 256])),
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 4),
    )

    # Determine number of clusters (GANs)
    if args.num_clusters is not None:
        num_clusters = args.num_clusters
    else:
        # Use the max allowed K as the pool size; clustering at runtime may
        # use fewer, but unused GANs simply have no data.
        clust_cfg = config.get("clustering", {})
        num_clusters = clust_cfg.get("k_max", 64)
    print(f"[INFO] Initializing Multi-GAN with {num_clusters} cluster GANs")

    # Build Multi-GAN
    gen_cfg = config.get("generator", {})
    disc_cfg = config.get("discriminator", {})
    train_cfg = config.get("training", {})

    multi_gan = MultiGAN(
        num_clusters=num_clusters,
        device=device,
        gen_kwargs=gen_cfg,
        disc_kwargs=disc_cfg,
        lr_g=train_cfg.get("lr_g", 2e-4),
        lr_d=train_cfg.get("lr_d", 2e-4),
        l1_lambda=train_cfg.get("l1_lambda", 100.0),
        gan_lambda=train_cfg.get("gan_lambda", 1.0),
    )

    # Trainer
    trainer = MultiGANTrainer(
        multi_gan=multi_gan,
        config=config,
        device=device,
    )

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
