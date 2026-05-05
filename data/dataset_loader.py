"""Dataset loaders for image colorization.

Supports:
- ImageFolderDataset: generic folder of RGB images
- CIFAR10Dataset:     CIFAR-10 wrapped for colorization
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.datasets as tv_datasets


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _default_transform(image_size: Tuple[int, int]) -> T.Compose:
    """Default training transform: resize, random flip, to tensor."""
    return T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),                   # -> [0, 1] float
    ])


def _val_transform(image_size: Tuple[int, int]) -> T.Compose:
    """Validation transform: resize, to tensor (no augmentation)."""
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Image Folder Dataset
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """Colorization dataset from a folder of RGB images.

    Input:  grayscale version of the image.
    Target: original RGB image.

    Args:
        root_dir:  Path to folder containing .jpg/.png/.jpeg files.
        image_size: (H, W) target size.
        transform:  Optional additional transform (applied after ToTensor).
    """

    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform

        self.image_paths: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            self.image_paths.extend(sorted(self.root_dir.glob(ext)))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {root_dir}")

        self.base_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (grayscale, rgb) pair.

        grayscale: (1, H, W) in [0, 1]
        rgb:       (3, H, W) in [0, 1]
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")
        rgb = self.base_transform(img)  # (3, H, W), [0, 1]

        if self.transform:
            # Re-apply transform on the RGB tensor (e.g., random flip)
            # Using PIL for consistent random seed
            pass

        # Convert to grayscale via luminance formula
        gray = 0.299 * rgb[0:1] + 0.587 * rgb[1:2] + 0.114 * rgb[2:3]  # (1, H, W)

        return gray, rgb


# ---------------------------------------------------------------------------
# CIFAR-10 Dataset
# ---------------------------------------------------------------------------

class CIFAR10Dataset(Dataset):
    """CIFAR-10 wrapped for colorization: label is ignored, image is target."""

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.cifar = tv_datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )
        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.cifar)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (grayscale, rgb) pair."""
        img_pil, _ = self.cifar[idx]  # PIL image
        rgb = self.to_tensor(img_pil)  # (3, 32, 32), [0, 1]

        # Grayscale via luminance
        gray = 0.299 * rgb[0:1] + 0.587 * rgb[1:2] + 0.114 * rgb[2:3]

        return gray, rgb


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloader(
    dataset_type: str = "image_folder",
    train_path: str = "./data/train",
    val_path: str = "./data/val",
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 16,
    num_workers: int = 4,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Factory to create train and validation dataloaders.

    Args:
        dataset_type: "image_folder" or "cifar10".
        train_path:   Training data path.
        val_path:     Validation data path.
        image_size:   (H, W) resize target.
        batch_size:   Batch size.
        num_workers:  Data loading workers.
        download:     If True, download CIFAR-10.

    Returns:
        (train_loader, val_loader)
    """
    train_transform = _default_transform(image_size)
    val_transform = _val_transform(image_size)

    if dataset_type == "cifar10":
        train_ds = CIFAR10Dataset(
            root=train_path, train=True, download=download,
            transform=train_transform,
        )
        val_ds = CIFAR10Dataset(
            root=val_path, train=False, download=download,
            transform=val_transform,
        )
    elif dataset_type == "image_folder":
        train_ds = ImageFolderDataset(
            root_dir=train_path, image_size=image_size,
            transform=train_transform,
        )
        val_ds = ImageFolderDataset(
            root_dir=val_path, image_size=image_size,
            transform=val_transform,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
