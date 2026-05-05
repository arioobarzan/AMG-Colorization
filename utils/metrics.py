"""Image quality metrics: PSNR, SSIM, FID."""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scipy import linalg


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        pred:  Predicted image tensor (C, H, W) or (B, C, H, W), range [0, max_val].
        target: Ground truth tensor, same shape as pred.
        max_val: Maximum pixel value (1.0 for [0,1], 255.0 for [0,255]).

    Returns:
        PSNR value in dB.
    """
    mse = F.mse_loss(pred, target, reduction="mean")
    if mse == 0:
        return float("inf")
    return float(20.0 * torch.log10(torch.tensor(max_val)) - 10.0 * torch.log10(mse))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    kernel = kernel_2d.expand(channels, 1, size, size).contiguous()
    return kernel


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """Compute Structural Similarity Index (SSIM).

    Args:
        pred:   Predicted image (B, C, H, W), range [0, max_val].
        target: Ground truth image, same shape.
        window_size: Gaussian window size.
        max_val: Max pixel value.

    Returns:
        Mean SSIM across batch.
    """
    C = pred.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5, C).to(pred.device)
    kernel = kernel.type_as(pred)

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=C) - mu1_mu2

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# FID  (Frechet Inception Distance)
# ---------------------------------------------------------------------------

class InceptionV3FeatureExtractor(nn.Module):
    """Pretrained InceptionV3 truncated to the final average-pooling layer."""

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT, transform_input=False
        )
        # Drop the final classification head; keep up to the avgpool
        inception.fc = nn.Identity()          # 2048-D features
        inception.aux_logits = False
        self.model = inception.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 299, 299) in range [-1, 1]. Returns (B, 2048)."""
        return self.model(x)


def _compute_activation_stats(
    images: torch.Tensor,
    model: InceptionV3FeatureExtractor,
    batch_size: int = 32,
) -> tuple:
    """Compute mean and covariance of Inception features.

    images: (N, 3, H, W) in [0, 1].
    """
    device = next(model.parameters()).device
    upsample = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=False)

    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        # Inception expects [-1, 1]
        batch = batch * 2.0 - 1.0
        batch = upsample(batch)
        with torch.no_grad():
            feat = model(batch)
        feats.append(feat.cpu())
    feats = torch.cat(feats, dim=0).numpy()

    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Compute Frechet Inception Distance.

    Args:
        real_images: (N, 3, H, W) real images in [0, 1].
        fake_images: (N, 3, H, W) generated images in [0, 1].
        device: torch device.

    Returns:
        FID score (lower is better).
    """
    extractor = InceptionV3FeatureExtractor(device)
    mu_real, sigma_real = _compute_activation_stats(real_images, extractor)
    mu_fake, sigma_fake = _compute_activation_stats(fake_images, extractor)

    diff = mu_real - mu_fake
    # Numerical stability via sqrtm
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2.0 * np.trace(covmean))
    return fid
