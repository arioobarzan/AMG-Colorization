"""Single GAN wrapper: pairs a U-Net generator with a PatchGAN discriminator."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator


class GAN(nn.Module):
    """A single conditional GAN for colorizing images of a specific cluster.

    Each GAN is trained independently on the subset of regions/images
    assigned to its cluster by the dynamic K-Means algorithm.
    """

    def __init__(
        self,
        generator: UNetGenerator,
        discriminator: PatchGANDiscriminator,
        device: torch.device = torch.device("cpu"),
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        l1_lambda: float = 100.0,
        gan_lambda: float = 1.0,
    ):
        super().__init__()
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device

        # Losses
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Optimizers
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(beta1, beta2)
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2)
        )

        # Hyperparameters
        self.l1_lambda = l1_lambda
        self.gan_lambda = gan_lambda

    def train_step(
        self,
        gray: torch.Tensor,
        color: torch.Tensor,
    ) -> dict:
        """Single training step: update discriminator then generator.

        Args:
            gray:  (B, 1, H, W) grayscale input.
            color: (B, 3, H, W) ground truth RGB.

        Returns:
            Dict of loss values for logging.
        """
        B = gray.size(0)
        gray = gray.to(self.device)
        color = color.to(self.device)

        real_label = torch.ones(B, 1, 1, 1, device=self.device) * 0.9  # label smoothing
        fake_label = torch.zeros(B, 1, 1, 1, device=self.device)

        # ----------------------------------------------------------------
        # 1. Train Discriminator
        # ----------------------------------------------------------------
        self.opt_d.zero_grad()

        # Real
        pred_real = self.discriminator(color, gray)
        loss_d_real = self.bce_loss(pred_real, real_label.expand_as(pred_real))

        # Fake
        with torch.no_grad():
            fake_color = self.generator(gray)
        pred_fake = self.discriminator(fake_color.detach(), gray)
        loss_d_fake = self.bce_loss(pred_fake, fake_label.expand_as(pred_fake))

        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        self.opt_d.step()

        # ----------------------------------------------------------------
        # 2. Train Generator
        # ----------------------------------------------------------------
        self.opt_g.zero_grad()

        fake_color = self.generator(gray)
        pred_fake = self.discriminator(fake_color, gray)
        loss_g_gan = self.bce_loss(pred_fake, real_label.expand_as(pred_fake))
        loss_g_l1 = self.l1_loss(fake_color, color)
        loss_g = self.gan_lambda * loss_g_gan + self.l1_lambda * loss_g_l1
        loss_g.backward()
        self.opt_g.step()

        return {
            "loss_d": float(loss_d),
            "loss_g": float(loss_g),
            "loss_g_l1": float(loss_g_l1),
            "loss_g_gan": float(loss_g_gan),
        }

    def generate(self, gray: torch.Tensor) -> torch.Tensor:
        """Colorize a grayscale image.

        Args:
            gray: (B, 1, H, W) tensor.

        Returns:
            (B, 3, H, W) RGB tensor in [-1, 1].
        """
        self.generator.eval()
        with torch.no_grad():
            out = self.generator(gray.to(self.device))
        return out.cpu()

    def save(self, path: str) -> None:
        """Save generator and discriminator state."""
        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load generator and discriminator state."""
        ckpt = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
