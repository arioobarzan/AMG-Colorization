"""PatchGAN discriminator for image colorization.

Classifies NxN overlapping patches as real or fake, which captures
local texture/color consistency.
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator (70x70 receptive field, following pix2pix).

    Input:  RGB image (real or generated) concatenated with grayscale condition.
    Output: (B, 1, H/16, W/16) patch-wise real/fake logits.
    """

    def __init__(
        self,
        in_channels: int = 4,  # grayscale (1) + RGB (3)   OR 3 for RGB only
        base_features: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels

        # C64: no batchnorm on first layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, base_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # C128
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_features, base_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # C256
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_features * 2, base_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # C512
        self.layer4 = nn.Sequential(
            nn.Conv2d(base_features * 4, base_features * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output: 1-channel patch logits
        self.output = nn.Conv2d(base_features * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, img: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            img:       (B, 3, H, W) RGB image.
            condition: (B, 1, H, W) grayscale image. If provided, concat along channel dim.
                       If None, use img only (fewer input channels).

        Returns:
            (B, 1, H/16, W/16) logits.
        """
        if condition is not None:
            x = torch.cat([condition, img], dim=1)
        else:
            x = img

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.output(x)
