"""U-Net generator for grayscale-to-RGB colorization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBlock(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, use_bias: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=use_bias),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=use_bias),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block: MaxPool2d + ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsampling block: ConvTranspose2d or Upsample + Conv + skip concat + ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        # After concatenating skip connection, channels double
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetGenerator(nn.Module):
    """U-Net for image colorization.

    Encoder-decoder architecture with skip connections.
    Input:  1-channel grayscale image.
    Output: 3-channel RGB image with tanh activation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_features: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input convolution
        self.in_conv = ConvBlock(in_channels, base_features)

        # Encoder (downsampling path)
        self.down1 = DownBlock(base_features, base_features * 2)      # -> 128
        self.down2 = DownBlock(base_features * 2, base_features * 4)  # -> 256
        self.down3 = DownBlock(base_features * 4, base_features * 8)  # -> 512

        # Bottleneck
        self.bottleneck = ConvBlock(base_features * 8, base_features * 16)  # -> 1024

        # Decoder (upsampling path)
        self.up3 = UpBlock(base_features * 16, base_features * 8, bilinear)
        self.up2 = UpBlock(base_features * 8, base_features * 4, bilinear)
        self.up1 = UpBlock(base_features * 4, base_features * 2, bilinear)
        self.up0 = UpBlock(base_features * 2, base_features, bilinear)

        # Output layer
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_features * 2, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 1, H, W) grayscale tensor.

        Returns:
            (B, 3, H, W) RGB tensor in [-1, 1].
        """
        # Encoder
        e0 = self.in_conv(x)                         # (B, 64, H, W)
        e1 = self.down1(e0)                          # (B, 128, H/2, W/2)
        e2 = self.down2(e1)                          # (B, 256, H/4, W/4)
        e3 = self.down3(e2)                          # (B, 512, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(e3)                       # (B, 1024, H/8, W/8)

        # Decoder with skip connections
        d3 = self.up3(b, e3)                          # (B, 512, H/4, W/4)
        d2 = self.up2(d3, e2)                         # (B, 256, H/2, W/2)
        d1 = self.up1(d2, e1)                         # (B, 128, H, W)
        d0 = self.up0(d1, e0)                         # (B, 64, H, W)

        out = self.out_conv(d0)                       # (B, 3, H, W)
        return out
