"""
src/model.py
-------------
ImageEnhancerCNN -- lightweight residual CNN for image enhancement.

Architecture:
    Input (C, H, W)
        |
    Head: ConvBNReLU(C -> num_features)
        |
    Body: num_blocks x ResidualBlock(num_features)
        |
    Tail: Conv2d(num_features -> C)
        |
    Global residual: output = input + tail
        |
    Output (C, H, W)  -- same spatial size as input throughout
"""

import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Conv2d (bias=False) -> BatchNorm2d -> ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              padding=kernel_size // 2,
                              bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Two ConvBNReLU layers with a residual skip connection."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(channels, channels),
            ConvBNReLU(channels, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ImageEnhancerCNN(nn.Module):
    """
    Image-to-image enhancement network.

    Args:
        in_channels:  Number of input/output channels. 1=grayscale, 3=RGB.
        num_features: Number of feature channels in the residual body.
        num_blocks:   Number of residual blocks.
    """

    def __init__(self, in_channels=1, num_features=32, num_blocks=4):
        super().__init__()
        self.head = ConvBNReLU(in_channels, num_features)
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        self.tail = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.head(x)
        enhanced = self.body(features)
        delta    = self.tail(enhanced)
        return (x + delta).clamp(0.0, 1.0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
