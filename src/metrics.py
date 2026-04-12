"""
src/metrics.py
---------------
Image quality metrics: PSNR and SSIM.

Both functions operate on float32 tensors in [0, 1].
Both support batched inputs (N, C, H, W) and return the mean over the batch.
"""

import torch
import torch.nn.functional as F


def psnr(pred, target, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio (dB).

    Higher is better. Typical ranges:
        < 30 dB  -- poor
        30-35 dB -- acceptable
        35-40 dB -- good
        > 40 dB  -- excellent

    Args:
        pred:    Predicted tensor  (N, C, H, W), float32 in [0, 1]
        target:  Ground truth      (N, C, H, W), float32 in [0, 1]
        max_val: Maximum pixel value (1.0 for normalised images)

    Returns:
        Mean PSNR over the batch (scalar tensor).
    """
    mse = F.mse_loss(pred, target, reduction="none")
    # Mean over C, H, W -- keep N for per-image PSNR
    mse_per_image = mse.mean(dim=[1, 2, 3])
    # Avoid log(0) if prediction is perfect
    mse_per_image = mse_per_image.clamp(min=1e-10)
    psnr_per_image = 10.0 * torch.log10(max_val ** 2 / mse_per_image)
    return psnr_per_image.mean()


def ssim(pred, target, window_size=11, max_val=1.0):
    """
    Structural Similarity Index (SSIM).

    Higher is better. Range: [-1, 1], typically [0, 1] for natural images.
    Correlates better with human perception than MSE/PSNR.

    Computed per-channel using a Gaussian-weighted window.

    Args:
        pred:        Predicted tensor  (N, C, H, W), float32 in [0, 1]
        target:      Ground truth      (N, C, H, W), float32 in [0, 1]
        window_size: Size of the Gaussian window (default 11)
        max_val:     Maximum pixel value (1.0 for normalised images)

    Returns:
        Mean SSIM over the batch (scalar tensor).
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    channels = pred.shape[1]
    kernel   = _gaussian_kernel(window_size, sigma=1.5, channels=channels)
    kernel   = kernel.to(pred.device)

    pad = window_size // 2

    mu1    = F.conv2d(pred,   kernel, padding=pad, groups=channels)
    mu2    = F.conv2d(target, kernel, padding=pad, groups=channels)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred   * pred,   kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12   = F.conv2d(pred   * target, kernel, padding=pad, groups=channels) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean()


def _gaussian_kernel(size, sigma, channels):
    """Build a normalised Gaussian kernel for conv2d."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.outer(g)
    # Shape: (channels, 1, size, size) for depthwise conv (groups=channels)
    return kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
