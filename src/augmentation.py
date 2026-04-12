"""
src/augmentation.py
--------------------
Augmentation transforms for paired (low, high) image training.

All transforms operate on a sample dict  {"low": tensor, "high": tensor}
where each tensor is float32 (C, H, W) in [0, 1].

Spatial transforms (flip, crop) are applied identically to both images
so the low/high pairing is preserved.

Noise is applied to the low image only -- it simulates additional
degradation and prevents the model from overfitting to a fixed noise level.

Usage:
    from src.augmentation import PairedAugmentation
    from src.dataset import ImagePairDataset

    transform = PairedAugmentation(
        crop_size=96,
        flip=True,
        noise_std=0.05,   # in [0, 1] scale
    )
    ds = ImagePairDataset("data/train", transform=transform)
"""

import torch
import random


class RandomCrop:
    """
    Crop both images to the same randomly chosen region.

    Args:
        size: Output crop size (square). Both H and W will equal size.
              Must be <= the smallest dimension of the input images.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        low, high = sample["low"], sample["high"]

        _, H, W = low.shape

        if H < self.size or W < self.size:
            raise ValueError(
                f"Crop size {self.size} is larger than image size ({H}x{W})"
            )

        # Pick one random top-left corner, apply to both
        top  = random.randint(0, H - self.size)
        left = random.randint(0, W - self.size)

        low  = low[:,  top:top + self.size, left:left + self.size]
        high = high[:, top:top + self.size, left:left + self.size]

        return {"low": low, "high": high}


class RandomHorizontalFlip:
    """Flip both images horizontally with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return {
                "low":  torch.flip(sample["low"],  dims=[2]),
                "high": torch.flip(sample["high"], dims=[2]),
            }
        return sample


class RandomVerticalFlip:
    """Flip both images vertically with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return {
                "low":  torch.flip(sample["low"],  dims=[1]),
                "high": torch.flip(sample["high"], dims=[1]),
            }
        return sample


class AddGaussianNoise:
    """
    Add Gaussian noise to the low image only.

    The high image is untouched -- it remains the clean target.

    Args:
        std: Standard deviation of noise in [0, 1] pixel scale.
             Typical values: 0.02 (subtle) to 0.1 (strong).
    """

    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, sample):
        low  = sample["low"]
        noise = torch.randn_like(low) * self.std
        return {
            "low":  (low + noise).clamp(0.0, 1.0),
            "high": sample["high"],
        }


class PairedAugmentation:
    """
    Composes augmentation transforms for a paired image sample.

    Applies transforms in order:
        1. RandomCrop        (if crop_size is set)
        2. RandomHorizontalFlip  (if flip=True)
        3. RandomVerticalFlip    (if flip=True)
        4. AddGaussianNoise      (if noise_std > 0)

    Args:
        crop_size:  Square crop size. None to skip cropping.
        flip:       Enable random horizontal and vertical flips.
        noise_std:  Std of Gaussian noise added to low image. 0 to disable.
    """

    def __init__(self, crop_size=None, flip=True, noise_std=0.0):
        self.transforms = []

        if crop_size is not None:
            self.transforms.append(RandomCrop(crop_size))

        if flip:
            self.transforms.append(RandomHorizontalFlip(p=0.5))
            self.transforms.append(RandomVerticalFlip(p=0.5))

        if noise_std > 0:
            self.transforms.append(AddGaussianNoise(std=noise_std))

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        parts = [t.__class__.__name__ for t in self.transforms]
        return f"PairedAugmentation([{', '.join(parts)}])"
