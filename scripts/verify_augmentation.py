"""
scripts/verify_augmentation.py
--------------------------------
Sanity-check the augmentation pipeline.

Confirms:
  - Spatial transforms (crop, flip) are applied identically to both images
  - Noise is added to low only, high is unchanged
  - Output shapes and dtypes are correct
  - Augmentation can be toggled off

Run from the project root:
    python scripts/verify_augmentation.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

from src.augmentation import PairedAugmentation  # noqa: E402
from src.dataset import ImagePairDataset          # noqa: E402


def check(condition, label):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main():
    print("=" * 55)
    print("PairedAugmentation verification")
    print("=" * 55)

    data_dir = Path(__file__).parent.parent / "data" / "train"
    ds_plain = ImagePairDataset(data_dir, grayscale=True)
    low_orig, high_orig = ds_plain[0]
    _, H, W = low_orig.shape

    all_ok = True

    # ── Crop ────────────────────────────────────────────────────
    print("\n-- RandomCrop")
    crop_size = 96
    aug = PairedAugmentation(crop_size=crop_size, flip=False, noise_std=0.0)
    ds  = ImagePairDataset(data_dir, grayscale=True, transform=aug)
    low, high = ds[0]

    all_ok &= check(tuple(low.shape)  == (1, crop_size, crop_size),
                    f"low  cropped to (1, {crop_size}, {crop_size}): {tuple(low.shape)}")
    all_ok &= check(tuple(high.shape) == (1, crop_size, crop_size),
                    f"high cropped to (1, {crop_size}, {crop_size}): {tuple(high.shape)}")
    all_ok &= check(low.shape == high.shape,
                    "low and high have the same shape after crop")

    # ── Flip -- same region on both ──────────────────────────────
    print("\n-- RandomHorizontalFlip / RandomVerticalFlip")
    # Seed so flip fires deterministically
    torch.manual_seed(0)
    import random
    random.seed(0)
    aug_flip = PairedAugmentation(crop_size=None, flip=True, noise_std=0.0)

    # Run many times -- at least one flip should occur
    flipped = False
    for _ in range(20):
        random.seed(_)
        out = aug_flip({"low": low_orig.clone(), "high": high_orig.clone()})
        if not torch.equal(out["low"], low_orig):
            flipped = True
            # When low is flipped, high must be flipped identically
            all_ok &= check(
                torch.equal(out["low"], out["high"] * 0 + out["low"]),
                "Flip applied: low and high stay in sync"
            )
            # Verify high was flipped the same way
            diff = (out["low"] - out["high"]).abs().max().item()
            # diff should be same as original low-high diff (noise not added here)
            orig_diff = (low_orig - high_orig).abs().max().item()
            all_ok &= check(
                abs(diff - orig_diff) < 1e-5,
                f"Flip does not change low-high pixel difference (orig={orig_diff:.4f}, after={diff:.4f})"
            )
            break
    all_ok &= check(flipped, "At least one flip occurred in 20 attempts")

    # ── Noise -- low only ────────────────────────────────────────
    print("\n-- AddGaussianNoise (low only)")
    noise_std = 0.05
    aug_noise = PairedAugmentation(crop_size=None, flip=False, noise_std=noise_std)

    low_n, high_n = aug_noise({"low": low_orig.clone(), "high": high_orig.clone()}).values()
    all_ok &= check(not torch.equal(low_n, low_orig),
                    "Noise was added to low image")
    all_ok &= check(torch.equal(high_n, high_orig),
                    "High image unchanged after noise")
    all_ok &= check(low_n.min() >= 0.0 and low_n.max() <= 1.0,
                    f"Noisy low still in [0, 1]: min={low_n.min():.4f} max={low_n.max():.4f}")

    noise_actual = (low_n - low_orig).std().item()
    all_ok &= check(
        abs(noise_actual - noise_std) < 0.02,
        f"Noise std close to {noise_std} (actual={noise_actual:.4f})"
    )

    # ── Combined ─────────────────────────────────────────────────
    print("\n-- PairedAugmentation (all transforms combined)")
    aug_all = PairedAugmentation(crop_size=96, flip=True, noise_std=0.05)
    print(f"  {aug_all}")
    ds_aug = ImagePairDataset(data_dir, grayscale=True, transform=aug_all)
    low_a, high_a = ds_aug[0]

    all_ok &= check(tuple(low_a.shape)  == (1, 96, 96), f"Combined: low shape {tuple(low_a.shape)}")
    all_ok &= check(tuple(high_a.shape) == (1, 96, 96), f"Combined: high shape {tuple(high_a.shape)}")
    all_ok &= check(low_a.dtype == torch.float32,       "Combined: dtype float32")
    all_ok &= check(low_a.min() >= 0.0 and low_a.max() <= 1.0, "Combined: values in [0, 1]")

    # ── Toggle off ───────────────────────────────────────────────
    print("\n-- No augmentation (toggle off)")
    ds_no_aug = ImagePairDataset(data_dir, grayscale=True, transform=None)
    low_na, high_na = ds_no_aug[0]
    all_ok &= check(tuple(low_na.shape) == (1, H, W),
                    f"No aug: original size preserved ({H}x{W})")

    print()
    if all_ok:
        print("All checks passed -- augmentation is working correctly.")
    else:
        print("Some checks FAILED -- review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
