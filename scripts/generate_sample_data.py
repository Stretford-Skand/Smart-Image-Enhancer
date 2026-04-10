"""
scripts/generate_sample_data.py
--------------------------------
Generates synthetic paired (low, high) image data for testing the pipeline
without needing a real dataset download.

High-quality images: clean synthetic patterns (gradients, shapes)
Low-quality images:  high-quality + Gaussian noise (simulates denoising task)

Run from the project root:
    python scripts/generate_sample_data.py

Output:
    data/train/low/   -- 80 noisy images
    data/train/high/  -- 80 clean images
    data/val/low/     -- 20 noisy images
    data/val/high/    -- 20 clean images
"""

import cv2
import numpy as np
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
TRAIN_COUNT  = 80
VAL_COUNT    = 20
IMG_SIZE     = 128        # H and W (square)
NOISE_STD    = 30         # Gaussian noise std in [0, 255] range
SEED         = 42
# ───────────────────────────────────────────────────────────────────────────


def make_clean_image(rng, size):
    """Generate one synthetic clean grayscale image (H, W), uint8."""
    pattern = rng.integers(0, 4)

    if pattern == 0:
        # Horizontal gradient
        img = np.tile(np.linspace(30, 220, size, dtype=np.float32), (size, 1))

    elif pattern == 1:
        # Vertical gradient
        img = np.tile(np.linspace(30, 220, size, dtype=np.float32), (size, 1)).T

    elif pattern == 2:
        # Radial gradient from centre
        y, x = np.ogrid[:size, :size]
        cx, cy = size // 2, size // 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        img  = np.clip(220 - dist * (190 / (size // 2)), 30, 220).astype(np.float32)

    else:
        # Smooth random blobs (blurred noise)
        raw = rng.standard_normal((size, size)).astype(np.float32)
        img = cv2.GaussianBlur(raw, (0, 0), sigmaX=size // 8)
        lo, hi = img.min(), img.max()
        img = (img - lo) / (hi - lo + 1e-8) * 190 + 30

    return np.clip(img, 0, 255).astype(np.uint8)


def add_noise(clean, rng, std):
    """Add Gaussian noise and clamp to [0, 255]."""
    noise = rng.normal(0, std, clean.shape).astype(np.float32)
    noisy = clean.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_split(split, count, rng, base_dir):
    low_dir  = base_dir / split / "low"
    high_dir = base_dir / split / "high"
    low_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        filename = f"{i:04d}.png"
        clean    = make_clean_image(rng, IMG_SIZE)
        noisy    = add_noise(clean, rng, NOISE_STD)

        cv2.imwrite(str(high_dir / filename), clean)
        cv2.imwrite(str(low_dir  / filename), noisy)

    print(f"  {split}: {count} pairs -> {base_dir / split}")


def main():
    rng      = np.random.default_rng(SEED)
    base_dir = Path(__file__).parent.parent / "data"

    print("Generating synthetic paired image data...")
    generate_split("train", TRAIN_COUNT, rng, base_dir)
    generate_split("val",   VAL_COUNT,   rng, base_dir)

    total = TRAIN_COUNT + VAL_COUNT
    print(f"Done -- {total} image pairs ({IMG_SIZE}x{IMG_SIZE} grayscale)")
    print(f"Noise std: {NOISE_STD}  (out of 255)")


if __name__ == "__main__":
    main()
