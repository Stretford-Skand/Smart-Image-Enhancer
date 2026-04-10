"""
scripts/explore_dataset.py
---------------------------
Interactive walkthrough of the dataset pipeline.
Shows what the code is doing at each step, with visuals.

Run from the project root:
    python scripts/explore_dataset.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

from src.dataset import ImagePairDataset  # noqa: E402

# ─────────────────────────────────────────────────────────────
# STEP 1: What is on disk?
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 -- What is on disk?")
print("=" * 60)

data_root = Path(__file__).parent.parent / "data"
for split in ("train", "val"):
    low_files  = sorted((data_root / split / "low").glob("*.png"))
    high_files = sorted((data_root / split / "high").glob("*.png"))
    print(f"  {split}/low/  -- {len(low_files)} files  (e.g. {low_files[0].name})")
    print(f"  {split}/high/ -- {len(high_files)} files  (e.g. {high_files[0].name})")

# Load one raw image with OpenCV to show what it looks like before PyTorch
raw_high = cv2.imread(str(data_root / "train" / "high" / "0000.png"), cv2.IMREAD_GRAYSCALE)
raw_low  = cv2.imread(str(data_root / "train" / "low"  / "0000.png"), cv2.IMREAD_GRAYSCALE)

print()
print(f"  Raw OpenCV image (high): shape={raw_high.shape}  dtype={raw_high.dtype}  "
      f"min={raw_high.min()}  max={raw_high.max()}")
print(f"  Raw OpenCV image (low):  shape={raw_low.shape}   dtype={raw_low.dtype}  "
      f"min={raw_low.min()}  max={raw_low.max()}")
print()
print("  OpenCV gives us a (H, W) uint8 array with values in [0, 255].")
print("  PyTorch needs (C, H, W) float32 with values in [0, 1].")
print("  The Dataset handles that conversion automatically.")

# ─────────────────────────────────────────────────────────────
# STEP 2: What does the Dataset return?
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2 -- What does ImagePairDataset return?")
print("=" * 60)

ds = ImagePairDataset(data_root / "train", grayscale=True)
print(f"  {ds}")
print(f"  len(ds) = {len(ds)}")
print()

low, high = ds[0]
print("  ds[0] returns two tensors:")
print(f"    low  -- shape={tuple(low.shape)}  dtype={low.dtype}  "
      f"min={low.min():.3f}  max={low.max():.3f}")
print(f"    high -- shape={tuple(high.shape)}  dtype={high.dtype}  "
      f"min={high.min():.3f}  max={high.max():.3f}")
print()
print("  Shape (1, 128, 128) = (C=1 channel, H=128, W=128)")
print("  float32 in [0, 1] -- ready for the CNN, no further conversion needed.")

# ─────────────────────────────────────────────────────────────
# STEP 3: What does a DataLoader batch look like?
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3 -- What does a DataLoader batch look like?")
print("=" * 60)

loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
batch_low, batch_high = next(iter(loader))

print("  batch_size = 8")
print(f"  batch_low  shape: {tuple(batch_low.shape)}   (N, C, H, W)")
print(f"  batch_high shape: {tuple(batch_high.shape)}   (N, C, H, W)")
print()
print("  Each training step receives one batch.")
print("  The model sees all 8 images in parallel -- that is what N=8 means.")
print()
print(f"  Batches per epoch: {len(loader)}  ({len(ds)} images / batch_size 8)")

# ─────────────────────────────────────────────────────────────
# STEP 4: How much noise was added?
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4 -- Noise statistics (what the model has to learn to remove)")
print("=" * 60)

differences = []
for i in range(min(20, len(ds))):
    l, h = ds[i]
    diff = (l - h).abs()
    differences.append(diff.mean().item())

avg_diff = np.mean(differences)
print(f"  Mean absolute pixel difference (low vs high): {avg_diff:.4f}")
print(f"  In [0,255] scale: {avg_diff * 255:.1f} out of 255")
print()
print("  This is the average error the model starts with (before training).")
print("  After training, the model output should be close to the high-quality image.")

# ─────────────────────────────────────────────────────────────
# STEP 5: Visual -- sample pairs
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 5 -- Visualising sample pairs")
print("=" * 60)

n_samples = 4
fig, axes = plt.subplots(3, n_samples, figsize=(14, 7))
fig.suptitle(
    "Dataset sample pairs  |  Row 1: Low (noisy)  |  Row 2: High (clean)  |  Row 3: Difference",
    fontsize=11
)

for col in range(n_samples):
    low_t, high_t = ds[col * 10]   # spread across the dataset

    low_np  = low_t[0].numpy()
    high_np = high_t[0].numpy()
    diff_np = np.abs(low_np - high_np)

    axes[0, col].imshow(low_np,  cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"Low (noisy) #{col * 10}", fontsize=9)
    axes[0, col].axis("off")

    axes[1, col].imshow(high_np, cmap="gray", vmin=0, vmax=1)
    axes[1, col].set_title(f"High (clean) #{col * 10}", fontsize=9)
    axes[1, col].axis("off")

    axes[2, col].imshow(diff_np, cmap="hot", vmin=0, vmax=0.4)
    axes[2, col].set_title(f"Diff (noise) avg={diff_np.mean():.3f}", fontsize=9)
    axes[2, col].axis("off")

plt.tight_layout()
out_path = Path(__file__).parent.parent / "scripts" / "explore_dataset.png"
plt.savefig(out_path, dpi=120)
plt.show()
print(f"  Plot saved -> {out_path}")

# ─────────────────────────────────────────────────────────────
# STEP 6: Variety of image patterns
# ─────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 8, figsize=(16, 5))
fig2.suptitle("All 4 synthetic patterns -- high (top) and low/noisy (bottom)", fontsize=11)

sample_indices = [0, 1, 2, 3, 20, 21, 22, 23]
for col, idx in enumerate(sample_indices):
    low_t, high_t = ds[idx]
    axes2[0, col].imshow(high_t[0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes2[0, col].set_title(f"#{idx} clean", fontsize=8)
    axes2[0, col].axis("off")
    axes2[1, col].imshow(low_t[0].numpy(),  cmap="gray", vmin=0, vmax=1)
    axes2[1, col].set_title(f"#{idx} noisy", fontsize=8)
    axes2[1, col].axis("off")

plt.tight_layout()
out_path2 = Path(__file__).parent.parent / "scripts" / "explore_patterns.png"
plt.savefig(out_path2, dpi=120)
plt.show()
print(f"  Pattern variety plot saved -> {out_path2}")

print()
print("=" * 60)
print("Summary")
print("=" * 60)
print("  Disk:       PNG files in data/train/low and data/train/high")
print("  Dataset:    loads pairs, converts uint8 (H,W) -> float32 (1,H,W)")
print("  DataLoader: stacks N pairs into (N, 1, H, W) batches")
print("  Task:       model receives low, must predict high")
print("  Next:       Issue #2 -- add augmentation transforms")
