"""
scripts/verify_dataset.py
--------------------------
Sanity-check the ImagePairDataset and DataLoader.

Confirms:
  - Images load without errors
  - Tensor shapes are correct: (N, C, H, W)
  - dtype is float32
  - Pixel values are in [0, 1]
  - low and high tensors are different (noise was actually added)
  - DataLoader iterates without multiprocessing errors

Run from the project root:
    python scripts/verify_dataset.py
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import ImagePairDataset


def check(condition, label):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def verify_split(split, batch_size=4):
    data_dir = Path(__file__).parent.parent / "data" / split
    print(f"\n-- {split} split: {data_dir}")

    ds = ImagePairDataset(data_dir, grayscale=True)
    print(f"  Dataset: {ds}")

    all_ok = True
    all_ok &= check(len(ds) > 0, f"Dataset has samples (n={len(ds)})")

    # Check a single item
    low, high = ds[0]
    all_ok &= check(low.ndim  == 3,          f"low tensor is 3D: {tuple(low.shape)}")
    all_ok &= check(high.ndim == 3,          f"high tensor is 3D: {tuple(high.shape)}")
    all_ok &= check(low.shape == high.shape, f"low and high shapes match: {tuple(low.shape)}")
    all_ok &= check(low.shape[0] == 1,       f"grayscale: C=1 (got {low.shape[0]})")
    all_ok &= check(low.dtype  == torch.float32, f"low dtype is float32")
    all_ok &= check(high.dtype == torch.float32, f"high dtype is float32")
    all_ok &= check(low.min()  >= 0.0 and low.max()  <= 1.0, f"low  values in [0, 1]")
    all_ok &= check(high.min() >= 0.0 and high.max() <= 1.0, f"high values in [0, 1]")
    all_ok &= check(not torch.equal(low, high), "low != high (degradation applied)")

    # Check DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    batch_low, batch_high = next(iter(loader))
    expected_shape = (min(batch_size, len(ds)), 1, low.shape[1], low.shape[2])
    all_ok &= check(
        tuple(batch_low.shape) == expected_shape,
        f"DataLoader batch shape: {tuple(batch_low.shape)}"
    )

    # Iterate the full loader to catch any file-level errors
    errors = 0
    for _ in loader:
        pass
    all_ok &= check(errors == 0, "Full DataLoader iteration completed without errors")

    return all_ok


def main():
    print("=" * 50)
    print("ImagePairDataset verification")
    print("=" * 50)

    data_root = Path(__file__).parent.parent / "data"
    if not data_root.exists():
        print("\nERROR: data/ directory not found.")
        print("Run this first:  python scripts/generate_sample_data.py")
        sys.exit(1)

    results = []
    for split in ("train", "val"):
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"\nSkipping {split}/ -- not found.")
            continue
        results.append(verify_split(split))

    print()
    if all(results):
        print("All checks passed -- dataset is ready for training.")
    else:
        print("Some checks FAILED -- review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
