"""
src/dataset.py
--------------
Custom PyTorch Dataset for paired image enhancement training.

Expected folder layout:
    data/
      train/
        low/    <-- degraded input images  (PNG / JPG)
        high/   <-- clean target images    (PNG / JPG)
      val/
        low/
        high/

Usage:
    from src.dataset import ImagePairDataset
    from torch.utils.data import DataLoader

    train_ds = ImagePairDataset("data/train")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

    for low, high in train_loader:
        # low:  (N, C, H, W)  float32  [0, 1]
        # high: (N, C, H, W)  float32  [0, 1]
        ...
"""

import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path


# Supported image extensions
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImagePairDataset(Dataset):
    """
    Loads paired (low-quality, high-quality) images from disk.

    Args:
        root:        Path to a split folder, e.g. "data/train".
                     Must contain sub-folders  low/  and  high/  with
                     matching filenames.
        grayscale:   If True, load images as single-channel grayscale.
                     If False, load as 3-channel BGR -> RGB.
        transform:   Optional callable applied to BOTH images after
                     loading (receives a dict {"low": tensor, "high": tensor},
                     returns the same dict). Pass None to skip.
    """

    def __init__(self, root, grayscale=True, transform=None):
        self.root      = Path(root)
        self.grayscale = grayscale
        self.transform = transform

        low_dir  = self.root / "low"
        high_dir = self.root / "high"

        if not low_dir.exists():
            raise FileNotFoundError(f"low/ directory not found: {low_dir}")
        if not high_dir.exists():
            raise FileNotFoundError(f"high/ directory not found: {high_dir}")

        # Collect and sort image paths -- sorted() ensures deterministic order
        self.low_paths  = sorted(
            p for p in low_dir.iterdir() if p.suffix.lower() in _IMG_EXTS
        )
        self.high_paths = sorted(
            p for p in high_dir.iterdir() if p.suffix.lower() in _IMG_EXTS
        )

        if len(self.low_paths) == 0:
            raise RuntimeError(f"No images found in {low_dir}")

        if len(self.low_paths) != len(self.high_paths):
            raise RuntimeError(
                f"Mismatch: {len(self.low_paths)} low images vs "
                f"{len(self.high_paths)} high images in {self.root}"
            )

        # Verify filenames match (same stem, possibly different extension)
        for lp, hp in zip(self.low_paths, self.high_paths):
            if lp.stem != hp.stem:
                raise RuntimeError(
                    f"Filename mismatch: {lp.name} vs {hp.name}. "
                    "low/ and high/ images must have the same filenames."
                )

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low_img  = self._load(self.low_paths[idx])
        high_img = self._load(self.high_paths[idx])

        if self.transform is not None:
            sample = self.transform({"low": low_img, "high": high_img})
            low_img, high_img = sample["low"], sample["high"]

        return low_img, high_img

    def _load(self, path):
        """Load one image as a normalised float32 tensor (C, H, W)."""
        flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        img  = cv2.imread(str(path), flag)

        if img is None:
            raise RuntimeError(f"cv2.imread failed -- file may be corrupt: {path}")

        if not self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalise to [0, 1]
        img = img.astype("float32") / 255.0

        # Add channel dim for grayscale: (H, W) -> (1, H, W)
        # Reorder for colour:            (H, W, 3) -> (3, H, W)
        if self.grayscale:
            tensor = torch.from_numpy(img).unsqueeze(0)
        else:
            tensor = torch.from_numpy(img).permute(2, 0, 1)

        return tensor

    def __repr__(self):
        return (
            f"ImagePairDataset(root={self.root}, "
            f"n={len(self)}, "
            f"grayscale={self.grayscale})"
        )
