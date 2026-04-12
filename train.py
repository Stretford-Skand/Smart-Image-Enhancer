"""
train.py
---------
Full training loop for ImageEnhancerCNN.

Trains on paired (low, high) image data, evaluates with PSNR and SSIM
on the validation set after each epoch, and saves a loss/metric plot.

Usage:
    python train.py                          # fresh training run
    python train.py --resume                 # resume from last checkpoint
    python train.py --resume outputs/last_checkpoint.pth  # explicit path

Checkpoints:
    outputs/best_model.pth       -- best val PSNR (use for inference)
    outputs/last_checkpoint.pth  -- latest epoch  (use for resume)
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.augmentation import PairedAugmentation
from src.checkpoint import load_checkpoint, save_checkpoint
from src.dataset import ImagePairDataset
from src.metrics import psnr, ssim
from src.model import ImageEnhancerCNN

# ── Config ──────────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir":     "data",
    "epochs":       30,
    "batch_size":   8,
    "lr":           1e-3,
    "num_features": 32,
    "num_blocks":   4,
    "crop_size":    96,
    "noise_std":    0.05,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}
# ────────────────────────────────────────────────────────────────────────────


def make_loaders(cfg):
    train_aug = PairedAugmentation(
        crop_size=cfg["crop_size"],
        flip=True,
        noise_std=cfg["noise_std"],
    )
    train_ds = ImagePairDataset(
        Path(cfg["data_dir"]) / "train",
        grayscale=True,
        transform=train_aug,
    )
    val_ds = ImagePairDataset(
        Path(cfg["data_dir"]) / "val",
        grayscale=True,
        transform=None,    # no augmentation at validation
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=0,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for low, high in loader:
        low, high = low.to(device), high.to(device)
        pred = model(low)
        loss = criterion(pred, high)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for low, high in loader:
            low, high = low.to(device), high.to(device)
            pred = model(low)
            total_loss += criterion(pred, high).item()
            total_psnr += psnr(pred, high).item()
            total_ssim += ssim(pred, high).item()
    n = len(loader)
    return total_loss / n, total_psnr / n, total_ssim / n


def save_plots(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training history", fontsize=12)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_psnr"], color="steelblue")
    axes[1].set_title("Val PSNR (dB)")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history["val_ssim"], color="darkorange")
    axes[2].set_title("Val SSIM")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    path = out_dir / "training_history.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def save_sample_images(model, val_loader, device, out_dir, n=4):
    """Save a grid of low / predicted / high images from the val set."""
    model.eval()
    low_batch, high_batch = next(iter(val_loader))
    low_batch = low_batch[:n].to(device)
    high_batch = high_batch[:n]

    with torch.no_grad():
        pred_batch = model(low_batch).cpu()

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    fig.suptitle("Val samples: Low  |  Predicted  |  High (target)", fontsize=11)
    row_labels = ["Low (input)", "Predicted", "High (target)"]

    for col in range(n):
        for row, img in enumerate([
            low_batch[col].cpu(),
            pred_batch[col],
            high_batch[col],
        ]):
            axes[row, col].imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(row_labels[row], fontsize=9)

    plt.tight_layout()
    path = out_dir / "sample_predictions.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Train ImageEnhancerCNN")
    parser.add_argument(
        "--resume", nargs="?", const="outputs/last_checkpoint.pth",
        metavar="CHECKPOINT",
        help="Resume from a checkpoint. Defaults to outputs/last_checkpoint.pth",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = CONFIG
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    # Data
    train_loader, val_loader = make_loaders(cfg)
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # Model and optimizer
    model = ImageEnhancerCNN(
        in_channels=1,
        num_features=cfg["num_features"],
        num_blocks=cfg["num_blocks"],
    ).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Output dir
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    # Resume or fresh start
    history   = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}
    best_psnr = 0.0
    start_epoch = 1

    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            return
        start_epoch, best_psnr, history = load_checkpoint(
            ckpt_path, model, optimizer, device
        )
        start_epoch += 1   # resume from the next epoch
        print(f"Resumed from {ckpt_path}  (epoch {start_epoch}, best PSNR {best_psnr:.2f} dB)")

    # Training loop
    print()
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'PSNR':>8}  {'SSIM':>8}  {'Time':>6}")
    print("-" * 58)

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr, val_ssim_val = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim_val)

        elapsed = time.time() - t0

        # Save last checkpoint every epoch (for resume)
        save_checkpoint(
            out_dir / "last_checkpoint.pth",
            epoch, model, optimizer, best_psnr, history,
        )

        # Save best checkpoint when PSNR improves
        marker = ""
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(
                out_dir / "best_model.pth",
                epoch, model, optimizer, best_psnr, history,
            )
            marker = " *"

        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{val_psnr:>8.2f}  {val_ssim_val:>8.4f}  {elapsed:>5.1f}s{marker}")

    print()
    print(f"Best val PSNR: {best_psnr:.2f} dB")
    print(f"Best model     -> {out_dir / 'best_model.pth'}")
    print(f"Last checkpoint-> {out_dir / 'last_checkpoint.pth'}")

    # Plots
    plot_path   = save_plots(history, out_dir)
    sample_path = save_sample_images(model, val_loader, device, out_dir)
    print(f"History plot   -> {plot_path}")
    print(f"Sample images  -> {sample_path}")


if __name__ == "__main__":
    main()
