"""
src/checkpoint.py
------------------
Checkpoint save and load utilities.

Two checkpoint files are maintained during training:

    best_model.pth      -- saved whenever val PSNR improves
                           used for inference after training

    last_checkpoint.pth -- saved at the end of every epoch
                           used to resume an interrupted training run

Both store the full training state so a resumed run is identical
to an uninterrupted one:
    - epoch
    - model state_dict
    - optimizer state_dict
    - best_psnr seen so far
    - loss/metric history up to this point
"""

import torch


def save_checkpoint(path, epoch, model, optimizer, best_psnr, history):
    """
    Save full training state to disk.

    Args:
        path:       Destination file path (Path or str).
        epoch:      Current epoch number (int).
        model:      nn.Module -- state_dict is saved, not the full object.
        optimizer:  Optimizer -- state_dict is saved.
        best_psnr:  Best validation PSNR seen so far (float).
        history:    Dict of lists recording per-epoch metrics.
    """
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_psnr":      best_psnr,
        "history":        history,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    """
    Load training state from a checkpoint file.

    Restores model weights, optimizer state, and training history so
    training can resume exactly where it left off.

    Args:
        path:      Checkpoint file path (Path or str).
        model:     nn.Module -- weights are loaded in-place.
        optimizer: Optimizer -- state is loaded in-place.
        device:    torch.device -- maps tensors to the correct device.

    Returns:
        epoch (int), best_psnr (float), history (dict)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    epoch     = checkpoint["epoch"]
    best_psnr = checkpoint["best_psnr"]
    history   = checkpoint["history"]

    return epoch, best_psnr, history
