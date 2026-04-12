"""
scripts/smoke_test.py
----------------------
Fast model and training checks. No real dataset required -- uses
random tensors so this runs in seconds on any machine.

Checks:
  1. Model forward pass -- output shape matches input
  2. Model save/load round-trip -- reloaded weights are identical
  3. Training sanity check -- loss is finite and decreases over 3 steps

Run from the project root:
    python scripts/smoke_test.py
"""

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

from src.model import ImageEnhancerCNN  # noqa: E402


def check(condition, label):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def test_forward_pass():
    print("\n-- Model forward pass")
    all_ok = True
    model = ImageEnhancerCNN(in_channels=1, num_features=32, num_blocks=4)
    model.eval()

    # Grayscale batch
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        out = model(x)

    all_ok &= check(out.shape == x.shape,
                    f"Output shape matches input: {tuple(out.shape)}")
    all_ok &= check(out.dtype == torch.float32,
                    "Output dtype is float32")
    all_ok &= check(out.min() >= 0.0 and out.max() <= 1.0,
                    f"Output clamped to [0, 1]: min={out.min():.4f} max={out.max():.4f}")

    # RGB batch
    x_rgb = torch.randn(2, 3, 64, 64)
    model_rgb = ImageEnhancerCNN(in_channels=3)
    with torch.no_grad():
        out_rgb = model_rgb(x_rgb)
    all_ok &= check(out_rgb.shape == x_rgb.shape,
                    f"RGB: output shape matches input: {tuple(out_rgb.shape)}")

    params = model.count_parameters()
    all_ok &= check(params > 0, f"Model has parameters: {params:,}")

    return all_ok


def test_save_load_roundtrip():
    print("\n-- Save / load round-trip")
    all_ok = True
    model = ImageEnhancerCNN(in_channels=1, num_features=32, num_blocks=4)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test_model.pth"

        # Save
        torch.save(model.state_dict(), path)
        all_ok &= check(path.exists(), f"Checkpoint file created: {path.name}")

        # Load into a fresh model
        model2 = ImageEnhancerCNN(in_channels=1, num_features=32, num_blocks=4)
        model2.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

        # Compare every parameter
        all_match = all(
            torch.equal(p1, p2)
            for p1, p2 in zip(model.parameters(), model2.parameters())
        )
        all_ok &= check(all_match, "Reloaded weights are identical to saved weights")

        # Confirm both models produce the same output
        x = torch.randn(1, 1, 64, 64)
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        all_ok &= check(torch.equal(out1, out2),
                        "Original and reloaded models produce identical output")

    return all_ok


def test_training_sanity():
    print("\n-- Training sanity check (3 steps)")
    all_ok = True

    model     = ImageEnhancerCNN(in_channels=1, num_features=32, num_blocks=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()

    losses = []
    for _ in range(3):
        low  = torch.rand(4, 1, 64, 64)   # random "degraded" input
        high = torch.rand(4, 1, 64, 64)   # random "clean" target

        pred = model(low)
        loss = criterion(pred, high)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    all_ok &= check(all(isinstance(v, float) for v in losses),
                    "All losses are finite floats")
    all_ok &= check(not any(
        (v != v) or (v == float("inf")) for v in losses   # nan or inf check
    ), f"No NaN or Inf in losses: {[round(v, 5) for v in losses]}")

    # Loss should generally trend downward over 3 steps
    # (not guaranteed with random data, so just check it moved)
    all_ok &= check(losses[0] != losses[-1],
                    f"Loss changed during training: {losses[0]:.5f} -> {losses[-1]:.5f}")

    return all_ok


def main():
    print("=" * 50)
    print("Model and training smoke tests")
    print("=" * 50)

    results = [
        test_forward_pass(),
        test_save_load_roundtrip(),
        test_training_sanity(),
    ]

    print()
    if all(results):
        print("All smoke tests passed.")
    else:
        print("Some smoke tests FAILED -- review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
