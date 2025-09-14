#!/usr/bin/env python3
"""Test device selection for obstacle avoidance system."""

import torch
import argparse

def test_device_selection():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to use for inference")
    args = parser.parse_args()

    # Device selection logic (same as in main.py)
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Selected device: {device}")
    print(f"Device type: {device.type}")

    # Test tensor creation on the device
    try:
        test_tensor = torch.randn(10, 10).to(device)
        print(f"✅ Successfully created tensor on {device}")
        print(f"Tensor device: {test_tensor.device}")
    except Exception as e:
        print(f"❌ Failed to create tensor on {device}: {e}")

if __name__ == "__main__":
    test_device_selection()
