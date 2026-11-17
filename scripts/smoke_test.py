"""Build the model, run a few training steps on MNIST, generate a grid.

Proves the pieces fit together; it does not train anything meaningful.
Output grid goes to ``config.output_root / smoke_grid.png`` on the D drive.

    python scripts/smoke_test.py [--steps 3]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgan import MemoryGAN, MemoryGANConfig
from mgan.data import images_only, mnist_loader


def main(steps: int) -> None:
    config = MemoryGANConfig()
    torch.manual_seed(0)

    loader = mnist_loader(config.data_root, config.batch_size)
    model = MemoryGAN(config)
    print(f"memory: {model.memory.n_slots} slots x {config.feature_dim} dims, "
          f"{int(model.memory.values.sum())} real")

    batches = iter(loader)
    for step in range(steps):
        t0 = time.perf_counter()
        losses = model.train_step(images_only(next(batches)))
        print(f"step {step}: " + ", ".join(f"{k}={v:.3f}" for k, v in losses.items())
              + f"  ({time.perf_counter() - t0:.2f}s)")

    images = model.generate(64)
    config.output_root.mkdir(parents=True, exist_ok=True)
    out = config.output_root / "smoke_grid.png"
    save_image(images, out, nrow=8, normalize=True, value_range=(-1, 1))
    print(f"generated {tuple(images.shape)} -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    main(parser.parse_args().steps)
