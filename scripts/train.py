"""Optional short training loop.  Not needed for the baseline.

    python scripts/train.py --steps 500 --log-every 50

Saves a sample grid and a checkpoint (networks + memory, no optimisers)
under ``config.output_root`` on the D drive.
"""

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgan import MemoryGAN, MemoryGANConfig
from mgan.data import images_only, mnist_loader


def main(steps: int, log_every: int) -> None:
    config = MemoryGANConfig()
    torch.manual_seed(0)
    loader = mnist_loader(config.data_root, config.batch_size)
    model = MemoryGAN(config)

    step = 0
    while step < steps:
        for batch in loader:
            losses = model.train_step(images_only(batch))
            step += 1
            if step % log_every == 0:
                print(f"step {step}: " + ", ".join(f"{k}={v:.3f}" for k, v in losses.items()))
            if step >= steps:
                break

    config.output_root.mkdir(parents=True, exist_ok=True)
    save_image(model.generate(64), config.output_root / f"samples_{steps}.png",
               nrow=8, normalize=True, value_range=(-1, 1))
    torch.save(model.state_dict(), config.output_root / f"memorygan_{steps}.pt")
    print(f"saved samples and checkpoint to {config.output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    main(args.steps, args.log_every)
