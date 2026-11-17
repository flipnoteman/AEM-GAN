r"""Configuration for the memoryGAN baseline.

One dataclass holds every size, hyper-parameter and path so that scripts and
tests build the model from a single object.  Code lives on C:, everything the
model writes (datasets, sample grids, checkpoints) goes under ``D:\M-GAN``.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemoryGANConfig:
    # --- data -----------------------------------------------------------
    image_size: int = 28          # MNIST is 28x28
    channels: int = 1             # grayscale
    batch_size: int = 64

    # --- memory ---------------------------------------------------------
    memory_slots: int = 4096      # N: number of key/value slots
    feature_dim: int = 128        # D: size of encoder features and memory keys
    neighbours: int = 16          # k: slots consulted per lookup
    kappa: float = 10.0           # sharpness of the posterior over neighbours

    # --- networks -------------------------------------------------------
    latent_dim: int = 64          # Z: generator noise size
    base_channels: int = 32       # width of the first conv layer in both nets

    # --- training -------------------------------------------------------
    lr: float = 2e-4
    betas: tuple = (0.5, 0.999)   # Adam betas, the usual GAN setting
    mi_weight: float = 1.0        # weight of the memory-conditional (MI) term

    # --- environment ----------------------------------------------------
    device: str = "cpu"
    data_root: Path = field(default_factory=lambda: Path(r"D:\M-GAN\data"))
    output_root: Path = field(default_factory=lambda: Path(r"D:\M-GAN\outputs"))
