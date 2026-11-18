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
    # Paper / authors' MNIST settings unless noted.
    memory_slots: int = 4096      # N: number of key/value slots
    feature_dim: int = 256        # D: size of encoder features and memory keys
    neighbours: int = 128         # k: candidate slots per write (and per lookup if not dense)
    kappa: float = 10.0           # vMF concentration (paper: 1; 10 tuned here, FD 675 -> 453 at 1.5k steps)
    hist_decay: float = 0.5       # alpha: histogram decay per EM update
    dense_posterior: bool = False # p(real|q) over the top-k (authors' code) or all slots; top-k: FD 47 vs 85

    # --- networks -------------------------------------------------------
    latent_dim: int = 16          # Z: generator noise size (paper: 2; 16 tuned here, FD 453 -> 61)
    base_channels: int = 64       # width of the first conv layer in both nets (paper f_dim: 64)

    # --- training -------------------------------------------------------
    lr: float = 2e-4
    betas: tuple = (0.5, 0.999)   # Adam betas, the usual GAN setting
    mi_weight: float = 0.01       # lambda of the memory-conditional (MI) term (paper MNIST: 0.01)
    mi_form: str = "cosine"       # "cosine" (paper: pull toward own key) or "contrastive" (-log p(slot | image))

    # --- environment ----------------------------------------------------
    device: str = "cpu"
    data_root: Path = field(default_factory=lambda: Path(r"D:\M-GAN\data"))
    output_root: Path = field(default_factory=lambda: Path(r"D:\M-GAN\outputs"))
