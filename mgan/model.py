"""memoryGAN: the discriminative memory network and the training/generation wrapper.

``MemoryDiscriminator`` is the paper's DMN: an ``Encoder`` whose features are
classified by a ``Memory`` instead of a linear head.  ``MemoryGAN`` wires it to
a ``Generator`` and owns the two optimisers.

Step invariant (see ``Memory``): within ``train_step`` the memory is read
during the D and G passes and written exactly once, after both backward
calls.  The slot indices sampled for conditioning stay valid until that write.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MemoryGANConfig
from .memory import Memory
from .networks import Encoder, Generator


class MemoryDiscriminator(nn.Module):
    """Encoder + Memory.  Turns an image into p(real | image).

    All memory writes go through ``remember`` so the trainer never touches the
    memory's write path directly.
    """

    def __init__(self, encoder: Encoder, memory: Memory):
        super().__init__()
        self.encoder = encoder
        self.memory = memory

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B] log-odds that each image is real (sigmoid for a probability)."""
        return self.memory.real_logit(self.encoder(x))

    def slot_posterior(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, N] probability over memory slots."""
        return self.memory.slot_posterior(self.encoder(x))

    @torch.no_grad()
    def remember(self, real: torch.Tensor, fake: torch.Tensor) -> None:
        """Encode a real and a generated batch and write both into memory.

        The two batches are encoded separately so batch-norm statistics match
        the forward passes the losses were computed on; the memory then sees
        one write call with the real rows first.
        """
        q = torch.cat([self.encoder(real), self.encoder(fake)])
        is_real = torch.cat([torch.ones(len(real), dtype=torch.bool, device=q.device),
                             torch.zeros(len(fake), dtype=torch.bool, device=q.device)])
        self.memory.write(q, is_real)


class MemoryGAN(nn.Module):
    """Generator + MemoryDiscriminator with one training step and generation.

    Everything is built on ``config.device``.  ``state_dict`` covers the
    networks and the memory but not the optimisers.
    """

    def __init__(self, config: MemoryGANConfig):
        super().__init__()
        self.config = config
        self.generator = Generator(config.latent_dim, config.feature_dim,
                                   config.channels, config.base_channels)
        self.discriminator = MemoryDiscriminator(
            Encoder(config.channels, config.feature_dim, config.base_channels),
            Memory(config.memory_slots, config.feature_dim, config.neighbours, config.kappa),
        )
        self.to(config.device)
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=config.lr, betas=config.betas)
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=config.betas)

    @property
    def memory(self) -> Memory:
        return self.discriminator.memory

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def train_step(self, real: torch.Tensor) -> dict:
        """One generator/discriminator update on a batch of real images.

        D step: non-saturating GAN loss on the memory's real/fake log-odds.
        G step: non-saturating GAN loss plus ``mi_weight`` times the
                memory-conditional term, -log p(slot | G(z, key_slot)),
                which rewards images the memory can trace back to their key.
                It updates the generator only.
        Then the real and generated features are written to memory.

        Args:
            real: [B, C, H, W] images in [-1, 1].
        Returns:
            dict of float losses for logging.
        """
        real = real.to(self.config.device)
        batch = real.shape[0]
        ones = torch.ones(batch, device=real.device)
        zeros = torch.zeros(batch, device=real.device)

        keys, slot_idx = self.memory.sample_real_keys(batch)
        z = torch.randn(batch, self.config.latent_dim, device=real.device)
        fake = self.generator(z, keys)

        # --- discriminator ---------------------------------------------
        self.opt_d.zero_grad()
        loss_d = (F.binary_cross_entropy_with_logits(self.discriminator(real), ones)
                  + F.binary_cross_entropy_with_logits(self.discriminator(fake.detach()), zeros))
        loss_d.backward()
        self.opt_d.step()

        # --- generator -------------------------------------------------
        self.opt_g.zero_grad()
        loss_g_adv = F.binary_cross_entropy_with_logits(self.discriminator(fake), ones)
        p_slot = self.discriminator.slot_posterior(fake)
        rows = torch.arange(batch, device=real.device)
        loss_g_mi = -torch.log(p_slot[rows, slot_idx] + 1e-8).mean()
        loss_g = loss_g_adv + self.config.mi_weight * loss_g_mi
        loss_g.backward()
        self.opt_g.step()

        # --- memory write (after both backward calls) --------------------
        self.discriminator.remember(real, fake.detach())

        return {"loss_d": loss_d.item(), "loss_g_adv": loss_g_adv.item(), "loss_g_mi": loss_g_mi.item()}

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, n: int) -> torch.Tensor:
        """``n`` images, each conditioned on a key drawn from the real slots."""
        keys, _ = self.memory.sample_real_keys(n)
        return self.generate_from_keys(keys)

    @torch.no_grad()
    def generate_from_keys(self, keys: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Images for explicitly chosen keys ``[n, D]``; ``z`` defaults to N(0, I).

        Runs the generator in eval mode (fixed batch-norm statistics) and
        restores the previous mode afterwards.
        """
        if z is None:
            z = torch.randn(keys.shape[0], self.config.latent_dim, device=keys.device)
        was_training = self.generator.training
        self.generator.eval()
        images = self.generator(z, keys)
        self.generator.train(was_training)
        return images
