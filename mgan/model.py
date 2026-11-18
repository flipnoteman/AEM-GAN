"""memoryGAN: the discriminative memory network and the training/generation wrapper.

``MemoryDiscriminator`` is the paper's DMN: an ``Encoder`` whose features are
classified by a ``Memory`` instead of a linear head.  ``MemoryGAN`` wires it to
a ``Generator`` and owns the two optimisers.

Step invariant (see ``Memory``): within ``train_step`` the memory is read
during the D and G passes and written exactly once, after both backward
calls, so the keys sampled for conditioning are the ones the losses saw.
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
            Memory(config.memory_slots, config.feature_dim, config.neighbours, config.kappa,
                   hist_decay=config.hist_decay, dense=config.dense_posterior),
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

    def mutual_information(self, images: torch.Tensor, keys: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """The memory-conditional term for images generated from ``keys`` (stored in ``slots``).

        ``config.mi_form == "cosine"`` (the paper): ``-kappa * mean(K_c . mu(G(z, K_c)))``,
        minus the vMF log-likelihood of each image's feature under its own
        key's component (up to a constant).  It only *pulls* the feature
        toward its key.

        ``"contrastive"``: ``-mean(log p(slot_c | G(z, K_c)))`` under the
        memory's posterior over all slots.  The normaliser also *pushes* the
        feature away from every other slot, which keeps features spread over
        the sphere.  A coarse discriminator such as the QAE Hamming-radius
        count needs that: if all features sit inside one radius, every key is
        everyone's neighbour and the counts carry no information.
        """
        if self.config.mi_form == "contrastive":
            rows = torch.arange(len(slots), device=images.device)
            return -torch.log(self.discriminator.slot_posterior(images)[rows, slots] + 1e-8).mean()
        return -(self.memory.kappa * (keys * self.discriminator.encoder(images)).sum(1)).mean()

    def train_step(self, real: torch.Tensor) -> dict:
        """One generator/discriminator update on a batch of real images.

        D step: non-saturating GAN loss on the memory's real/fake log-odds,
                plus ``mi_weight`` times the memory-conditional term on the
                (detached) generated batch.
        G step: non-saturating GAN loss plus ``mi_weight`` times the same term.
        Then the real and generated features are written to memory.

        The memory-conditional term trains both sides (InfoGAN-style; the
        paper's Eq. 7 adds it to both objectives).  With it on the generator
        only, nothing asks the encoder to tell images apart, only real from
        fake: its features collapsed to one direction (cosine ~0.99 between
        different digits) and the memory never left its random initialisation.

        Args:
            real: [B, C, H, W] images in [-1, 1].
        Returns:
            dict of float losses for logging.
        """
        real = real.to(self.config.device)
        batch = real.shape[0]
        lam = self.config.mi_weight
        ones = torch.ones(batch, device=real.device)
        zeros = torch.zeros(batch, device=real.device)

        keys, slots = self.memory.sample_real_keys(batch)
        z = torch.randn(batch, self.config.latent_dim, device=real.device)
        fake = self.generator(z, keys)

        # --- discriminator ---------------------------------------------
        self.opt_d.zero_grad()
        loss_d = (F.binary_cross_entropy_with_logits(self.discriminator(real), ones)
                  + F.binary_cross_entropy_with_logits(self.discriminator(fake.detach()), zeros))
        loss_d_mi = self.mutual_information(fake.detach(), keys, slots)
        (loss_d + lam * loss_d_mi).backward()
        self.opt_d.step()

        # --- generator -------------------------------------------------
        self.opt_g.zero_grad()
        loss_g_adv = F.binary_cross_entropy_with_logits(self.discriminator(fake), ones)
        loss_g_mi = self.mutual_information(fake, keys, slots)
        (loss_g_adv + lam * loss_g_mi).backward()
        self.opt_g.step()

        # --- memory write (after both backward calls) --------------------
        self.discriminator.remember(real, fake.detach())

        return {"loss_d": loss_d.item(), "loss_d_mi": loss_d_mi.item(),
                "loss_g_adv": loss_g_adv.item(), "loss_g_mi": loss_g_mi.item()}

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, n: int) -> torch.Tensor:
        """``n`` images, each conditioned on a real key drawn in proportion to its histogram."""
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
