"""Key/value memory for memoryGAN.  This is the module to replace.

The memory holds ``N`` slots.  Each slot has a *key* (a unit-norm feature
vector from the encoder), a *value* (1.0 if the key came from a real image,
0.0 if from a generated one) and an *age* (write calls since the slot was
last touched).  Keys, values and ages are registered buffers: they move with
``.to(device)``, are saved in ``state_dict``, and never receive gradients.

Methods are in two groups.

Group A - storage and lookup primitives.  A replacement memory overrides
these.  ``similarity`` is the only place the distance metric lives;
``read_keys`` is the only place keys are read out; ``_merge`` and
``_replace`` are the only places a key is changed.

Group B - what the GAN asks of the memory.  Built entirely from group A plus
the ``values`` buffer.  A replacement should not need to touch these, but it
must keep ``values`` as an ordinary float tensor of 0/1 because group B reads
it directly.  Only ``keys`` may change representation.

Contract for a replacement
--------------------------
* Queries ``q`` are float32, unit-norm, shape ``[B, D]``, on the memory's
  device.  Results come back on ``q.device``.
* ``similarity`` must be on the cosine scale (about [-1, 1]) or ``kappa``
  needs retuning, and it must be differentiable with respect to ``q``.  The
  encoder is trained only through ``real_logit`` and ``slot_posterior``, so a
  non-differentiable lookup (e.g. a sampled swap test) needs a surrogate
  gradient or a frozen encoder.
* ``write`` carries no gradient and must only be called after every
  ``backward()`` of the current step, because it mutates ``keys`` in place.
* ``lookup`` returns slots sorted by descending similarity.
* ``read_keys`` returns classical ``[n, D]`` vectors the generator can consume.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):
    """Cosine-similarity key/value memory with the memoryGAN update rule."""

    def __init__(self, slots: int, dim: int, neighbours: int = 16, kappa: float = 10.0):
        """
        Args:
            slots:      N, number of key/value slots.
            dim:        D, key size; must equal the encoder's feature size.
            neighbours: k, slots consulted by ``lookup`` and ``write``.
            kappa:      sharpness of the softmax over similarities (vMF concentration).
        """
        super().__init__()
        self.n_slots = slots
        self.dim = dim
        self.neighbours = min(neighbours, slots)
        self.kappa = kappa
        # Initial policy (not part of the contract): random keys, half real, half fake.
        self.register_buffer("keys", F.normalize(torch.randn(slots, dim), dim=1))
        values = torch.zeros(slots)
        values[: slots // 2] = 1.0
        self.register_buffer("values", values)
        self.register_buffer("ages", torch.zeros(slots, dtype=torch.long))

    # ------------------------------------------------------------------
    # Group A: storage and lookup primitives (override these)
    # ------------------------------------------------------------------

    def similarity(self, q: torch.Tensor) -> torch.Tensor:
        """Similarity of every query to every key.

        Args:
            q: [B, D] unit-norm queries.
        Returns:
            [B, N] cosine similarities in [-1, 1]; differentiable w.r.t. ``q``.
        """
        return q @ self.keys.t()

    def lookup(self, q: torch.Tensor, k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``k`` most similar slots for each query.

        Args:
            q: [B, D] unit-norm queries.
            k: how many slots; defaults to ``self.neighbours``.
        Returns:
            indices: [B, k] slot ids, best first (no gradient).
            sims:    [B, k] their similarities, taken from ``similarity``,
                     so gradients flow back to ``q``.
        """
        k = self.neighbours if k is None else min(k, self.n_slots)
        top = self.similarity(q).topk(k, dim=1)
        return top.indices, top.values

    def read_keys(self, indices: torch.Tensor) -> torch.Tensor:
        """Classical copies of the keys in ``indices`` ([n] -> [n, D], no gradient)."""
        return self.keys[indices].clone()

    @torch.no_grad()
    def write(self, q: torch.Tensor, is_real: torch.Tensor) -> None:
        """Store a batch of encoder features (the memoryGAN update rule).

        Rows are written one at a time, in order, so a later row sees the
        effect of an earlier one.  For each row:

        1. look up the ``neighbours`` nearest slots;
        2. if any of them has the same label, *merge* into the nearest such
           slot (sum the key with the query, renormalise, reset age);
        3. otherwise *replace* the oldest slot in the whole memory with the
           query and its label.

        Then age every slot by one.

        Args:
            q:       [B, D] unit-norm features.
            is_real: [B] bool, True where the feature came from a real image.
        """
        for query, label in zip(q, is_real.bool()):
            indices, _ = self.lookup(query[None])
            candidates = indices[0]
            same_label = candidates[self.values[candidates] == float(label)]
            if same_label.numel() > 0:
                self._merge(same_label[0], query)
            else:
                self._replace(self._oldest_slot(), query, label)
        self.ages += 1

    def _merge(self, slot: torch.Tensor, q: torch.Tensor) -> None:
        """Step 2 of ``write``.  Falls back to ``q`` if the sum cancels to zero."""
        merged = self.keys[slot] + q
        self.keys[slot] = F.normalize(merged if merged.norm() > 1e-6 else q, dim=0)
        self.ages[slot] = 0

    def _replace(self, slot: torch.Tensor, q: torch.Tensor, is_real) -> None:
        """Step 3 of ``write``."""
        self.keys[slot] = q
        self.values[slot] = float(is_real)
        self.ages[slot] = 0

    def _oldest_slot(self) -> torch.Tensor:
        """A slot with the largest age; ties broken uniformly at random."""
        oldest = (self.ages == self.ages.max()).nonzero(as_tuple=True)[0]
        return oldest[torch.randint(oldest.numel(), (1,), device=oldest.device)][0]

    # ------------------------------------------------------------------
    # Group B: what the GAN asks of the memory (built on group A)
    # ------------------------------------------------------------------

    def real_logit(self, q: torch.Tensor) -> torch.Tensor:
        """log p(real | q) - log p(fake | q): the discriminator's output.

        The paper's vMF mixture over *all* slots: slot weights are
        softmax(``kappa`` * similarity), and p(real) is the total weight on
        real slots.  Computed in log space over the full memory rather than
        the top-k so a query surrounded by fake slots still gets a gradient
        toward the nearest real ones.

        Args:
            q: [B, D] unit-norm queries.
        Returns:
            [B] log-odds of real, differentiable w.r.t. ``q``.
        """
        scores = self.kappa * self.similarity(q)                      # [B, N]
        is_real = (self.values == 1.0)[None]                           # [1, N]
        log_real = torch.logsumexp(scores.masked_fill(~is_real, float("-inf")), dim=1)
        log_fake = torch.logsumexp(scores.masked_fill(is_real, float("-inf")), dim=1)
        return log_real - log_fake

    def real_posterior(self, q: torch.Tensor) -> torch.Tensor:
        """p(real | q) in (0, 1); see ``real_logit``."""
        return torch.sigmoid(self.real_logit(q))

    def slot_posterior(self, q: torch.Tensor) -> torch.Tensor:
        """p(slot | q) over all N slots, for the generator's memory-conditional loss.

        Dense, O(B * N * D).  Fine at a few thousand slots; a replacement that
        approximates it with top-k must guarantee the conditioning slot is
        inside the top-k or the loss is undefined.

        Args:
            q: [B, D] unit-norm queries.
        Returns:
            [B, N] probabilities summing to 1 per row, differentiable w.r.t. ``q``.
        """
        return F.softmax(self.kappa * self.similarity(q), dim=1)

    def sample_real_keys(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``n`` keys uniformly from the slots labelled real.

        These condition the generator.  Raises if no slot is real, which
        would mean the write rule has broken the memory.

        Returns:
            keys:    [n, D] copies of the keys (no gradient).
            indices: [n] the slot of each key, for the memory-conditional loss.
        """
        real_slots = (self.values == 1.0).nonzero(as_tuple=True)[0]
        if real_slots.numel() == 0:
            raise RuntimeError("memory has no real slots to sample from")
        pick = torch.randint(real_slots.numel(), (n,), device=real_slots.device)
        indices = real_slots[pick]
        return self.read_keys(indices), indices
