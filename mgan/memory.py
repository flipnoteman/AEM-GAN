"""Key/value memory for memoryGAN.  This is the module to replace.

The memory holds ``N`` slots.  Each slot has a *key* (a unit-norm feature
vector from the encoder), a *value* (1.0 if the key came from a real image,
0.0 if from a generated one), a *histogram* weight ``h`` (how much data the
slot has absorbed; the mixture prior of the paper) and an *age* (write calls
since the slot was last touched).  All four are registered buffers: they move
with ``.to(device)``, are saved in ``state_dict``, and never receive gradients.

The model is the one of Kim, Kim & Kim (ICLR 2018), Sec. 3.1, and the
authors' code (github.com/whyjay/memoryGAN): the memory is a mixture of
von Mises-Fisher components, ``p(q | slot i) ~ exp(kappa * q . K_i)`` with
prior ``p(i) ~ h_i``, so

    p(slot i | q) ~ h_i * exp(kappa * q . K_i)
    p(real | q)   = sum of p(slot i | q) over real slots.

Methods are in two groups.

Group A - storage and lookup primitives.  A replacement memory overrides
these.  ``similarity`` is the only place the distance metric lives;
``read_keys`` is the only place keys are read out; ``_update`` and
``_replace`` are the only places a key is changed.

Group B - what the GAN asks of the memory.  Built entirely from group A plus
the ``values`` and ``hist`` buffers.  A replacement should not need to touch
these, but it must keep ``values`` (0/1) and ``hist`` as ordinary float
tensors because group B reads them directly.  Only ``keys`` may change
representation.

Contract for a replacement
--------------------------
* Queries ``q`` are float32, unit-norm, shape ``[B, D]``, on the memory's
  device.  Results come back on ``q.device``.
* ``similarity`` must be on the cosine scale (about [-1, 1]) or ``kappa``
  needs retuning, and it must be differentiable with respect to ``q``.  The
  encoder is trained only through ``real_logit`` (and, through the encoder
  features, the MI term), so a non-differentiable lookup (e.g. a sampled swap
  test) needs a surrogate gradient or a frozen encoder.
* ``write`` carries no gradient and must only be called after every
  ``backward()`` of the current step, because it mutates ``keys`` in place.
* ``lookup`` returns slots sorted by descending similarity.
* ``read_keys`` returns classical ``[n, D]`` vectors the generator can consume.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):
    """vMF-mixture key/value memory with the memoryGAN incremental-EM update."""

    def __init__(self, slots: int, dim: int, neighbours: int = 128, kappa: float = 10.0, *,
                 hist_decay: float = 0.5, beta: float = 1e-8, eps: float = 1e-3, dense: bool = False):
        """
        Args:
            slots:      N, number of key/value slots.
            dim:        D, key size; must equal the encoder's feature size.
            neighbours: k, candidate slots considered by ``write`` (and by
                        ``real_logit`` when ``dense`` is False).
            kappa:      vMF concentration: sharpness of the posterior over slots.
            hist_decay: alpha of the paper: histogram decay per EM update of a slot (0.5).
            beta:       added to ``hist`` inside the log to keep it finite.
            eps:        p(real) is squashed into [eps, 1 - eps] (paper: 0.001).
            dense:      ``real_logit`` uses all slots (True) or only the top-k
                        by posterior score, as the authors' code does (False,
                        default: at 3000 steps it gave Frechet 47 vs 85 dense).
        """
        super().__init__()
        self.n_slots = slots
        self.dim = dim
        self.neighbours = min(neighbours, slots)
        self.kappa = kappa
        self.hist_decay, self.beta, self.eps, self.dense = hist_decay, beta, eps, dense
        # Initial policy (not part of the contract): random keys, half real, half fake,
        # a tiny uniform histogram (paper: 1e-5) so the first writes dominate it.
        self.register_buffer("keys", F.normalize(torch.randn(slots, dim), dim=1))
        values = torch.zeros(slots)
        values[: slots // 2] = 1.0
        self.register_buffer("values", values)
        self.register_buffer("hist", torch.full((slots,), 1e-5))
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
        effect of an earlier one.  For each row with label ``y``:

        1. score every slot by ``kappa * q.K_i + log(h_i + beta)``, minus
           ``2 * kappa`` if its label differs from ``y``, and take the
           top-``neighbours`` as candidates;
        2. if even the best candidate has the wrong label, nothing in memory
           explains ``q``: *replace* the oldest slot with it;
        3. otherwise run one incremental EM step on the same-label candidates
           (``_update``): their responsibilities for ``q`` pull their keys
           toward it and are added to their histogram.

        Then age every slot by one.  This is the update of Kim et al. 2018
        (Sec. 3.1 and the authors' code).

        Args:
            q:       [B, D] unit-norm features.
            is_real: [B] bool, True where the feature came from a real image.
        """
        log_h = torch.log(self.hist + self.beta)
        for query, label in zip(q, is_real.bool()):
            y = float(label)
            wrong = (self.values != y).float()
            score = self.kappa * self.similarity(query[None])[0] + log_h - 2 * self.kappa * wrong
            cand = score.topk(self.neighbours).indices
            if self.values[cand[0]] != y:
                self._replace(self._oldest_slot(), query, label)
            else:
                self._update(cand[self.values[cand] == y], query)
            log_h = torch.log(self.hist + self.beta)
        self.ages += 1

    def _update(self, cand: torch.Tensor, q: torch.Tensor) -> None:
        """Step 3 of ``write``: one incremental EM step of slots ``cand`` toward ``q``.

        E: responsibilities ``gamma ~ h_hat * exp(kappa * q.K)`` with the decayed
        histogram ``h_hat = hist_decay * h``.  M: ``h <- h_hat + gamma`` and each key
        moves to the running mean ``K <- normalise(K + (gamma / h) * (q - K))``.
        """
        h_hat = self.hist_decay * self.hist[cand]
        gamma = torch.softmax(self.kappa * (self.keys[cand] @ q) + torch.log(h_hat + self.beta), dim=0)
        h_new = h_hat + gamma
        ratio = (gamma / h_new)[:, None]
        self.keys[cand] = F.normalize(self.keys[cand] * (1 - ratio) + q[None] * ratio, dim=1)
        self.hist[cand] = h_new
        self.ages[cand] = 0

    def _replace(self, slot: torch.Tensor, q: torch.Tensor, is_real) -> None:
        """Step 2 of ``write``.  The new slot gets the mean histogram weight."""
        self.keys[slot] = q
        self.values[slot] = float(is_real)
        self.hist[slot] = self.hist.mean()
        self.ages[slot] = 0

    def _oldest_slot(self) -> torch.Tensor:
        """A slot with the largest age; ties broken uniformly at random."""
        oldest = (self.ages == self.ages.max()).nonzero(as_tuple=True)[0]
        return oldest[torch.randint(oldest.numel(), (1,), device=oldest.device)][0]

    # ------------------------------------------------------------------
    # Group B: what the GAN asks of the memory (built on group A)
    # ------------------------------------------------------------------

    def joint_score(self, q: torch.Tensor) -> torch.Tensor:
        """[B, N] unnormalised log p(slot, q): ``kappa * similarity + log(h + beta)``."""
        return self.kappa * self.similarity(q) + torch.log(self.hist + self.beta)

    def real_logit(self, q: torch.Tensor) -> torch.Tensor:
        """log p(real | q) - log p(fake | q): the discriminator's output.

        p(real | q) is the posterior mass on real slots of the vMF mixture,
        squashed into [eps, 1 - eps] as in the paper.  With ``dense`` the sum
        runs over every slot, so a query whose neighbours are all fake still
        gets a gradient toward the nearest real slot; otherwise over the
        top-``neighbours`` slots by joint score, as in the authors' code.

        Args:
            q: [B, D] unit-norm queries.
        Returns:
            [B] log-odds of real, differentiable w.r.t. ``q``.
        """
        s = self.joint_score(q)                                        # [B, N]
        is_real = (self.values == 1.0).expand_as(s)
        if not self.dense and self.neighbours < self.n_slots:
            top = s.topk(self.neighbours, dim=1)
            s, is_real = top.values, is_real.gather(1, top.indices)
        log_z = torch.logsumexp(s, dim=1)
        log_p_real = torch.logsumexp(s.masked_fill(~is_real, float("-inf")), dim=1) - log_z
        log_p_fake = torch.logsumexp(s.masked_fill(is_real, float("-inf")), dim=1) - log_z
        # log(eps + (1 - 2 eps) p), finite even when p = 0
        log_eps, log_scale = math.log(self.eps), math.log1p(-2 * self.eps)
        return (torch.logaddexp(torch.full_like(log_p_real, log_eps), log_p_real + log_scale)
                - torch.logaddexp(torch.full_like(log_p_fake, log_eps), log_p_fake + log_scale))

    def real_posterior(self, q: torch.Tensor) -> torch.Tensor:
        """p(real | q) in [eps, 1 - eps]; see ``real_logit``."""
        return torch.sigmoid(self.real_logit(q))

    def slot_posterior(self, q: torch.Tensor) -> torch.Tensor:
        """p(slot | q) over all N slots (diagnostics; the MI loss uses keys directly).

        Args:
            q: [B, D] unit-norm queries.
        Returns:
            [B, N] probabilities summing to 1 per row, differentiable w.r.t. ``q``.
        """
        return F.softmax(self.joint_score(q), dim=1)

    def sample_real_keys(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``n`` real slots with probability proportional to ``hist``.

        These condition the generator: the paper samples from the learned
        prior over real clusters, so well-populated clusters are drawn more
        often and stale or empty slots almost never.  Raises if no slot is
        real, which would mean the write rule has broken the memory.

        Returns:
            keys:    [n, D] copies of the keys (no gradient).
            indices: [n] the slot of each key.
        """
        w = self.hist * self.values
        if not (w > 0).any():
            raise RuntimeError("memory has no real slots to sample from")
        indices = torch.multinomial(w / w.sum(), n, replacement=True)
        return self.read_keys(indices), indices
