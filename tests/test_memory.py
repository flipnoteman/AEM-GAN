"""Contract tests for ``Memory``, using only its public API.

``FACTORIES`` is the one thing to change when testing a replacement: add a
factory for it and every test below runs against it too.
"""

import pytest
import torch
import torch.nn.functional as F

from mgan.memory import Memory

SLOTS, DIM, K = 32, 8, 4


def classical():
    torch.manual_seed(0)
    return Memory(SLOTS, DIM, neighbours=K, kappa=10.0)


FACTORIES = [classical]


def unit(n: int) -> torch.Tensor:
    return F.normalize(torch.randn(n, DIM), dim=1)


def real(n: int) -> torch.Tensor:
    return torch.ones(n, dtype=torch.bool)


def fake(n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.bool)


@pytest.fixture(params=FACTORIES)
def memory(request):
    return request.param()


# --- initialisation policy (classical default, optional for a replacement) --

def test_initial_state(memory):
    assert memory.n_slots == SLOTS and memory.dim == DIM
    assert memory.values.sum() == SLOTS // 2
    assert memory.ages.dtype == torch.long and memory.ages.sum() == 0


# --- group A: lookup, read, write ------------------------------------------

def test_lookup_sorted_and_differentiable(memory):
    q = unit(3).requires_grad_()
    idx, sims = memory.lookup(q)
    assert idx.shape == sims.shape == (3, K)
    assert (sims[:, :-1] >= sims[:, 1:]).all()
    sims.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0


def test_lookup_k_clamped_to_slots(memory):
    idx, _ = memory.lookup(unit(1), k=SLOTS * 10)
    assert idx.shape[1] == SLOTS


def test_read_keys_returns_unit_norm_copies(memory):
    idx = torch.tensor([0, 3])
    keys = memory.read_keys(idx)
    assert keys.shape == (2, DIM)
    assert torch.allclose(keys.norm(dim=1), torch.ones(2), atol=1e-5)
    keys.zero_()
    assert memory.read_keys(idx).norm(dim=1).min() > 0.5   # memory untouched


def test_write_moves_nearest_slot_toward_query(memory):
    q = unit(1)
    before = memory.lookup(q, k=1)[1][0, 0]
    memory.write(q, real(1))                         # merge or replace, either way:
    slot_idx, sims = memory.lookup(q, k=1)
    assert sims[0, 0] > before                       # q is now closer to memory
    assert memory.values[slot_idx[0, 0]] == 1.0      # and its nearest slot is real


def test_write_merges_into_nearest_same_label(memory):
    q = unit(1)
    memory.write(q, real(1))
    slot, sim = (t[0, 0] for t in memory.lookup(q, k=1))
    n_real, keys = memory.values.sum(), memory.keys.clone()
    memory.write(q, real(1))                         # same query, same label
    assert memory.values.sum() == n_real             # merged, nothing replaced
    assert memory.ages[slot] == 1                    # updated (age reset to 0), then aged once
    changed = (memory.keys - keys).norm(dim=1) > 0
    assert (memory.values[changed] == 1.0).all()     # only real slots moved
    assert memory.lookup(q, k=1)[1][0, 0] >= sim - 1e-6   # and memory is no further from q


def test_write_replaces_oldest_when_no_label_match(memory):
    memory.values[:] = 0.0                           # nothing is real
    memory.ages[7] = 100                             # slot 7 is the oldest
    q = unit(1)
    memory.write(q, real(1))
    assert memory.lookup(q, k=1)[0][0, 0] == 7
    assert memory.values[7] == 1.0 and memory.ages[7] == 1


def test_write_ages_every_slot_once_per_call(memory):
    before = memory.ages.clone()
    memory.write(unit(5), real(5))
    assert (memory.ages - before)[memory.ages > 1].eq(1).all()
    assert memory.ages.max() == before.max() + 1


def test_write_rows_are_sequential(memory):
    memory.values[:] = 0.0
    q = unit(1).repeat(2, 1)                         # same query twice, both real
    memory.write(q, real(2))
    assert memory.values.sum() == 1                  # second row merged into the first


def test_keys_stay_unit_norm_after_writes(memory):
    for _ in range(3):
        memory.write(unit(8), torch.rand(8) > 0.5)
    keys = memory.read_keys(torch.arange(SLOTS))
    assert torch.allclose(keys.norm(dim=1), torch.ones(SLOTS), atol=1e-5)


# --- group B: what the GAN relies on ---------------------------------------

def test_real_posterior_bounds_and_gradient(memory):
    q = unit(5).requires_grad_()
    p = memory.real_posterior(q)
    assert p.shape == (5,)
    assert (p > 0).all() and (p < 1).all()
    assert torch.allclose(p, torch.sigmoid(memory.real_logit(q)))
    p.sum().backward()
    assert q.grad.abs().sum() > 0


def test_real_logit_has_gradient_when_neighbours_all_fake(memory):
    # One real key opposite q, k fake copies of q: the dense mixture must
    # still pull q toward the real key.  The top-k posterior of the authors'
    # code (the default, it trains better) gives 0 grad here by design.
    if type(memory).real_logit is Memory.real_logit and not memory.dense:
        pytest.skip("top-k posterior: no gradient from outside the top-k (dense=False)")
    q = unit(1)
    memory.values[:] = 0.0
    memory.ages[0] = 10
    memory.write(-q, real(1))                        # goes to slot 0, the oldest
    memory.write(q.repeat(K, 1), fake(K))            # first replaces, rest merge
    q = q.clone().requires_grad_()
    memory.real_logit(q).sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0


def test_retrieval_after_write(memory):
    q = unit(1)
    memory.write(q, real(1))
    slot = memory.lookup(q, k=1)[0][0, 0]
    assert memory.slot_posterior(q).argmax() == slot
    before = memory.real_posterior(q)
    assert before > 0.5
    memory.write(q.repeat(3, 1), fake(3))            # fake evidence near q
    assert memory.real_posterior(q) < before
    assert memory.values[slot] == 1.0                # the real slot itself is untouched


def test_slot_posterior_normalised(memory):
    p = memory.slot_posterior(unit(4))
    assert p.shape == (4, SLOTS)
    assert torch.allclose(p.sum(dim=1), torch.ones(4), atol=1e-5)


def test_sample_real_keys(memory):
    keys, idx = memory.sample_real_keys(10)
    assert keys.shape == (10, DIM)
    assert (memory.values[idx] == 1.0).all()
    assert torch.allclose(keys, memory.read_keys(idx))


def test_sample_real_keys_raises_when_empty(memory):
    memory.values[:] = 0.0
    with pytest.raises(RuntimeError):
        memory.sample_real_keys(1)
