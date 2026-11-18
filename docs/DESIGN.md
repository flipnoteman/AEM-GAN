# M-GAN design: a minimal memoryGAN baseline

Target paper: Kim, Kim & Kim, *Memorization Precedes Generation: Learning
Unsupervised GANs with Memory Networks* (ICLR 2018).  This repo is a
deliberately small PyTorch baseline of that model.  It is written to be
**modified**: the memory (how image features are stored and looked up) will
later be replaced by a quantum implementation, so that part is isolated and
documented more heavily than the rest.

## 1. What a memoryGAN is, in one paragraph

A normal GAN has a generator `G(z)` and a discriminator `D(x)`.  memoryGAN
adds a **key/value memory** of `N` slots.  Every slot holds a *key* (a
unit-norm feature vector produced by an encoder from an image), a *value*
(whether that feature came from a real or a generated image), a *histogram*
weight `h` (how much data the slot has absorbed) and an *age* (how long
since the slot was last touched).  Together the slots are a mixture of von
Mises-Fisher clusters with prior `h`.  The discriminator no longer has a
classification head: it encodes an image to a query `q` and reports "real"
as the posterior mass, `h_i * exp(kappa * q.K_i)`, that falls on real slots.
The generator is conditioned on a real key sampled in proportion to `h`,
`G(z, k)`, so each generated image is "about" a remembered cluster.  After
every training step the encoder's features for the batch are written back
into the memory with one incremental-EM step.

## 2. Module layout

```
M-GAN/
  mgan/
    __init__.py       re-exports the public classes
    config.py         MemoryGANConfig dataclass (sizes, paths, hyper-params)
    memory.py         Memory            <- the part that will be swapped
    networks.py       Encoder, Generator (plain conv nets)
    model.py          MemoryDiscriminator, MemoryGAN (training + generation)
    data.py           MNIST loader rooted on D:\M-GAN\data
  scripts/
    smoke_test.py     build model, one train step, generate, save grid
    train.py          optional short training loop (not needed for baseline)
  tests/
    test_memory.py    contract tests parametrised by a memory factory;
                      add the replacement's factory and the same tests run
  docs/DESIGN.md      this file
```

Paths: code lives on C:, everything written at run time (datasets, sample
grids, checkpoints) goes under `D:\M-GAN\` via `MemoryGANConfig.data_root`
and `MemoryGANConfig.output_root`.

## 3. Class hierarchy

Only `torch.nn.Module` subclasses; no abstract base classes.  Swapping the
memory means subclassing `Memory` and overriding a small, named set of
methods (section 4), not re-implementing an interface.

```
nn.Module
 ├── Memory               keys [N,D], values [N], hist [N], ages [N]  (buffers, no grad)
 ├── Encoder              image -> unit-norm feature q  [B,D]
 ├── Generator            (z [B,Z], key [B,D]) -> image
 ├── MemoryDiscriminator  owns one Encoder and one Memory
 │      forward(x)   -> p(real | x)        (differentiable w.r.t. Encoder)
 │      slot_posterior(x) -> p(slot | x)   (diagnostics)
 │      remember(x, is_real)  -> writes encoder features into memory (no grad)
 └── MemoryGAN            owns Generator + MemoryDiscriminator + optimisers
        train_step(real_images) -> dict of scalar losses
        generate(n)              -> images in [-1, 1]
```

Why `MemoryDiscriminator` is its own class rather than folding Encoder +
Memory into `MemoryGAN`: it is the exact unit the paper calls the
*discriminative memory network*, and it keeps "turn an image into a real/fake
probability" separate from "run a GAN training step".

## 4. The Memory class (the swappable part)

State (registered as buffers so `.to(device)` and `state_dict` work, and so
no gradient ever flows into stored keys):

| buffer   | shape  | meaning                                                       |
|----------|--------|---------------------------------------------------------------|
| `keys`   | [N, D] | unit-norm feature vectors (vMF cluster means)                 |
| `values` | [N]    | 1.0 = came from a real image, 0.0 = from a generated image    |
| `hist`   | [N]    | mixture prior: decayed count of the data each slot absorbed   |
| `ages`   | [N]    | write calls since the slot was last touched; large = stale    |

Initialisation: random unit-norm keys, first half of the slots real, second
half fake, `hist = 1e-5` everywhere (as in the authors' code), all ages 0.
The tiny histogram means the first writes dominate it, so the random keys
are overwritten or never sampled.

Methods, split into two groups.  **Group A is the storage/lookup primitive
set; a quantum memory overrides these.  Group B is built only from group A
plus the `values`/`hist` buffers and should not need to change.**  `values`,
`hist` and `ages` must stay ordinary tensors; only `keys` may change
representation.

Group A – storage and lookup primitives
* `similarity(q) -> [B, N]`  cosine similarity of each query to every key.
  This is the *only* place the distance metric lives.  It must stay on the
  cosine scale (or `kappa` is retuned) and be differentiable w.r.t. `q`.
* `lookup(q, k) -> (indices [B,k], sims [B,k])`  top-k slots by `similarity`,
  sorted best first; `sims` carries the gradient, `indices` does not.
* `read_keys(indices) -> [n, D]`  classical copies of stored keys; the only
  way keys leave the memory (the generator consumes them).
* `write(q [B,D], is_real [B] bool)`  one call per step, rows handled
  sequentially.  For each row with label `y`: score every slot by
  `kappa*q.K + log(h + beta)`, minus `2*kappa` for the wrong label, and take
  the top-`neighbours` candidates.  If even the best candidate has the wrong
  label, `_replace` the oldest slot (random tie-break) with the query, its
  label and the mean histogram.  Otherwise `_update` runs one incremental-EM
  step on the same-label candidates: with `h_hat = alpha*h`,
  responsibilities `gamma = softmax(kappa*q.K + log h_hat)`, then
  `h <- h_hat + gamma` and `K <- normalise(K + (gamma/h)(q - K))`, ages
  reset.  After all rows, age every slot by 1.  This is Sec. 3.1 of the
  paper and the authors' code.  `write` carries no gradient and runs only
  after both `backward()` calls.

  History: the first version merged if *any* of the top-16 slots matched
  (almost always true, so nothing was replaced and the memory stayed at its
  random initialisation).  The second used the single nearest slot (Kaiser
  et al. 2017) with uniform key sampling; that filled the memory but reached
  a Fréchet distance of ~200 at best.  The EM update with the histogram
  prior reached ~50 (section 5).

  Caveat (measured): with this update all live keys collapse.  Each EM step
  moves all 128 same-label candidates toward the query by the same fraction
  `gamma/(alpha*h + gamma)`, close to 1 while `h` is small, so after a few
  steps the 128 live real keys are one vector and the 128 live fake keys
  another.  The cosine discriminator then acts as a linear real/fake head and
  the model trains as an ordinary GAN that ignores its key.  A discriminator
  that needs distinct keys (the QAE neighbour count in `memorygan_qae.ipynb`)
  fails on it; the notebook runs that model on the nearest-slot rule instead.

Group B – what the GAN asks of the memory
* `real_logit(q) -> [B]`  log-odds of real: the posterior mass of the vMF
  mixture, `p(i|q) ~ h_i exp(kappa q.K_i)`, on real slots, squashed into
  `[eps, 1-eps]` (paper: 0.001).  By default (`dense=False`, as in the
  authors' code) the sum runs over the top-k slots by posterior score.  The
  earlier version of this repo used all slots (`dense=True`), so that a
  query whose neighbours are all fake still gets a gradient toward the
  nearest real slot.  With the EM memory that trained worse (Fréchet 85 vs
  47 at 3000 steps; the discriminator ended at chance), so top-k is the
  default and the contract test
  `test_real_logit_has_gradient_when_neighbours_all_fake` is skipped for it.
  `real_posterior` is its sigmoid.
* `slot_posterior(q) -> [B, N]`  `p(slot | q)` over all slots (diagnostics).
* `sample_real_keys(n) -> (keys [n,D], indices [n])`  real slots drawn in
  proportion to `hist`, read out via `read_keys`; what the generator is
  conditioned on.

`kappa` (sharpness), `k` (neighbours), `alpha`, `beta`, `eps` and `dense`
are constructor arguments; `MemoryGANConfig` sets them.

## 5. Training step (MemoryGAN.train_step)

```
keys, _ = memory.sample_real_keys(B);  z ~ N(0, I)
fake  = G(z, keys)
MI(x, keys) = -kappa * mean(keys . Enc(x))          # paper's memory-conditional term

D step:  L_D = BCE_logits(D(real), 1) + BCE_logits(D(fake.detach()), 0)
              + mi_weight * MI(fake.detach(), keys)
G step:  L_G = BCE_logits(D(fake), 1) + mi_weight * MI(fake, keys)
D.remember(real, fake)                   # single write, after both backwards
```

Standard non-saturating GAN loss.  The MI term is the vMF log-likelihood of
each generated image's feature under the cluster of the key it was
conditioned on (a lower bound on the mutual information between key and
image).  It trains both sides, as in the paper's Eq. 7.  On the generator
only, the encoder had no reason to tell images apart and its features
collapsed (cosine ~0.99 between different digits).  Invariant: the memory is
written exactly once per step, after both `backward()` calls.  `MemoryGAN`
never calls `memory.write` itself; all writes go through
`MemoryDiscriminator.remember`.

Hyper-parameters (`MemoryGANConfig`) follow the paper's MNIST setup: key
size 256, 4096 slots, k = 128, conv width 64, lr 2e-4, Adam (0.5, 0.999),
batch 64, `mi_weight` (lambda) 0.01.  Two were tuned on MNIST, one seed per
setting, scored by Fréchet distance / Inception-style score in a small MNIST
classifier's feature space (1500 steps unless noted):

| change from the paper                     | Fréchet / IS |
|-------------------------------------------|--------------|
| as published (kappa = 1, z dim = 2)       | 675 / 3.1    |
| kappa = 10                                | 453 / 4.3    |
| z dim = 16 (kappa = 1)                    | 216 / 4.1    |
| kappa = 10 and z dim = 16 (default)       | 61 / 6.1     |
| ... same, 3000 steps                      | 47 / 6.4     |
| ... same, 3000 steps, dense posterior     | 85 / 5.9     |

For comparison, the earlier nearest-slot memory reached 206 / 4.2 at its best
`mi_weight`.  Things tried that did not help: lambda 0.1 or 1 (the MI term
overpowers the adversarial loss: 218, and collapse at 1), instance noise on
D's inputs (354), z dim 64 (87).

## 6. Generation

`MemoryGAN.generate(n)`: sample `n` real keys and `n` noise vectors, run the
generator, return images.  Also exposed: `generate_from_keys(keys, z)` for
experiments that pick slots by hand.

## 7. Deliberate omissions

* No pretrained weights: the official implementation is TensorFlow with no
  published checkpoint, and this architecture is bespoke, so nothing can be
  imported.  `scripts/smoke_test.py` proves the pieces fit (one step, one
  generation) in seconds on CPU.
* No logging/checkpoint framework, no argparse beyond the smoke script.
* No CUDA-specific code; `config.device` is the only switch.
