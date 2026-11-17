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
(whether that feature came from a real or a generated image) and an *age*
(how long since the slot was last touched).  The discriminator no longer has
a classification head: it encodes an image to a query `q`, looks up the
nearest slots, and reports "real" as the similarity-weighted share of those
slots that are real.  The generator is conditioned on a key sampled from the real
slots, `G(z, k)`, so each generated image is "about" a remembered feature.
After every training step the encoder's features for the batch are written
back into the memory.

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
 ├── Memory               keys [N,D], values [N], ages [N]  (buffers, no grad)
 ├── Encoder              image -> unit-norm feature q  [B,D]
 ├── Generator            (z [B,Z], key [B,D]) -> image
 ├── MemoryDiscriminator  owns one Encoder and one Memory
 │      forward(x)   -> p(real | x)        (differentiable w.r.t. Encoder)
 │      slot_posterior(x) -> p(slot | x)   (used by the generator's MI loss)
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
| `keys`   | [N, D] | unit-norm feature vectors                                     |
| `values` | [N]    | 1.0 = came from a real image, 0.0 = from a generated image    |
| `ages`   | [N]    | write calls since the slot was last touched; large = stale    |

Initialisation: random unit-norm keys, first half of the slots real, second
half fake, all ages 0.  The paper does the same; it just means the memory is
"full of noise" at the start and gets overwritten by real features quickly.

Methods, split into two groups.  **Group A is the storage/lookup primitive
set; a quantum memory overrides these.  Group B is built only from group A
plus the `values`/`ages` buffers and should not need to change.**  `values`
and `ages` must stay ordinary tensors; only `keys` may change representation.

Group A – storage and lookup primitives
* `similarity(q) -> [B, N]`  cosine similarity of each query to every key.
  This is the *only* place the distance metric lives.  It must stay on the
  cosine scale (or `kappa` is retuned) and be differentiable w.r.t. `q`: the
  encoder trains only through the posteriors.
* `lookup(q, k) -> (indices [B,k], sims [B,k])`  top-k slots by `similarity`,
  sorted best first; `sims` carries the gradient, `indices` does not.
* `read_keys(indices) -> [n, D]`  classical copies of stored keys; the only
  way keys leave the memory (the generator consumes them).
* `write(q [B,D], is_real [B] bool)`  one call per step, rows handled
  sequentially.  For each row: if one of its top-k slots has the same label,
  `_merge` into the nearest such slot (sum, renormalise, age 0); otherwise
  `_replace` the oldest slot (random tie-break) with the key and label.
  After all rows, age every slot by 1.  This is the paper's memory update
  rule.  `write` carries no gradient and runs only after both `backward()`
  calls.

Group B – what the GAN asks of the memory
* `real_logit(q) -> [B]`  log-odds of real: the paper's vMF mixture over
  *all* slots, `logsumexp(kappa*sims[real]) - logsumexp(kappa*sims[fake])`.
  Dense rather than top-k so a query whose neighbours are all fake still
  gets a gradient toward the nearest real slot.  `real_posterior` is its
  sigmoid.
* `slot_posterior(q) -> [B, N]`  `p(slot | q)` over all slots, for the
  mutual-information term that ties generated images to their key.
* `sample_real_keys(n) -> (keys [n,D], indices [n])`  uniform over slots
  whose value is 1, read out via `read_keys`; what the generator is
  conditioned on.

`kappa` (sharpness) and `k` (neighbours) are constructor arguments.

## 5. Training step (MemoryGAN.train_step)

```
real  -> q_real = Enc(real)
keys, idx = memory.sample_real_keys(B);  z ~ N(0, I)
fake  = G(z, keys)

D step:  L_D = BCE_logits(D(real), 1) + BCE_logits(D(fake.detach()), 0)
G step:  L_G = BCE_logits(D(fake), 1) + mi_weight * NLL(slot_posterior(fake), idx)
D.remember(real, fake)                   # single write, after both backwards
```

Standard non-saturating GAN loss.  The MI term (paper's "memory-conditional"
loss) is one line, updates the generator only, and has a weight you can set
to 0.  Invariant: the memory is written exactly once per step, after both
`backward()` calls, so `idx` stays valid and no in-place/autograd clash can
occur.  `MemoryGAN` never calls `memory.write` itself; all writes go through
`MemoryDiscriminator.remember`.

Observed on CPU, 500 steps, 0.1 s/step: with a top-k posterior the
discriminator loss collapsed to ~0.01 (the generator got zero gradient once
its neighbours were all fake); with the dense posterior the losses stay
balanced (D ~0.3-0.7, G ~1-2).  Samples are still noise at 500 steps,
which is expected; nothing here is tuned.

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
