# M-GAN

A minimal PyTorch baseline of memoryGAN (Kim, Kim & Kim, ICLR 2018), written
to be modified: the memory (`mgan/memory.py`) is the part that will be
swapped for a quantum implementation.  See `docs/DESIGN.md` for the rationale.

Code lives here on C:.  Data and outputs go to `D:\M-GAN\` (set in
`mgan/config.py`).

```
pip install -r requirements.txt
python scripts/smoke_test.py          # build, 3 steps, write a sample grid
python -m pytest tests                # memory contract tests
python scripts/train.py --steps 500   # optional short training run
```

Public API (all in `mgan`):

* `Memory` – keys/values/ages; `similarity`, `lookup`, `read_keys`, `write`
  are the storage primitives to replace; `real_logit`, `slot_posterior`,
  `sample_real_keys` are what the GAN uses.
* `MemoryDiscriminator` – `Encoder` + `Memory`; `forward(x)` gives real/fake
  log-odds, `remember(real, fake)` is the only write path.
* `MemoryGAN` – `train_step(real_images) -> losses`, `generate(n) -> images`.
