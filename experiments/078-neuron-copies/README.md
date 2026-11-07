# 078 neuron copies: are GPT-2 small MLP SAE latents near-copies of single neurons?

## Question

Cammarata's essay "Neural Networks Are Already Clean" reports that on vision layers most SAE activity is near-copies of
single neurons. If that held for GPT-2 small's MLP-output dictionaries, a neuron-basis WCM would be a natural
comparison. This measures, per layer, how many of the latents of the OpenAI v5 32k Top-K MLP dictionaries (jbloom
release, k = 32, d_sae 32,768; 3,072 neurons per layer) behave like one neuron.

## What the code does

All three scripts use the 070 public-model helpers (`experiments/070-public-model-adapters/common.py`), the HF `gpt2`
weights and `sae.public_bank.PublicSAEBank`. Data: 1,600 wikitext windows of 64 tokens (102,400 tokens, no BOS),
skipping the first 20,000 windows that 053 / 070 smoke runs used. Layers 0–9 and 11 (the default `LAYERS` list leaves
out layer 10; the code does not say why).

1. **`neuron_copies.py`** → `out/layer_<L>.json`, `out/per_latent_L<L>.pt`.
   - **Weights:** for each latent, the max over neurons of |cos(decoder row, centred W_out row)|. The SAE reads the
     layer-normed MLP output, so the W_out rows are centred over d_model first.
   - **Activations:** for each latent, the max over neurons of Pearson r(latent code, post-GELU neuron) over the
     tokens; latents that never fire are counted separately. Also r for the latent's best neuron by weight, the share
     of latents where the two choices agree, and the share of total latent activation carried by latents with
     r ≥ 0.8.
2. **`control.py`** → `out/control.json`. Calibration.
   - **r ceiling:** for 512 random neurons per layer, an exact sparse copy of the neuron (firing only on the neuron's
     top-q tokens with the neuron's value, q = the median firing rate of that layer's alive latents). Its r with the
     neuron is the best a true near-copy could score at that sparsity. Also r for a copy that is the neuron's whole
     positive part.
   - **cos chance:** the max |cos| of 4,096 random unit directions with the centred W_out rows.
3. **`copies_overlap.py`** → `out/overlap_L<L>.json/.pt`, `out/overlap_all.json`. The replacement activation test,
   since the control shows Pearson r cannot detect a sparse copy. For each neuron, its tail set = tokens where it is
   in its top q (q in 1e-2 … 1e-4). For each latent, its firing set, and the best neuron by Jaccard overlap, with
   precision (when the latent fires, the neuron is in its tail) and recall (the reverse).
   - **Matched:** each latent is scored at the tail size nearest its own firing rate (log scale). A copy = precision
     ≥ 0.5 and recall ≥ 0.5.
   - **Any q:** a copy = precision ≥ 0.8 and recall ≥ 0.5 at some q.

## Files and commands

| File | What |
|---|---|
| `neuron_copies.py` | Weight cosines and activation correlations per layer |
| `control.py` | r ceiling for a perfect sparse copy, cos chance level (needs `neuron_copies.py`'s outputs) |
| `copies_overlap.py` | Firing-set overlap with the best neuron |
| `run_078.sh` | Runs `neuron_copies.py` on GPU 0 |
| `out/` | All outputs (JSON summaries; `.pt` per-latent tensors) |
| `run.log` | Console log of `neuron_copies.py` |

```
bash experiments/078-neuron-copies/run_078.sh
PYTHONPATH=src HF_HUB_OFFLINE=1 python experiments/078-neuron-copies/control.py
PYTHONPATH=src HF_HUB_OFFLINE=1 python experiments/078-neuron-copies/copies_overlap.py
#   env (all three): N_WIN (1600)  LAYERS (0,1,2,3,4,5,6,7,8,9,11)
```

## Results

**Weights and correlations** (`out/layer_<L>.json`, `out/control.json`). Alive latents: 30,820–32,452 of 32,768 per
layer.

| layer | median max cos | cos chance (median / p99) | median max r | r ceiling of a perfect sparse copy | latents with r ≥ 0.8 |
|---|---|---|---|---|---|
| 0 | 0.19 | 0.13 / 0.17 | 0.11 | 0.21 | 0.8% |
| 1 | 0.21 | 0.13 / 0.17 | 0.10 | 0.11 | 0.3% |
| 5 | 0.21 | 0.13 / 0.17 | 0.09 | 0.20 | 0.3% |
| 11 | 0.20 | 0.13 / 0.17 | 0.07 | 0.20 | 0.1% |

Over layers 1–11 the median max cos is 0.20–0.22 and the share with cos ≥ 0.7 is 2.9–8.2% (layer 0: 2.4%). The r
ceiling of a sparse copy is 0.11–0.23 on every layer, while a copy of the neuron's whole positive part scores
0.91–0.96. So a low Pearson r does not rule out copies at these firing rates (median latent firing rate 1.7e-4 to
3.0e-4), which is why `copies_overlap.py` replaces it.

**Firing-set overlap** (`out/overlap_all.json`, matched tail size):

| layer | share of alive latents that are copies | share of activation mass | median Jaccard (p90) | median precision / recall |
|---|---|---|---|---|
| 0 | 11.1% | 24.4% | 0.12 (0.42) | 0.28 / 0.19 |
| 1 | 4.9% | 13.5% | 0.08 (0.28) | 0.18 / 0.13 |
| 2 | 3.5% | 20.6% | 0.07 (0.26) | 0.17 / 0.10 |
| 3 | 2.7% | 18.9% | 0.06 (0.21) | 0.13 / 0.09 |
| 4 | 2.0% | 14.3% | 0.06 (0.18) | 0.12 / 0.09 |
| 5 | 2.2% | 17.4% | 0.06 (0.19) | 0.12 / 0.09 |
| 6 | 2.0% | 16.9% | 0.06 (0.18) | 0.12 / 0.09 |
| 7 | 1.7% | 14.3% | 0.06 (0.18) | 0.12 / 0.09 |
| 8 | 1.6% | 13.0% | 0.06 (0.17) | 0.12 / 0.09 |
| 9 | 1.5% | 14.1% | 0.06 (0.17) | 0.12 / 0.09 |
| 11 | 0.7% | 16.3% | 0.06 (0.14) | 0.11 / 0.09 |

Under the stricter any-q rule the copy share is 0.1–3.4% of latents and 0.7–11.0% of activation mass.

Reading: on GPT-2 small's MLP dictionaries, near-copies of single neurons are a small minority of latents (11% at
layer 0, at most 5% elsewhere), though they carry a larger share of the activation mass (13–24%). Most latents are not
one neuron, unlike the vision result.

## Status and caveats

- Not cited in the paper. A calibration for whether a neuron-basis comparison would be close to the SAE basis; no
  neuron-basis WCM was run here.
- One model, MLP-output sites only, wikitext only, 102,400 tokens. Layer 10 is not measured.
- Firing sets are taken on wikitext; the dictionaries were trained on OpenWebText.
- The copy thresholds (precision and recall ≥ 0.5 matched, ≥ 0.8 / 0.5 any q) are the script's choices, set in code.
