# 060 — The running example under protocol v1: 3.resid.35381 (the "temperature" latent)

2026-09-25. Refit of the paper's running example (pre-v1 study: `../032-running-example`) under the final protocol:
- 32 strongest + 16 mid-band contexts, stratified split, close contrast contexts, rank-keep;
- arms B (γ 0.25, λ 1e-3) and Bw (γ 1, λ 2e-3).

Then: the eval pass, 056 specificity, 057 node attribution (the circuit's own run, consensus over Z/A/C), and a
token-driven reading of the top nodes. Script: `case.py`; full report: `results/case_report.md`.

## Results (arm B, the likely primary)

**Scores, held-out:**

| Held-out set | Nodes | free0 | freeM | freeN | Necessity | Sufficiency to induce |
|---|---|---|---|---|---|---|
| Strongest | 294 | 0.99 | 1.21 | 1.23 | 0.98 | 0.97 |
| Mid-band | 294 | 1.03 | 1.42 | 1.30 | 0.97 | 1.60 |

Faithful, necessary, and able to switch the target on, but it overshoots under the mean ablations (1.2–1.4).

**Specificity: this circuit is NOT specific.** Under A / C:
- target pre-activation faithfulness 1.21 / 1.23;
- siblings (47 latents co-firing on ≥ 50% of training contexts) 0.93 / 0.88, just under the 0.8 × target
  amplifier flag;
- other active latents in the clean Top-K 1.36 / 1.20, restored MORE than the target;
- the circuit lifts about 1,250 latents above the clean Top-K cut, against 2–5 for the empty circuit.

It rebuilds the layer-3 residual stream broadly, not the target selectively. That's plausible for a
residual-stream target, whose circuit is dominated by layer-2 residual nodes, i.e. largely the stream itself.

**What it reads from.** The target peaks on the token "temperature" 61/64 times (95%), in physics / thermodynamics
passages. The top-15 nodes by consensus attribution (all positive, α 1.0–1.7; about a quarter of attribution goes
directly to the target):
- **Pieces of the word "temperature" at layers 0–2.** "temper" + "atures" / "ature" / "temperature" subword
  latents, several at 100% consistency, i.e. string detectors: R2/31405, R2/12030, R1/10359, R1/27935, R0/4856.
  This is token-identity assembly, like the 057 "iv" relay.
- **Heat / thermal context latents.** R2/22529 (heat, thermal), R2/16195 (thermal, heat), M3/17686 ("temperature"
  in reaction-rate contexts). These also fire on the contrast contexts (seq max 1.5–1.7): topical context, not
  the word.
- **Oddities.** Mixed latents (R2/23176 "price"/"temperature", R1/16840 "price", R1/34680 "parameter"),
  plausibly "quantity that varies" features. Unverified.

**Arm Bw** (equal weights) finds nearly the same top nodes (Jaccard not computed here; the top-15 lists largely
coincide) and the same non-specificity (siblings 0.81 / 0.72, controls 1.34 / 1.12).

## Reading

- As a showcase, this example is weaker than the paper implies. Its circuit (a) mostly assembles the token
  "temperature" plus heat/thermal context, and (b) restores its neighbourhood as much as or more than the target.
- It's a good illustration of what circuits for word-level latents look like. It isn't evidence of computed
  concepts, and it would fail a specificity claim.
- **String-detector control (`string_control.py`, 16,376 random sequences, ~1M positions):**
  - it fires on **99%** of "temperature" tokens (326 / 328; the 2 silent ones are climate "temperatures"), so it
    is not sense-selective for the word;
  - but only **17%** of its firings are on "temperature". The rest are other physical-quantity words: temper, ice,
    pressure, gravity, cold, velocity, H, and also price.

  So it's a **"physical quantity / state variable" latent** with "temperature" as its strongest and most reliable
  trigger (activation up to 9.6 in thermodynamics text). That fits the circuit: the "temperature" subword
  detectors drive the strongest firings, while the price / parameter / heat nodes match the latent's broader
  firing on other quantity words.
- **Arm Bw** (rescored after a resume-key fix): 301 nodes, strongest 0.98 / 1.19 / 1.19, necessity 1.00,
  sufficiency to induce 1.04. It matches B.

## For the paper

- Either keep it as the running example, but described accurately ("a physical-quantity latent that fires on
  every *temperature* and on pressure, velocity, gravity, cold…", with the specificity caveat), or pick a showcase
  from the at-scale run that is specific
  (sibling faithfulness well below the target's) and conceptual (not dominated by string detectors).
- Selection rule for the showcase (to decide): specificity gap ≥ X, top nodes not ≥ 90% single-token, and
  ablation-stable wiring.
