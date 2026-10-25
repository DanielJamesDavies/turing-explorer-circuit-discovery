# 072 — Random-example gallery (DAN-162)

Circuits for **randomly drawn** production targets, shown next to the curated case study (Nanda / Carlini: show
random examples, not only the best ones). Paper: App. "Circuits for Randomly Drawn Targets" (`app:random-gallery`,
`tab:random-gallery`), pointed to from §5.3.

**Design (Daniel, 2026-10-12):**

* **Sample:** 9 targets, one per depth band (L0-4 / L5-7 / L8-11) × site kind (attn / mlp / resid).
  * Drawn uniformly from the non-vacuous circuits of each cell, `SEED` 20261012, any outcome.
* **Per target:**
  * scores (Z / A / C, necessity, induce with α, size, pass, amplifier, near-threshold);
  * its label from the 065 circuit describer (Sonnet, prompt v5);
  * its top 3 members by contribution share at the anchor, hubs excluded.
* **Member labels:** the 065 CLI call shape with a *latent-only* report.
  * Input: the member's own peak tokens, its distinct strongest windows (deduplicated, since the stores hold duplicated text, DAN-157), and its logit effect.
  * These are hypotheses, not verified readings.
  * Model: `claude-sonnet-5-5`.
* **Source:** the explorer bundle only (`outputs/explorer_bundle/2026-09-27_062-full`); no GPU.

```
PYTHONPATH=src python experiments/072-random-gallery/gallery.py      # DRY=1 prints one member prompt
```

**Outputs (`results/`):**

* `gallery.md`: for review;
* `gallery.json`;
* `member_labels.jsonl`: the label cache, so reruns don't call the CLI again;
* `gallery_table.tex`: the table rows pasted into `paper/main.tex`.

**First draw (seed 20261012):**

* 3.attn.9767 pass
* 0.mlp.12557 fail
* 0.resid.27712 fail
* 5.attn.10168 fail (near-threshold)
* 5.mlp.15947 pass
* 7.resid.29770 pass
* 9.attn.38891 fail (A undefined)
* 8.mlp.2453 fail
* 9.resid.10786 pass

That is 4/9 passing, against 46% for the run. 5 of the 6 targets at L5-11 are amplifiers.
