# 130 Selectivity of the production WCM v4 circuits (CPU, analysis only)

## Question

Exp 129 found that the IOI per-name latents' circuits are role-blind. Every required test (Z / A / C sufficiency,
`nec_b` necessity) runs on held-out *activating* contexts, so a circuit can reproduce *how strongly* the target fires
without encoding *whether* it fires. Is that an IOI quirk, or is it true of the 2026-11-07 production circuits in
general?

Two selectivity reads are already in every production row and are reported but not in the pass rule. This
experiment summarises them.

## Answer

**It is general: most passing production circuits are not selective.** The test sets every member to its fitted
α × its own live value on the target's verified-silent held-out contrast contexts, with the rest of the stream taken
from *other silent contrast contexts* (donor pool C). Under that edit, the passing circuits switch the target on at a
large share of positions:
- **TuringLLM:** median **45%** of positions; 92% of passes fire at more than 1% of positions.
- **GPT-2 small:** median **7%**; 67% of passes.
- **Pythia-70m:** median **1%**; 50% of passes.

The full model fires at **0** of these positions (verified silent, `a_base_tk` max = 0 in every model). The empty
circuit, where every latent comes from the silent donor, fires at about 0.1%. So the false firing is caused by the
circuit's members at their fitted α, not by the background. With α = 1 (variant `a1`, same members) the firing
drops back to about 0 (1-3% of circuits above 1%).

Sufficiency to induce (φ_ind) is high: median 0.78-0.90, against 0.01-0.02 for random-member nulls. So the members
do carry the drive that switches the target on when they are set to their activating state. But they also switch it
on from their **silent-context** state. The circuits encode level (a gain on features that are also partly active in
near-miss contexts), not the gate. Only **0.8%** (TuringLLM), **4.0%** (GPT-2) and **28.6%** (Pythia) of passing
circuits are "selective both ways": φ_ind ≥ 0.5 and silent firing ≤ 1% in all three donor pools.

**Worst cases:**
- **By site kind:** attention latents. Their passing circuits fire on median 81-87% of silent positions in TuringLLM
  and GPT-2 (pool C), and their φ_ind is low (GPT-2 attention median 0.00).
- **By depth:** the deep band. Passing L8-11 circuits fire on median 68% (TuringLLM) and 25% (GPT-2) of positions;
  Pythia L4-5 is at 5.5%.
- **By size:** larger circuits. Spearman ρ(silent firing, n) is 0.47 / 0.62 / 0.75 for TuringLLM / GPT-2 / Pythia.

**Most selective:** shallow residual and MLP latents in Pythia and GPT-2. Pythia L0-1 has median 0.1%, and 54% of
its circuits are selective both ways.

The ordering across models (TuringLLM k = 128 > GPT-2 k = 32 > Pythia k = 16) is consistent with a lower Top-K
entry bar making false entry easier. This is a hypothesis; it is not tested here.

## Headline table (held-out strongest rows, non-vacuous, `merge.passes`; WCM v4 main run)

Silent firing (SF) = share of the 16 × 64 silent-base positions with target post-Top-K > 0, mean over 8 donor draws,
variant `fit` (members α × live). Pool C = silent-donor background, pool A = activating-donor background (the 48
strongest-train rows). φ_ind = `phi_cf_alpha_blind_tk`.

| model | group | n | φ_ind median (IQR) | φ_ind ≥ 0.5 | SF pool C median (IQR) | SF C > 1% | SF C > 10% | SF pool A median (empty-circuit A) | selective both |
|---|---|---|---|---|---|---|---|---|---|
| TuringLLM | pass | 6565 | 0.80 (0.53-1.01) | 77% | 0.45 (0.10-0.78) | 92% | 75% | 0.59 (0.10) | 0.8% |
| | fail | 899 | 0.57 (0.07-0.95) | 54% | 0.63 (0.06-0.93) | 87% | 72% | 0.71 (0.14) | 4.1% |
| | L0-4 (all) | 2931 | 0.81 (0.58-1.07) | 83% | 0.22 (0.03-0.55) | 86% | 62% | 0.41 (0.10) | 3.1% |
| | L5-7 (all) | 1981 | 0.73 (0.41-0.92) | 71% | 0.54 (0.13-0.83) | 93% | 78% | 0.62 (0.09) | 0.0% |
| | L8-11 (all) | 2552 | 0.79 (0.27-1.04) | 66% | 0.70 (0.31-0.90) | 96% | 86% | 0.78 (0.13) | 0.0% |
| | L8-11 pass | 2101 | 0.83 (0.33-1.07) | 69% | 0.68 (0.30-0.88) | 96% | 86% | 0.76 (0.12) | 0.0% |
| GPT-2 small | pass | 421 | 0.84 (0.29-1.17) | 67% | 0.07 (0.00-0.65) | 67% | 47% | 0.18 (0.07) | 4.0% |
| | fail | 90 | 0.66 (0.14-0.94) | 62% | 0.35 (0.01-0.77) | 73% | 62% | 0.52 (0.09) | 6.7% |
| | L0-4 (all) | 206 | 0.92 (0.47-1.28) | 72% | 0.01 (0.00-0.20) | 51% | 31% | 0.06 (0.06) | 11.2% |
| | L5-7 (all) | 130 | 0.77 (0.07-1.08) | 61% | 0.18 (0.01-0.81) | 73% | 57% | 0.38 (0.08) | 0.0% |
| | L8-11 (all) | 175 | 0.69 (0.05-1.01) | 63% | 0.47 (0.04-0.77) | 83% | 67% | 0.59 (0.09) | 0.0% |
| | L8-11 pass | 115 | 0.83 (0.14-1.07) | 66% | 0.25 (0.02-0.69) | 79% | 61% | 0.44 (0.08) | 0.0% |
| Pythia-70m | pass | 416 | 0.90 (0.53-1.10) | 76% | 0.010 (0.001-0.10) | 50% | 25% | 0.03 (0.08) | 28.6% |
| | fail | 92 | 0.58 (0.07-1.05) | 55% | 0.55 (0.17-0.73) | 85% | 79% | 0.63 (0.09) | 5.4% |
| | L0-1 (all) | 127 | 1.01 (0.83-1.12) | 86% | 0.001 (0.000-0.007) | 19% | 6% | 0.004 (0.07) | 54.3% |
| | L2-3 (all) | 191 | 0.85 (0.43-1.10) | 71% | 0.02 (0.00-0.17) | 56% | 31% | 0.05 (0.09) | 26.2% |
| | L4-5 (all) | 190 | 0.71 (0.24-1.07) | 64% | 0.22 (0.02-0.61) | 81% | 58% | 0.36 (0.08) | 2.6% |
| | L4-5 pass | 149 | 0.82 (0.46-1.09) | 72% | 0.055 (0.01-0.36) | 75% | 47% | 0.13 (0.08) | 3.4% |

Passing circuits by site kind (SF pool C median / φ_ind median):

| | resid | resid_mid | mlp | attn |
|---|---|---|---|---|
| TuringLLM | 0.45 / 0.96 | n/a | 0.14 / 0.66 | 0.87 / 0.43 |
| GPT-2 small | 0.02 / 1.14 | 0.02 / 1.13 | 0.03 / 0.61 | 0.81 / 0.00 |
| Pythia-70m | 0.003 / 1.03 | n/a | 0.005 / 0.76 | 0.16 / 0.17 |

**References and baselines:**
- **Full model:** SF = 0 by construction, and φ_ind = 0 (natural value at the contrast anchors).
- **Empty circuit:** SF pool C ≈ 0.001 and pool R ≈ 0.001 in every model. With pool A, the activating background
  alone fires at median 7-10% of positions.
- **Harder nulls (084; draw 0, held-out; only φ_ind, no donor test):**

  | | WCM (null run) | WCM α = 1 | random (dan17) | anchor-active random | contribution-matched |
  |---|---|---|---|---|---|
  | TuringLLM (112) | 0.77 | 0.63 | 0.01 | 0.02 | 0.52 |
  | GPT-2 (128) | 0.81 | 0.55 | n/a | 0.01 | 0.18 |
  | Pythia (126) | 0.93 | 0.87 | n/a | 0.00 | 0.16 |

- **Pythia unweighted circuits** (`donor_unweighted`, the same 127 seeds): SF pool C median 0.003 vs 0.013 for WCM,
  and SF C > 1% in 31% of circuits vs 53% for WCM. So the unweighted circuits are more selective here. They are also
  much larger, and their pass rate is lower.
- **α:** the median α per circuit does not predict SF (ρ −0.13 to −0.26). Circuit size does (ρ 0.47-0.75). Setting α
  to 1 removes SF in pool C, but α = 1 circuits fail Z / A / C (2.7% pass on TuringLLM, 084 nulls).

## Examples (passing circuits; `out/worst.csv`)

| model | key | layer / kind | n | φ_ind | SF C | SF R | SF A (empty A) | Z / A / C / nec |
|---|---|---|---|---|---|---|---|---|
| TuringLLM | 6.attn.20737 | L6 attn | 332 | 0.55 | 1.000 | 1.000 | 1.000 (0.14) | 0.95 / 0.93 / 0.95 / 1.00 |
| TuringLLM | 9.attn.36674 | L9 attn | 444 | 0.11 | 1.000 | 1.000 | 1.000 (0.95) | 0.88 / 0.91 / 0.91 / 1.00 |
| GPT-2 small | 7.attn.19455 | L7 attn | 686 | 0.00 | 0.991 | 0.992 | 0.995 (0.15) | 0.83 / 0.85 / 0.82 / 1.00 |
| GPT-2 small | 2.attn.12077 | L2 attn | 268 | 0.39 | 0.998 | 0.987 | 0.998 (0.18) | 0.94 / 0.96 / 0.96 / 1.00 |
| Pythia-70m | 4.attn.31323 | L4 attn | 938 | 0.00 | 0.936 | 0.885 | 0.923 (0.04) | 0.89 / 0.90 / 0.91 / 1.00 |

These circuits pass every required test, yet they turn the target on at almost every position of contexts where the
model keeps it off. Two of them (GPT-2 7.attn.19455, Pythia 4.attn.31323) also have φ_ind = 0: setting the members
to their activating pins does not raise the target at the contrast anchors above its natural value. The lowest-φ_ind
passing circuits are all large attention circuits (2,000-5,000 members, e.g. TuringLLM 11.attn.7486, n 5077, φ_ind
0).

## Caveats: what these reads do and do not measure

- **How the silent contexts are chosen.** They are "close" near-misses, not unrelated text (protocol §3.4,
  `NegContextSelector` "close").
  - Candidates are ranked by maximum cosine similarity, in mean-pooled final-layer `seq_repr` space, to the target's
    top-context store.
  - A candidate is kept only if the target's post-Top-K activation is 0 at every one of its 64 positions.
  - The first 64 kept are split by stratified order into 48 training and 16 held out; SF and φ_ind use the 16 held
    out. All scored rows have `ctr_source = close`.
  - These are similar documents, not minimal pairs: nothing guarantees that they hold the target's trigger token in
    the wrong role (the IOI case in 129).
  - The target's natural pre-activation at the contrast anchors is small: median 1.6% (GPT-2), 4.5% (TuringLLM) and
    6.6% (Pythia) of `a_pos_pre`. It is above 0 in 85-99% of rows, but outside the Top-K.
- **SF counts positions, not magnitude.** It is the share of the 16 × 64 positions where the target enters the Top-K.
  How strongly it fires there is not stored (`raw.shard*.jsonl` keeps only anchor reads of the activating bases).
  All 64 positions count, including the early positions of each window.
- **SF uses the fitted α.** That is the circuit as scored in Z / A / C, so it is the fair read of the reported
  circuit.
  - The pool-C version keeps the whole stream "silent". Non-members come from verified-silent contrast rows (pool C
    is the same 16 held-out contrast rows, drawn by `draw_selections`; whether a base can draw itself was not
    checked), so any firing comes from the members.
  - Pool A (activating-donor background) is harder: there the empty circuit already fires at 7-14%. Pool R (random
    windows) is the easiest.
- **φ_ind is not a selectivity test on its own.** It asks whether members set to their activating pin (α × the mean
  at the anchors of the 48 strongest-train rows, position-uniform, rest of the stream live) switch the target on. A
  level-only circuit can score high on it, and many do. It complements SF: φ_ind high and SF low is the selective
  pattern.
- **The full model is not run through the donor test as a "circuit".** Its SF reference is 0 by construction of the
  contrast set, and the self-donor identity check (`self_empty`) is in the rows.
- **Population.** This is 109's headline population: held-out strongest rows, error rows dropped, keep-last,
  `merge.vacuous` excluded, pass = `merge.passes`. TuringLLM has 7,464 scored, GPT-2 511, Pythia 508.
  - The 124 paraphrase-filtered re-score is a 300-target subset check that the headline does not apply, so it is not
    applied here.
  - Donor rows join every scored target (TuringLLM T11 `donor/main`, 8,000 rows; public `donor/`, 512).
  - Harder nulls exist only for 112-128 targets per model, with φ_ind only.
- **The α tercile analysis** (`out/report.txt`) does not show a monotone α effect. The link to α comes only from the
  `fit` vs `a1` contrast, which changes every α at once.

## Files

- `sel130.py`: the analysis. It reads `experiments/109-pod-run-analysis/out/archives` through `lib109` (with
  `merge.passes` / `vacuous` and the 084 `report.py` frame for nulls). CPU only, under 1 GB, about 1 minute.
- `out/summary.csv`: every group × read (median, p25, p75, p90, threshold shares; SF for fit / a1 / empty × A / C / R).
- `out/nulls_ind.csv`: φ_ind of the 084 null arms.
- `out/per_target.csv`: per target: band, kind, n, pass, α median, φ_ind, SF per variant × pool, Z / A / C / nec.
- `out/worst.csv`: the highest-SF and lowest-φ_ind passing circuits per model.
- `out/report.txt`: population checks, Spearman correlations, the α tercile split, and the Pythia unweighted paired
  comparison.

Run: `python -X utf8 experiments/130-selectivity-production/sel130.py`
