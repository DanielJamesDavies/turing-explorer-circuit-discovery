# Cross-SAE replication: Pythia-70m-deduped + Marks et al. ReLU dictionaries
# (2026-08-10)

The paper's last run gate. Different model (Pythia-70m-deduped),
different corpus (wikitext-103), different SAE architecture (ReLU+L1
DENSE dictionaries, 512->32,768, saprmarks/dictionary_learning
10_32768) from everything in the paper's evidence (Top-K k=128).
Standalone harness (crosssae.py — no pipeline imports): GPTNeoX
submodule hooks, x + dec(c_hat - c) intervention, encoder
pre-activation seed reads, tri-amp transcription (triple floor
1.0/0.25/0.10, free amplitudes, leak guard, 400-step anneal), 48/16
held-out split, VERIFIED-INACTIVE negatives (zero firing in window —
stronger than home's unverified retrieval). 6 seeds (3 x resid L2,
3 x resid L4, fire-frac 2-5%), arms triamp400/gate400 + 3
amplitude-fitted nulls, at lambda 1e-3 (main) and 1.0 (compact
control; 1e-2 tested and inert — see finding 5). Data: rows.jsonl
(62 rows), scan.pt, crosssae.py.

## Verdict: THE METHOD AND CIRCUITS REPLICATE; THE RECONSTRUCTION
## NULL'S VALIDITY IS ARCHITECTURE-DEPENDENT

1. **Tri-amp replicates.** On 11/12 tri-amp rows the discovered
   circuit is both-floors faithful held-out (ampF0/ampFM 0.80-1.00)
   and necessary (sup 0.81-1.0). Compact examples with FULL fidelity:
   n=4 (L2/1253, 0.80/0.91/0.81), n=29 (L2/28259, 1.0/1.0/1.0 and
   drives 0.78), n=1,093 (L4/26044, 1.0/1.0/1.0, drives 8.6-overshoot
   at alpha). Gate-only also finds tiny faithful circuits here (n=2 at
   L2/1253: 1.0/1.0/1.0) — on dense ReLU dictionaries some latents
   have essentially direct one-to-two-latent explanations, i.e. the
   faithfulness cost on this architecture can be MINUSCULE. The one
   tri-amp failure is the bulk-regime L4/9548 at lambda 1e-3
   (n=142,729; sup explodes) — not a compact-circuit claim.
2. **The reconstruction null is architecture-dependent — the
   replication's headline methods finding.** Amplitude-fitted random
   sets: at compact n (<=~1,100) they FAIL decisively (F0 0.00-0.73,
   sup ~0, cf 0.0; 15/15 draws); at mid n (~2,000-6,000) they PASS
   reconstruction (F0 0.71-1.00) while failing necessity+drive; at
   bulk n (>=28k) nothing discriminates (the live pool is exhausted —
   at L4 n_ref exceeded the ~106k pool, making "random" = everything).
   Mechanism: dense ReLU codes are nonzero at the anchor positions,
   so thousands of free amplitudes can synthesize the target from any
   large live set. Top-K blocks this by construction (random latents
   are outside the top-k at the anchors -> amplitudes multiply zero),
   which is why 0/124 nulls passed at home. **Top-K sparsity is
   load-bearing for the reconstruction null; on dense SAEs, necessity
   and drive are the load-bearing exams, and ampF0 is quotable only
   at compact n.**
3. **Dense empty floors are extremely off-manifold**: e0 = 2,069-2,386
   at L2 and 4.3e7-1.8e8 at L4 against a_pos of 2-12 — the dense-SAE
   amplification of the home L10 finding. Denominators remain
   well-defined (huge separation), but drive injection at bulk sizes
   explodes (cf up to 3.8e7) — the drive eval is only meaningful for
   compact circuits on this architecture.
4. **Reconstruction-without-necessity exists at tiny n too**: a
   37-member gate circuit (L2/14800 @ lambda 1) scores F0 1.0 with
   sup 0.0001 — removable-yet-reconstructing, the mirror of the
   home closure-without-drive. One object, multiple exams, again.
5. **The lambda regime does not transfer**: 1e-2 was inert (n GREW on
   one seed — the off-manifold floor losses drown the L1 term at
   home-scale lambdas); lambda=1.0 bites (n 5,933->2,156; 47->4;
   5,043->29). Calibration is architecture- and seed-specific
   (per-seed sizes at fixed lambda span 4 to 72,174).

## Reporting rules

Quote the replication as: method + compact circuits + necessity
replicate on a public model with public dense dictionaries under a
held-out, verified-negative protocol; the reconstruction null is
valid there only at compact n (state the regime); drive evals are
compact-only on dense SAEs. 6 seeds, 2 layers, resid-only — a wider
panel is future work.

## Mechanism follow-up (2026-08-10 late): ANCHOR SUPPORT is the cause;
## eval-time top-k projection is NOT a repair (a recorded negative)

Two tests of *why* the dense null passes reconstruction at mid n.

**(a) NEGATIVE RESULT — eval-time top-k projection does not repair the
null** (topkeval.py, topk_rows.jsonl). Projecting the reconstructed
code to per-position top-k (k in 32/64/128) before decoding, with the
empty-circuit floors re-measured under each regime, leaves the nulls
essentially untouched on the zero fill: null ampF0 0.9887 -> 0.9513
(k32) / 0.9857 (k64) / 0.9877 (k128) at L2/14800, and the same
pattern on L2/28259 (0.975 -> 0.951-0.975). The prediction that
motivated the test was WRONG, and the reason is instructive: under a
ZERO fill the null's own members are the only nonzero entries, so a
top-k projection of the reconstructed code cannot remove them — it
keeps their largest fitted coefficients. Projection acts on the FILL,
not on the exploit. (Byproduct worth keeping: under the MEAN fill the
projection does separate the two — discovered stays 0.9995-1.0000
while nulls fall to 0.56-0.95 — because there the projection reshapes
the non-member mass.) Discovered circuits are robust to projection on
both fills at every k (0.9967-1.0004), which is itself a nice
on-manifold sanity result.

**(b) POSITIVE RESULT — the cause is the ENCODER's sparsity, measured**
(anchor_support.py, anchor_support.json). For each seed, the fraction
of LIVE latents that are nonzero AT THE PROBE ANCHOR (i.e. usable by a
fitted amplitude):

  L2/14800 0.088 | L2/1253 0.368 | L2/28259 0.123
  L4/3837  0.065 | L4/9548 0.129 | L4/26044 0.219
  Top-K expectation (k=128 of 40,960): 0.0031

6.5-37% versus 0.31% — a 20-120x difference. Under Top-K a random
latent is outside the top-k at the anchor almost always, so its
amplitude multiplies EXACTLY zero and the null is dead on arrival;
under ReLU it has purchase, and with thousands of coefficients the
target is synthesizable. This is the mechanism behind finding 2 above,
now measured rather than inferred.

Paper: sec:weighted quotes the 6.5-37% vs 0.3% contrast. The
projection negative is dev-record (it clarifies the mechanism but
changes no claim).

NEXT (planned): EleutherAI/sae-pythia-70m-32k — Top-K (k=32, 32,768
latents) SAEs on the SAME model, so the dictionary family is the only
variable. Registered prediction: discovered circuits stay
faithful/necessary AND fitted-random nulls fail at every n, restoring
the home behaviour on a public model.
