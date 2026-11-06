# Arena A: the WCM adapter for GPT-2 small + Dunefsky transcoders + Kissane z-SAEs (DAN-195)

Written 2026-11-03 (agent `arenaA`), with the code (`wcm_tcz.py`) and its CPU tests (`test_wcm_tcz.py`, 3/3 pass).
**No fit has run and no GPU was used.** Nothing under `src/`, `third_party/`, `experiments/062*` or `082*` is changed:
the family, site model and bank register at runtime, as arena C's `wcm_marks.py` does. Tags: **[ours]** = our choice,
**[decide]** = Daniel's call.

## 1. Sites

Family `gpt2-small-tcz`, HF `gpt2` (eager attention, float32), kinds `("z", "z2", "tc")`, every site 24,576 wide,
`k = d_sae` (whole ReLU code, as arena C).

| site | encode reads | edit lands on | dictionary |
|---|---|---|---|
| (L, `z`) | input of `h[L].attn.c_proj` = merged heads `[B, T, 768]` = TL `hook_z` flattened, head-major | the same tensor (forward pre-hook on `c_proj`); the z error passes through W_O | Kissane L, latents 0..24,575 |
| (L, `z2`), L ∈ {5, 7} | the same clean `c_proj` input | the same tensor; the `z` and `z2` edits add | Kissane L, latents 24,576..49,151 |
| (L, `z2`), other L | – | – (inert `NullSAE`) | – |
| (L, `tc`) | input of `h[L].mlp` (full `ln_2` output) | the **MLP output**: `mlp_out + (y − x_in)` = `mlp_out + (ĉ − c) W_dec` (035); the transcoder error passes through | Dunefsky L via 035's LN-folded `transcoders_ct/` |

- `same_layer_upstream = {z: (), z2: (), tc: (z, z2)}`; `capture_sites` order per layer is z, z2, tc.
- **The Kissane value-bias convention (measured 2026-11-03, `ge_checks.py format`, 8 × 127 OWT tokens).** The z-SAEs
  were trained on TL's default `fold_value_biases=True`, so their input is `z_TL = z_HF − b_V`. Centred encoder on
  `z_TL`: FVU 0.20–0.49, L0 3–21 at every layer. On `z_HF` (b_V kept): FVU 0.44–441, L0 up to 564. Uncentred: worse
  everywhere. The bank stores `decoder_bias = b_dec + b_V'`, so `b_eff = b_enc − W_enc (b_dec + b_V')` is exactly
  Kissane's encoder on `z_TL`, and `decode()` is the HF-space reconstruction. Edits are differences, so unaffected.
  **Fix 2026-11-05 (agent `arenaA2`):** TL's b_V is taken *after* its default `fold_ln=True`, so the offset is
  `b_V' = b_V + ln_1.bias · W_V` (`wcm_tcz.tl_value_bias`), not the raw HF `b_V`. With the raw b_V the adapter's z
  input kept `ln_1.bias · W_V`: on 8 smoke-corpus sequences centred FVU 21.6 / 21.3 / 12.9 at L0–L2 (L0 65–163), and
  zeroing the z codes blew the residual stream up (std 2 → 650 by L3). With `b_V'`: FVU 0.46 / 0.28 / 0.34 / 0.39
  (L0–L3), 0.51–0.81 at L6–L11, L0 3–23. The parity test could not catch it (same offset on both sides); new test
  `test_z_fvu_smoke_corpus` checks FVU < 0.75 at L0–L2 on real text.
- **Transcoder bias bookkeeping:** our SAE subtracts `decoder_bias` before encoding; `transcoders_ct` does not centre.
  The bank stores `decoder_bias = b_dec_out` and `encoder.bias = b_enc + W_enc b_dec_out`, so `b_eff = b_enc` exactly
  and `decode()` is the MLP-output reconstruction.
- **Why the fit is compatible:** `LearnedMaskPatcher` (learned_mask.py:507-604) only ever returns `x + Δcode W_dec`
  (one `decode(…, add_bias=False)` of a code difference) and returns `x` itself when a site has no gate. Both are
  exactly what the transcoder re-route needs; nothing in the fit forms `x − decode(c)` at a site. The seed tap reads the
  target's pre-activation from its own site input, which for a tc target is the MLP input: correct.
- **BOS [decide; default on]:** `ARENA_A_BOS=1` prepends `<|endoftext|>` inside `forward` and hides position 0 from every
  site. BOS is then a frozen live prefix (never a member, never ablated, never a root), as both dictionary sets were
  trained and as Ge (A14) and Dunefsky (A15) do. `ARENA_A_BOS=0` = no BOS.
- **Scopes:** `ARENA_A_SCOPE=a1` (z + z2 + tc, 26 real sites: 12 z, 2 z2, 12 tc) for Ge; `a2` (tc only, attention live)
  for Dunefsky.

## 2. CPU tests (`test_wcm_tcz.py`, 3/3 pass, 136 s, `.venv`)

1. **Splice:** the site model unpatched, and with a zero edit at every site of layers 0 / 5 / 7 / 11, equals the raw HF
   model with the same BOS prefix, **bitwise** (`torch.equal`).
2. **Edit semantics:** an edit d at (3, tc) equals adding d to `h[3].mlp`'s output; at (3, z) and (5, z2) it equals
   adding d to `c_proj`'s input; (5, z) + (5, z2) adds 2d (all ≤ 1e-4 on logits).
3. **Parity:** our z code (z and z2 concatenated at L5 / L7) equals Kissane's formula on `z_HF − b_V` (≤ 1e-5 × max);
   our tc code on the full `ln_2` output equals Dunefsky's `relu((x_norm − b_dec) W_enc + b_enc)` from the source
   pickles (≤ 1e-4 × max), layers 0 / 5 / 7 / 11.

Still to do: a 6-step tiny fit through the engine (CPU), the Z-empty diagnostic panel (arena C Finding 2), the data pass
(`data_pass_tcz.py`, as arena C's `data_pass_marks.py`: pass 1 + neg_ctx, no PMI pass at k = d_sae), and the GPU timing
pilot (a night).

## 3. Proposed src changes (none is needed for arena A; each would replace runtime glue, for review)

| # | change | replaces | benefit | risk |
|---|---|---|---|---|
| P1 | per-site `d_sae` in `SAESource` / `PublicSAEBank` / `learned_mask` / the stores | the `z2` half-site split | Kissane L5 / L7 as single sites; member ids need no remapping | wide: `d_sae` is used 223 times in 82 files (20 in `learned_mask.py`, every store); high regression risk for 062 / 082 |
| P2 | a native "read-input / write-output" site kind in `hf_adapters` (encode on a module's input, apply the delta to its output) | `TCZSiteModel.mlp_hook`'s `(y − x_in)` re-route | transcoder sites for any family (also D1's Gemma transcoders, DAN-203) | small; one new hook type |
| P3 | honour `SAESource.prepend_bos` (read by nothing today): the site model prepends BOS and hides position 0 | `TCZSiteModel.forward` / `_split` | any BOS-trained public dictionary | small, but touches the loader convention (`skip_first_token`) of 069 |
| P4 | per-kind site-space transforms in `PublicSAEBank.site_input` | the b_V-in-decoder-bias trick | clarity only | small |
| P5 | scorer audit: confirm `eval/wcm_eval.py` and the fills never use `x − decode(c)` at a site (true for the fit, section 1) | – | needed before trusting A / C / R scores on tc sites | read-only audit, not a change |

Recommendation: keep everything runtime for arena A (the tests pass without src changes); do P2 + P3 after the pod, when
DAN-203 needs the same hooks; P1 only if a third family needs mixed widths.
