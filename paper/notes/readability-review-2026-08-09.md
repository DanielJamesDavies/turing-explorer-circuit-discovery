# Readability Review — main.tex (2026-08-09)

A cold-reader pass over the full draft (body + appendices), focused on *how the paper reads*, not on whether the results hold. Line references are to `main.tex` at today's state. Counts quoted below were measured, not estimated.

**Verdict in one paragraph.** The paper has a genuinely strong spine — one object, two criteria, and the control/faithfulness split is stated early, carried consistently, and paid off in the Discussion. The intellectual organisation is not the problem. The problem is prose density: nearly every sentence is a compound claim with one or two em-dash asides, the abstract and contributions are written for someone who has already read the paper, and the Results prose in places compresses five findings into one paragraph of parenthetical numbers. A sympathetic expert can follow it with effort; a tired reviewer skimming at NeurIPS pace will bounce off the abstract and never recover. The fixes are mostly mechanical — deconstruct sentences, ration the asides, re-introduce terms at first body use — and none of them touch the science.

---

## What already works — keep these

- **The organising thesis and its discipline.** "One object, two exams" (§2.1) is a clean frame, and the paper genuinely honours it: every score names its semantics, Table 1 (`tab:semantics`) is an excellent anchor the reader can return to, and the Discussion cashes out exactly what the Results promised.
- **The opening of §5.1** (lines 570–573): "Which latents can impose a seed's activation? A handful. How many must remain...? Tens of thousands." This is the best-reading moment in the paper. It earns its punchiness because it lands the central result in fifteen words.
- **Several genuinely quotable sentences.** "A latent's activation, reached through an additive residual stream, is a dense function of many small contributions, and no sparse threshold captures a dense function" (line 742). "Circuit size is a measurement, not a knob." These carry the paper's voice at its best.
- **The Limitations section** is unusually honest and specific (named protocol debts, named open gates) — reviewers reward this.
- **Related Work** is well-positioned: the build-on tone, the explicit "we adopt unchanged and credit as theirs" for the IG estimator, and the 2×2 endpoint grid all read as confident rather than defensive.
- **The appendix negative results** (§H.3, binarisation sweep, floor studies) are clearly written and valuable — ironically, some of the appendix reads *better* than the Results body because each paragraph carries one finding.

---

## Major readability issues

### 1. Em-dash density: the prose never rests

There are **278 em-dashes** across roughly 970 sentences — and they concentrate in the abstract, intro, and Results, where many sentences carry two asides, some nested. Example (abstract, line 78):

> "That cost is indexed by amplitude semantics: granting each member a learned scalar amplitude collapses the faithful membership by one to two orders of magnitude --- a compact weighted circuit passes both criteria --- and survives a matched null in which random same-size sets receive identically fitted amplitudes and fail on every draw."

That is four claims (indexing, collapse magnitude, dual-criteria pass, null survival) in one sentence. Each is important; none gets its own sentence. The cumulative effect over ten pages is real fatigue: the reader is never handed a plain declarative sentence to consolidate on before the next qualification arrives.

**Fix:** a dedicated pass whose only job is sentence fission. Rule of thumb: at most one em-dash aside per sentence, and no sentence carrying more than two claims. The information content survives; the paragraphs get slightly longer and dramatically easier.

### 2. The abstract assumes the paper has already been read

Terms that appear in the abstract before any reader could know them:

- "TuringLLM" (line 73) — unglossed. A reviewer will wonder whether they should know this model. One clause fixes it: "a 254M-parameter transformer trained on a fully documented synthetic corpus".
- "indexed by amplitude semantics" (line 78) — opaque as a first exposure. What it *means* is "the cost depends on whether members keep their natural strengths or may be rescaled." Say that; introduce "amplitude semantics" as a term later.
- "size-calibrated families" (line 79) — meaningful only after §5.5.
- "position-local, and depth-scaled" (line 77) — "depth-scaled" is guessable, "position-local" is not, and the body itself never crisply defines it (see issue 8).

**Fix:** rewrite the abstract's first two-thirds in plain language and let the coined terms debut in the body where they can be defined. The abstract currently optimises for compression; it should optimise for a reader deciding in ninety seconds whether to continue.

### 3. The contribution list is six compressed abstracts, not six claims

Each bullet (lines 193–205) is a paragraph-length sentence with embedded caveats, sub-claims, and citations. Example: the first bullet packs the silence-attribution idea, the pre-activation mechanism, bidirectional intervention validation, the controlled-variable framing of negatives, *and* the random-beats-close result into one sentence. A reviewer skimming contributions extracts nothing quotable.

**Fix:** one crisp declarative sentence per contribution (what we did / what we found), followed by at most one supporting sentence. The caveats live in the body — the contribution list is the one place where hedging costs more than it buys.

### 4. Results prose in §5.2 reads as a compressed lab notebook

The weighted-circuits section is the newest material and it shows. The "budget is part of the method" paragraph (lines 637–642) contains: the convergence-by-100-steps finding, the drive-overshoot finding with two calibration numbers, the layer-9 case study with three numbers plus a parenthetical three-number sweep, the budget-vs-price membership divergence, the amplitude-only ablation, and the zero-term-removal explosion — six findings, one paragraph. The reader cannot tell which number is load-bearing and which is supporting.

Similarly, line 628: "zero-fill faithfulness ranges $0$ to $4.9\times10^4$ (median ${\sim}21$) against the discovered circuits' ${\sim}1$" — on first read, "faithfulness of 21" looks like a typo. §4.7 explains unclipped overshoot for $\phi^{\mathrm{cf}}$, but nothing warns the reader that *free* faithfulness is also unbounded and that "far from 1 in either direction = failure." One sentence at first occurrence fixes this.

**Fix:** for §5.2 (and the §5.5 bullets, which are better but still dense): lead each paragraph with the finding in words, then give the two or three numbers that establish it, and push the rest to the appendix. The section currently proves everything and emphasises nothing.

### 5. Catchphrase repetition

Measured counts: "the two criteria the word 'circuit' conflates" and variants — **5 occurrences** (abstract, intro, Fig. 1 caption, §4.1, Discussion). "Orders of magnitude" — **12**. "Home-turf" — **5**. "A method wins the metric whose semantics it trains" — 4 near-verbatim statements (§4.5, §5.3, Discussion, App. I). The first use of each is strong; by the fourth, the reader experiences it as a tic and starts discounting it. The conflation line in particular is the paper's opening move — repeating it verbatim makes the later sections feel like they are re-arguing the intro rather than building on it.

**Fix:** keep each catchphrase once at full strength (conflation: intro; home-turf: §5.3; wins-its-own-metric: Discussion), and elsewhere refer back briefly ("the home-turf effect of §5.3") or rephrase.

### 6. Register wobbles

Two kinds:

- **Infomercial second person** in Methods (lines 442–447): "Want the members that separate firing from almost-firing? Train the negative-context floor. Want a set that suffices alone? Train the zero floor. Want both...?" Three rhetorical questions in a row, in the most technical subsection of the paper. The device works once (§5.1's opening); here it reads as a different author.
- **Internal-project voice** in the body: "a complete head-to-head ... is queued" (line 650), "a threshold sweep is owed" (line 711), "what remains owed there" (line 793). The honesty is exactly right — but "queued" and "owed" are standup-meeting words. Neutral phrasing ("we defer X to the full-scale run"; "a threshold sweep is required before any ranking claim") carries identical content without sounding like a status report.

### 7. Cross-reference and caption load

**148 `\cref` calls** in the body; several sentences carry three. The Fig. 1 caption (lines 177–180) is a mini-section that itself contains five cross-references and restates the conflation thesis; the `fig:drivers-closure` caption (~120 words) largely duplicates body text. Long captions are defensible for skimming reviewers, but combined with the cross-reference density the paper constantly points the reader elsewhere mid-sentence.

**Fix:** prune crefs that point to where the reader already is or just was; cap captions at the figure's *reading instructions* (what the axes are, what to notice) and let the body carry interpretation. Also check the rendered form of mid-sentence `\cref{eq:p2}` — "ablation faithfulness \cref{eq:p2}" (line 179) likely renders as "ablation faithfulness Equation (P2)", which does not parse as English.

### 8. Small internal inconsistencies a cold reader trips on

- **"A handful" vs $10^2$–$10^3$.** §5.1 opens "Which latents can impose a seed's activation? A handful." — but (P1) solutions are stated as $10^2$–$10^3$ latents (line 303, Fig. 1). Hundreds is not a handful; the rhetorical overstatement invites a pedantic reviewer comment. "A few hundred" preserves the punch and the truth.
- **"Position-local" is asserted, never defined.** It headlines the abstract and central finding, but §5.1 supports it only via the 10%-of-positions results; no sentence says what position-locality *means* as a property. One defining sentence at line 581 would fix it.
- **(P1) vs (P2) asymmetry.** P2 gets a numbered equation; P1 is prose with an inline "(P1)" tag. Fine as a choice, but the referencing style then mixes plain "(P1)" with `\cref`'d "(P2)" — worth making uniform.

### 9. Spelling inconsistency: -ise vs -ize

**41 British forms** (optimised, optimiser, normaliser, binarisation, linearisation, characterisation...) against **10 instances of "optimization"** (American), sometimes in the same breath: line 98 "a circuit *optimised* for a single internal feature" vs line 74 "an *optimization* problem". If "circuit discovery as optimization" is being kept as Edge Pruning's term of art, that defends the section heading but not the running text. Pick British throughout (the majority) and normalise.

---

## Draft-hygiene checklist (known, but consolidated here)

- Missing figures: `sae-quality.pdf` (line 955), `gradient-size-curve.pdf` (line 1372) — placeholder boxes in the build.
- The v2-rewrite banner comment (lines 58–63) and per-section STEP/GATE comments — strip before circulation.
- Several numbers are flagged in comments as stale vintage (drivers/closure magnitudes, method-comparison numbers, the calibration exponent) — the body's "directional" hedges cover this, but the depth-stratified rerun is what actually closes it.
- **Correspondence email**: the footnote says `danieljamesdavies12@gmail.com` (line 50); verify this is the intended address — it differs from the account email on this machine.

---

## Prioritised recommendations

1. **Rewrite the abstract and contribution list for a cold reader** (issues 2, 3). Highest leverage per hour: these are the only parts most reviewers read carefully.
2. **Sentence-fission pass over §5.2 and §5.5** (issues 1, 4): one finding per paragraph lead, load-bearing numbers separated from supporting ones, and a one-line explanation of unbounded free scores at first occurrence.
3. **De-duplicate the catchphrases** (issue 5): one full-strength statement each, back-references elsewhere.
4. **Register pass** (issue 6): remove the second-person run in §4.6 and the "queued/owed" project voice.
5. **Mechanical sweep** (issues 7–9): -ise normalisation, cref pruning, caption trimming, "handful" → "a few hundred", define "position-local", unify (P1)/(P2) referencing.

None of these change a claim. The paper underneath the prose is coherent and the argument structure is genuinely good — the current draft simply makes the reader work for what the authors already earned.

---

## Change ledger: what has been done, what remains

Statuses cross-checked against `resources/rewrite-tracker.md` and `rewrite-tracker.md` (2026-08-09). Legend: ✅ done · 🔶 partially done · ⬜ to do · 🔒 gated on runs/data.

### Rewrite arc — completed

| Change | Status | Notes |
|---|---|---|
| v1 → v2 body rewrite (7 steps: skeleton, intro/contributions, related work, method, results, discussion/limitations/conclusion, appendices) | ✅ | Completed 2026-07-30; v1 backed up at `backup-2026-07-30-pre-rewrite/` |
| V3 length diet, phases R1–R5 (method merges, semantics table, results trims, discussion 8→5, running example, Q&A beats, targeted em-dash pass) | ✅ | Body ~20pp → ~17pp; the ~12.5pp aspiration was not reached — decision deferred |
| Figure 1 TikZ overview (3 rounds of polish) | ✅ | Gap-audit item 1 |
| Formal problem statement §4.1 (P1 / P2 / P2★, families Φ_ε) | ✅ | Gap-audit item 2 |
| Semantics taxonomy table (`tab:semantics`) | ✅ | Gap-audit item 6 |
| Contribution bullets compressed to 1–3 rendered lines | ✅ | Gap-audit item 5 — but see review issue 3: still too dense per line |
| Terminology revision: one object + two criteria; "closure" → "faithfulness"; n_ε = faithfulness cost | ✅ | 2026-08-07; labels kept for cref stability |
| Tri-amp content pass (weighted circuits §5.2, triple floor, α eval rows, fitted null, fifth underdetermination probe) | ✅ | 2026-08-07; panel numbers refreshed 2026-08-09 (22 seeds, held-out, 124 nulls) |
| Centrepiece figure v2 (3 seeds/band, median + ghosted traces, weighted-circuit diamonds) | ✅ | 2026-08-09; figure data gate closed |
| Tier-1 body cuts: Setting table → App A; co-activation figure → appendix | ✅ | First two of the tier-1 list |

### Open — content and structure (pre-existing, from the tracker)

| Change | Status | Notes |
|---|---|---|
| Definitive method-comparison matrix (current recipe, + tri-amp arms, held-out split) | 🔒 | Gates §5.3 numbers, boundary re-run, cost table; ~2h run |
| Depth-stratified 128-seed PA run | 🔒 | Gates the §5.1 drivers/faithfulness magnitudes (currently 14/23-Jul vintage) |
| λ re-anchor under anneal | 🔒 | Gates the calibration exponent (kept symbolic in §4.6) and size-matched comparisons |
| Cross-SAE / replication run | 🔒 | Named in Limitations as priority |
| Case study promotion (temperature seed, annotated circuit figure) | ⬜ | Gap-audit item 3; renderer built, two seeds selected |
| Held-out split as default protocol (beyond the weighted-circuit panel) | 🔶 | Adopted for §5.2 panel; not yet in the matrix runs |
| Regenerate `sae-quality.pdf` | ⬜ | Lost in 2026-07-22 teardown; placeholder box in build |
| Regenerate `gradient-size-curve.pdf` (or cut with its appendix) | ⬜ | Same teardown |
| Report circuit sizes as % of upstream dictionary (Daniel's 2026-07-31 decision) | ⬜ | Not yet applied to intro magnitudes, §5.1, Fig 1 annotations |
| Abstract headline numbers (2–3) once runs land | 🔒 | Gap-audit item 8 |
| Remaining tier-1 cuts (grid table → appendix, formulation compression, intro trims, etc.) | ⬜ | If the ~12.5pp body target is still wanted |
| Cost/scaling table; practitioner-recommendations close; Tracr-note in Limitations | ⬜ | Gap-audit items 7, 10, 9 — optional |
| Strip draft banners, STEP/GATE comments, "queued/owed" tripwires | ⬜ | At submission time |

### New — from this readability review

| Change | Status | Priority |
|---|---|---|
| Abstract rewrite for cold readers (TuringLLM glossed; coined terms replaced with plain language; concrete output-endpoint opener), then length-calibrated against SFC 115w / ACDC 151w / Cunningham 195w / COLM 196w / Edge Pruning 255w: cut ~290w → ~220w, 12 → 10 sentences | ✅ | 1 |
| Contribution list: one crisp claim sentence + one support sentence per bullet (bullets 1, 2, 3, 5 split; 4 and 6 were already two sentences after the em-dash pass) | ✅ | 1 |
| Sentence-fission pass on §5.2 and §5.5 (one finding per sentence: triple-result opener, null, host-kind, and budget paragraphs split in §5.2; all five probe bullets and the closing split in §5.5) | ✅ | 2 |
| Explain unbounded free scores at first occurrence (note added at the φ-free definition in §4.7 + a back-reference beside the null's "median ~21" in §5.2) | ✅ | 2 |
| De-duplicate catchphrases: conflation now 1 full-strength use (intro); "wins the metric whose semantics it trains" now 1 (the §5.3 home-turf definition), others are back-references or varied; "orders of magnitude" 12 → 7, all quantified (vague intensifiers replaced with concrete numbers) | ✅ | 3 |
| Register pass: §4.6 "Want X? Train Y." run converted to declarative; "owed" ×2 → "required"/"still open"; "Protocol debts" → "Protocol gaps"; §5.3's "queued" already removed by the matrix rewrite; stale "pending... final cross-method matrix" clause dropped from Limitations (the matrix landed 2026-08-10) | ✅ | 4 |
| Global em-dash removal (was 278; now 0 in prose — only TikZ comment separators and table no-data markers remain; build verified green, 35pp) | ✅ | 4 |
| Spelling normalisation: "optimization"/"factorize" → British throughout (only the verbatim TuringLLM completion quote keeps American) | ✅ | 5 |
| "A handful" → "A few hundred" in §5.1 opener | ✅ | 5 |
| Define "position-local" at first use (defining sentence added after the layer-7 example in §5.1) | ✅ | 5 |
| Unify (P1)/(P2) referencing: appositive `\cref{eq:p2}` → `\labelcref` (renders "(P2)", hyperlink kept); sentence-subject uses render "Equation (P2)"; zero "Eq. (P2)" remain | ✅ | 5 |
| Caption trims: Fig 1 and centrepiece captions split into sentences; scoped cref prune run (9 removed: back-pointers to the just-read subsection, repeat pointers within a subsection, one duplicate caption cref, one pure navigation pointer) — all evidence, roadmap, and definition pointers kept | ✅ | 5 |
| Break the Conclusion's ~100-word semicolon chain into short sentences (done during the em-dash pass) | ✅ | 5 |
| Thread the running example beyond §5.1: §5.2 and §5.5 sentences added from a dedicated single-seed run (dev-notes/data/running-example-2026-08-10; 192-member weighted circuit, Jaccard 0.60 between budgets); §5.6 deliberately skipped (would need an SFC replication run for a matched logit task) | ✅ | 6 |
| Verify correspondence email — confirmed by Daniel 2026-08-10: `danieljamesdavies12@gmail.com` is the work email and is correct in the footnote | ✅ | 6 |
