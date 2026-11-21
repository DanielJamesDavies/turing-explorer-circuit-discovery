# Random gallery, production run (seed 20261012)

Run: 6565/7464 = 88.0% pass (floor 0.75); amplifier 29%; near-threshold 8%. Gallery: 8/9 pass.

## 3.attn.29966 (L0-4, attn) — Social/political theme words in music-society text
PASS; 646 latents; Z 0.95 A 1.02 C 0.99; necessity 1.00; induce 1.38; 50 hubs dropped; trigger word_family, condition subject_area
> Fires on abstract social-topic nouns or adjectives (' political', ' change', ' interaction(s)', ' ideas') in lists of social/cultural themes, especially in text about music as a vehicle for social change. Silent on 'political' in other generic social-systems contexts.

- 2.resid.24439 (share 1.8%, α 1.12, spec 1.7): **Music theory terms** — Fires on tokens within music-theory words (major, chord progressions, tonality, tonal) in Western music contexts; predicts music-related next tokens like 'musical', 'music', 'melody'.
- 2.attn.4424 (share 1.4%, α 1.05, spec 2.3): **Music-context noun, predicts music** — Fires on nouns ending music-related phrases (harmony, structures, expressions) in music-theory text, and promotes music-related next tokens like "musical", "Music", "music".
- 0.mlp.17118 (share 1.4%, α 2.02, spec 1.2): **Network/system word pieces (unclear)** — Fires on mixed word-fragment tokens (' organ', 'connected', ' Boolean', ' cent') in network/city/system contexts; weak consistency. Logits mildly favor ' networks', ' nodes', ' relationships'.

## 0.mlp.12578 (L0-4, mlp) — Essentially/simply/merely adverbs
PASS, amplifier; 39 latents; Z 1.11 A 1.43 C 1.35; necessity 0.94; induce 0.35; 7 hubs dropped; trigger word_family, condition none
> Fires on hedging/minimizing adverbs such as " essentially", " simply", " merely" and " literally", mostly directly before a verb or predicate (means, represent, serves as). Members are mainly detectors of these adverb tokens.

- 0.attn.24285 (share 7.7%, α 2.78, spec 36.4): **Token ' essentially' (unclear output)** — Fires only on the token ' essentially' (6 of 6 peaks). The evidence is thin, and the logit effects are mixed with no clear next-token pattern, though ' essentially' itself is boosted.
- 0.attn.4540 (share 6.1%, α 1.28, spec 2.1): **Document-start token (unclear)** — Fires only on the document-start <s> token (6 of 6 peaks, one example window). Logit effects are mixed and don't show a clear next-token pattern, so the function is unclear.
- 0.attn.16863 (share 5.7%, α 2.93, spec 131.5): **' simply' token** — Fires on the token ' simply' (all 6 peaks, e.g. "simply means that instead…"). Logits boost ' simply' itself and are otherwise mixed, so the output effect is unclear.

## 0.resid.27712 (L0-4, resid) — Biochemistry word pieces (chrom, omer, II)
PASS; 184 latents; Z 0.94 A 0.99 C 0.98; necessity 1.00; induce 1.54; 13 hubs dropped; trigger word_family, condition subject_area
> Fires on word pieces in biology/chemistry science terms, mainly ' chrom' (chromosome/chromatography), 'omer' in Isomerase/Isomerism, ' II' in meiosis phases, and ' pol' (poles). Needs a technical science context; silent on abstract physics text.

- 0.mlp.13729 (share 6.0%, α 1.68, spec 2.6): **Biochemistry term fragments** — Fires on word fragments inside cell-biology/biochemistry terms (bisph-osph-ate, Kre-bs, cell-ular); promotes cell/biology next tokens like cells, memb, bio, protein. Few examples, so somewhat tentative.
- 0.mlp.205 (share 5.2%, α 0.95, spec 55.7): **" chrom" word-start fragment** — Fires only on the word-fragment token " chrom" (start of chromatography, chromatid, chromosome, chromatin), and it predicts continuations like "atin", "omer", "atch".
- 0.mlp.4393 (share 3.7%, α 0.99, spec 2.3): **"fect" in "Perfect" tense** — Fires on the token "fect" in "Perfect" within grammar headings like "Future/Past/Conditional Perfect Continuous". Predicts little clearly; weak push toward "-", "(", "is".

## 5.attn.10033 (L5-7, attn) — 'by' after 'replaced'
PASS; 566 latents; Z 0.90 A 1.01 C 0.91; necessity 1.00; induce 1.16; 38 hubs dropped; trigger word_family, condition construction
> Fires on ' by' right after "replaced" (passive replacement, e.g. "traditional methods were replaced by"), and weakly on the following word. Silent on ' by' after other verbs like "driven by" or "monitored by".

- 4.resid.32748 (share 4.0%, α 1.42, spec 68.7): **replaced [by/with]** — Fires on 'by' or 'with' right after a form of "replaced" (replaced by/with). Logits mildly favor words like 'newer'/'new' as the replacement, but the evidence is thin (6 examples).
- 3.resid.35172 (share 2.8%, α 2.17, spec 34.4): **Replacement words (replacing/with/of)** — Fires on replacement-related tokens, mainly " replacing", also " with" after "replaced" and " of" after "replacement". Predicts the thing replaced, such as " old", " original" or " existing".
- 4.resid.9953 (share 1.9%, α 3.00, spec 3.0): **Mixed function words, unclear** — Fires on mixed tokens (' at', ' with', ' lif', ' model') mostly after passive verbs or mid-phrase in technical/business prose; peaks are inconsistent and logits are unclear.

## 5.mlp.16658 (L5-7, mlp) — Word-start piece " ret"
PASS, amplifier; 336 latents; Z 0.98 A 1.03 C 1.04; necessity 1.00; induce 0.64; 27 hubs dropped; trigger one_token, condition none
> Fires on the word-start piece " ret" (retailer, retail, retention, retina), where more letters must follow. Members are detectors of " ret" plus related word-start pieces. Weak activity on "qui" and "df".

- 4.resid.13189 (share 4.2%, α 2.90, spec 13969.4): **" ret" word-start token** — Fires on the token " ret" as the start of "retention", mostly in chromatography "retention times" contexts. Predicts word completions like "ail", "ina", "irement", though the logits are mixed.
- 3.resid.34071 (share 3.0%, α 2.37, spec 467.5): **" ret" word-start token** — Fires on the split token " ret" at the start of a word, mostly "retention" (also "retail"). It predicts the continuation "ention" and similar suffixes.
- 3.resid.33878 (share 2.1%, α 2.86, spec 47.2): **ret/curr word-fragment tokens** — Fires on word-start fragments " ret" (in "online retail/retailers") and " curr" (in "currencies/currents"), i.e. a split-word first piece. Logits are mixed, so the next-token prediction is unclear.

## 7.resid.31916 (L5-7, resid) — '4' in 14xx years
PASS; 770 latents; Z 0.82 A 0.97 C 0.95; necessity 1.00; induce 0.97; 47 hubs dropped; trigger one_token, condition construction
> Fires on the digit '4' as the third digit of a year in the 1400s, such as 1443, 1449 or 1475. It stays silent on '4' in other year positions or other digit contexts.

- 5.resid.19437 (share 1.9%, α 1.20, spec 10.5): **Digit 4 in year (14xx)** — Fires on the token '4' as the second digit of a four-digit year in the 1400s (e.g., "1[[4]]43"). Only one example is shown, and the logit effects look unclear.
- 6.resid.34903 (share 1.8%, α 0.92, spec 390.9): **Digit 4 in 14xx years** — Fires on the digit "4" as the second digit of a 15th-century-style year (e.g., 1[4]43, 1[4]75) in historical dates. Logit effects are unclear and mixed.
- 6.resid.38060 (share 1.7%, α 1.59, spec 3.0): **Digit 2 inside numbers** — Fires on the digit "2" when it is the second digit inside a multi-digit number (e.g. 12, 1[2]00, 1[2]25) in numeric lists or codes. Logits are mixed and unclear.

## 8.attn.7985 (L8-11, attn) — Academic science text, mixed function-word positions
fail, near-threshold; 2226 latents; Z 0.61 A 1.05 C 1.03; necessity 1.00; induce 0.00; 54 hubs dropped; trigger varied, condition unclear
> Fires on varied tokens (' patterns', ' the', ' to', verbs like 'understand') in academic descriptions of scientific fields (geography, epidemiology, climate, energy). It is silent on ' patterns' in some climate contexts, so the evidence is mixed and the condition is unclear.

- 7.attn.37879 (share 1.1%, α 1.78, spec 1.2): **Predicts patterns/behavior continuation** — Fires on connective or verb tokens (' and', ' that', ' understand') after words like habits, patterns, or behavior, in analytics or science text. It predicts 'patterns', 'behavior' and 'predict' next.
- 6.attn.27112 (share 1.0%, α 2.64, spec 1.0): **List continuation in course descriptions (unclear)** — Fires on commas, "and", and word endings like "ics" inside lists of topics or activities in academic/course-style descriptions. Mixed peaks and weak logits, so unclear.
- 5.attn.33630 (share 0.7%, α 1.25, spec 1.0): **Mid-word subword tokens in all-caps text** — Fires on fragment tokens (VE, ED, H, U) inside words in ALL-CAPS text, often near document start. Peaks are mixed and logits are unclear, so the label is tentative.

## 9.mlp.11349 (L8-11, mlp) — ' sequ' word-start piece
PASS; 636 latents; Z 0.92 A 1.01 C 0.99; necessity 1.00; induce 0.99; 30 hubs dropped; trigger one_token, condition within_word
> Fires on the word-start piece ' sequ' (sequential, sequencing, sequenced), predicting continuations like 'ential', 'enced', 'encing'. Members are mostly ' sequ' detectors, plus a few other word-start pieces.

- 7.resid.15786 (share 2.4%, α 1.69, spec 219.6): **" sequ" word-start fragment** — Fires on the word-fragment token " sequ" (start of "sequential", "sequencing"), and predicts completions like "ential", "enced", "entially".
- 8.resid.34415 (share 2.4%, α 2.10, spec 70.7): **sequ token fragment (sequencing)** — Fires only on the word-fragment " sequ", mostly in "sequencing" contexts (genomics). It predicts suffix completions like "ential", "entially", "enced", "encing", "enc".
- 7.resid.33390 (share 2.3%, α 2.90, spec 94.8): **' Tra' token (Traffic), unclear** — Fires on the word-start token ' Tra' (as in "Traffic") at the start of a document or sentence. The logits push suffixes like "-entially" and "-ential", which don't fit "Traffic", so the evidence is thin and unclear.

## 10.resid.20721 (L8-11, resid) — ' random' / ' randomly' token
PASS; 1238 latents; Z 0.87 A 0.99 C 1.06; necessity 1.00; induce 0.91; 46 hubs dropped; trigger one_token, condition none
> Fires on the word "random" or "randomly" (also randomization pieces), mostly in statistics and sampling text such as random sampling, random assignment, random forest. It pushes toward words like sampling, generator, assignment.

- 7.resid.21949 (share 1.9%, α 1.27, spec 93.6): **" random" in randomization techniques** — Fires only on the token " random" in statistical or ML contexts, such as random subspaces, block randomization and stratified random sampling. It often comes before a suffix or noun like -ness, generator, Forest or injection.
- 7.resid.39583 (share 1.7%, α 1.09, spec 68.2): **"random" before randomization suffixes** — Fires only on the token " random" in statistics/ML/experimental-design contexts. It predicts word-completion suffixes like "ized", "ization", "ising" and "sampling", and suppresses unrelated tokens.
- 8.resid.32236 (share 1.5%, α 0.89, spec 58.4): **' random' token (statistics)** — Fires only on the token " random" in statistics and experimental-design contexts (random sampling, randomization). It predicts suffixes such as "ization", "ized", "walk" and "isation".
