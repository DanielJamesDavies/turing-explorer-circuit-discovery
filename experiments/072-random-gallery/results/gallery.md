# Random gallery (seed 20261012)

## 3.attn.9767 (L0-4, attn) — Token following "Capital" in finance terms
PASS; 257 latents; Z 0.97 A 1.10 C 1.16; necessity 1.00; induce 0.58; 71 hubs dropped
> Fires on continuation tokens right after the word "Capital" (e.g. " As" in "Capital Asset", " Bud", " and"), mostly in title-case finance headings. Silent on " As" when it is not preceded by "Capital".

- 1.resid.18903 (share 4.6%, α 1.39, spec 1.7): **Word-start fragments in Title Case (unclear)** — Fires on capitalized word-initial tokens (Success, Under, Know, Tem) in Title-Cased text, often where a longer word continues; the logit effect is mixed, so it is unclear.
- 0.resid.4768 (share 3.3%, α 1.24, spec 1.4): **Title-case word-start fragment (unclear)** — Fires on capitalized word-initial fragments (Fre, Ge, Th, Es) in Title Case headline-like text, where a word is split into pieces; logits are mixed, so unclear.
- 0.mlp.4768 (share 3.1%, α 1.14, spec 1.5): **Title-case word-start fragments (unclear)** — Fires on capitalized word-start fragments (Fre, Ge, Th, Es) beginning a split word in Title Case headline-like text; peaks vary and logits are weak, so unclear.

## 0.mlp.12557 (L0-4, mlp) — Control word detector
fail; 26 latents; Z 0.88 A 0.95 C 0.99; necessity 0.72; induce 0.39; 5 hubs dropped
> Fires on the word "control" (Control, control, controls), strongest in "Traffic Control Systems" and as a noun or verb in technical or management text. Members are mostly token detectors for control variants.

- 0.attn.16913 (share 8.5%, α 1.01, spec 105.1): **Token " control"** — Fires only on the token " control" (e.g., "control strategies", "control procedures"), in technical/engineering contexts. Logits mildly boost control-related continuations like "Control", "over", "acy"; evidence is thin.
- 0.attn.5256 (share 7.5%, α 1.12, spec 695.4): **Token " Control"** — Fires only on the token " Control" (e.g., "Control Systems", "Control materials"), in technical/engineering text. Logits mildly boost " Control"/" control" variants; evidence is thin.
- 0.attn.19725 (share 4.4%, α 0.82, spec 3.5): **Title-case word starts (unclear)** — Fires on scattered capitalized word-start tokens (Str, Tem, Other, Our) in title-case text; peaks are mixed, and logits favor word-fragment continuations like Tw, pol, Kos. Unclear.

## 0.resid.27712 (L0-4, resid) — Cell division terms (chrom, pol, II)
fail; 102 latents; Z 0.86 A 0.66 C 0.81; necessity 1.00; induce 1.24; 5 hubs dropped
> Fires on word pieces in cell-division (mitosis/meiosis) text: ' chrom' in chromatids/chromosomes, ' pol' in poles, ' II' after phase names. Predicts biology word endings. Silent in physics/abstract text.

- 0.mlp.205 (share 6.1%, α 0.69, spec 68.6): **' chrom' word-start token** — Fires only on the token " chrom" as a word fragment in chromatography, chromatid, chromosome, chromatin. Logits boost continuations like atin, omer, atch, ission.
- 0.mlp.4393 (share 5.8%, α 1.02, spec 2.2): **"fect" in "Perfect" tense headings** — Fires on the token "fect" in "Per|fect" within grammar headings like "Past/Future/Conditional Perfect Continuous". Logits are mixed and unclear, with weak boosts for '-' and ' ('.
- 0.mlp.21122 (share 5.0%, α 1.00, spec 1.5): **Arabic mid-word letters (unclear)** — Fires on mid-word Arabic letters (ة, ن, و) and occasionally Japanese ク or repeated commas; peaks are scattered, with no clear shared meaning and weak logits.

## 5.attn.10168 (L5-7, attn) — ' men' in "menstrual"
fail, near-threshold; 274 latents; Z 0.00 A 0.00 C 0.00; necessity 1.00; induce 0.00; 94 hubs dropped
> Fires on ' men' as the first piece of "menstrual" (and weakly on ' Race' in "Arms Race"), i.e. a word-initial fragment inside a longer word. Circuit is failed and diffuse, with only generic members.

- 4.resid.19412 (share 9.1%, α 1.38, spec 1.3): **Mid-word fragments, unclear** — Fires on varied mid-word or noun/adjective tokens (ative, pro, scene, balance, id, fier) in descriptive prose; peaks are inconsistent. Logits weakly favor ' of'. Unclear.
- 3.resid.12927 (share 5.8%, α 0.98, spec 1.2): **Word after "the" (unclear)** — Fires on mixed tokens, often word pieces or nouns/adjectives right after "the" (cout, lish, ree, risk, principal, expression); peaks are inconsistent. Weakly boosts " of" next.
- 3.resid.18667 (share 3.7%, α 1.45, spec 1.1): **Mixed tokens, unclear (further/later)** — Fires on mid-word or mixed tokens (' operation', 'aren', 'つ', ' IN') with no coherent pattern; weakly promotes adverbs like 'further', 'later', 'additional', 'moreover'.

## 5.mlp.15947 (L5-7, mlp) — "not limited to" negated limitation
PASS, amplifier; 543 latents; Z 0.99 A 1.19 C 1.19; necessity 1.00; induce 0.76; 89 hubs dropped
> Fires on ' limited' (also 'restricted', 'confined') right after a negation, as in "is not limited to", where a "not limited to..." construction is expected. Weaker for "but not limited to" and "mostly restricted to".

- 4.resid.21357 (share 3.9%, α 2.94, spec 160.9): **Token " limited"** — Fires solely on the token " limited" (adjective, as in "limited to", "limited allusions"). Logits weakly favor " access", " limited", " space", " edition", but the evidence is thin and mixed.
- 4.resid.25898 (share 3.5%, α 2.84, spec 17.9): **"not limited to" phrase** — Fires on " limited" (occasionally " restricted") right after "not" in "is not limited to…" constructions; logits show no clear next-token pattern.
- 4.resid.26922 (share 2.5%, α 1.82, spec 1.9): **Negated verb "do not contain/have"** — Fires on verbs or nouns (contain, have, elements) right after a negation ("does not contain", "no particular elements"); logits mildly favor nor/except/unless/instead, but evidence is sparse.

## 7.resid.29770 (L5-7, resid) — food token detector
PASS, amplifier; 721 latents; Z 0.96 A 0.87 C 1.27; necessity 1.00; induce 1.24; 103 hubs dropped
> Fires on the word "food"/"Food", strongest on repeated "Food and X" headings and on food in regulatory or sustainability phrases. Predicts follow-ons like security, service, handling. Members are mostly food-token detectors.

- 6.resid.32006 (share 2.3%, α 1.74, spec 10.6): **"red" in ingredient (food prep)** — Fires on the "red" subtoken inside "ingredient(s)" and once on " plate", in food and cooking contexts. It pushes up prep, prepared, presentation and allergen tokens. The evidence is thin, so this is somewhat unclear.
- 5.resid.3962 (share 1.7%, α 2.12, spec 9.4): **Grill/barbecue word fragments (food)** — Fires on mid-word subword pieces in grilling/barbecue words (gr-ill, gr-illed, barbec-ue) in food-description contexts; logits favor cooking/food continuations like cook, taste, restaurant.
- 5.resid.17590 (share 1.4%, α 1.53, spec 129.0): **Token " food" before noun** — Fires only on the token " food" in food safety/industry contexts. Logits favor following nouns like security, supply, trade, safety, service.

## 9.attn.38891 (L8-11, attn) — Connectors in lists of consequences
fail, amplifier; 613 latents; Z 0.88 A -- C 0.11; necessity 1.00; induce 1.21; 90 hubs dropped
> Fires on connectors and words (' and', ',', ' economic', 'ism', ' in') inside lists of negative consequences or effects (hardship, instability, unrest) in historical or social cause-and-effect text. Silent on ' and' joining ordinary nouns or topics.

- 8.resid.33386 (share 3.7%, α 1.05, spec 1.7): **Predicts negative-consequence words** — Fires on mixed tokens (" only", " can", ",", " communication") in expository text about problems, where a negative-outcome word (mis-, incorrect, errors, poor) comes next. Peaks are diverse; somewhat unclear.
- 7.resid.39521 (share 2.4%, α 0.92, spec 1.2): **Subject before past-tense verb** — Fires on the end of a subject noun phrase in historical or explanatory text about a concept, where a past-tense verb like "were developed", "became", or "was introduced" comes next. Peak tokens are mixed.
- 7.attn.15390 (share 2.1%, α 0.88, spec 1.1): **Humanism/Renaissance context function words** — Fires on function words (of, for, the, comma) in passages about humanism and the Renaissance period; logits favor historical verbs like began/became/were, suppress must/provides. Peaks are mixed, so unclear.

## 8.mlp.2453 (L8-11, mlp) — Structure nouns followed by that/can
fail, amplifier; 870 latents; Z 0.92 A 0.68 C 0.41; necessity 1.00; induce 1.46; 83 hubs dropped
> Fires on relative pronouns or modals/infinitive 'to' (that, can, could, which, to) right after a noun for built structures or buildings, in engineering text, where a verb about structural function (withstand, support, reach) comes next. Silent on 'that' in non-engineering clauses.

- 8.attn.4808 (share 3.0%, α 1.80, spec 3.7): **Engineering-context function words (unclear)** — Fires on mixed connectives (' and', ' to', ',') in engineering/structure-related text; promotes 'engineering/engine/engineer' as next token. Peaks are inconsistent, so somewhat unclear.
- 7.attn.27777 (share 2.9%, α 2.58, spec 2.6): **Composite materials engineering context (unclear)** — Fires on function words like " of", " in", " remarkable" in intros about composite/engineered materials; predicts materials-engineering words (mechanical, manufact, materials, designs). Mixed tokens, so somewhat unclear.
- 7.attn.18401 (share 2.8%, α 1.87, spec 1.5): **Plural subject, predicts plural verb** — Fires on commas, "these", and "),", typically right after a plural subject noun phrase, where a plural verb or pronoun comes next; boosts " are", " were", " they" and suppresses singular "its/his".

## 9.resid.10786 (L8-11, resid) — "global" as modifier
PASS, amplifier; 920 latents; Z 0.92 A 1.01 C 1.08; necessity 1.00; induce 1.30; 97 hubs dropped
> Fires on the word "global" (also "Global", "international", "-wide") used as a modifier, often before nouns like environmental, economy or issues. Predicts cooperation-type continuations. Silent on the contrast text, which has no such modifier.

- 8.resid.30012 (share 1.5%, α 1.38, spec 259.9): **Capitalized " Global" at title start** — Fires on the token " Global", seen at the start of a document or heading (e.g. "Global Transportation Systems"). Logits push suffixes like -ization/-ized and scale/citizens. Evidence is thin, with only 6 examples.
- 8.resid.26956 (share 1.2%, α 3.34, spec 2.1): **Growing awareness → environment** — Fires on "growing", "of", "the" in phrases like "growing concerns/awareness of the…" in environmental-movement text; logits boost "Environment"/"environment" next, so it predicts environmental topics.
- 7.resid.11194 (share 1.2%, α 4.05, spec 4.6): **Pre-"international/global" context words** — Fires on function words like " beyond" and " the" in globalization/international-relations text, where the next word is likely "international", "global", or "abroad"; evidence is thin and mixed.
