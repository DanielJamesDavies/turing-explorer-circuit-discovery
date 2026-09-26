# Node inspection: are some inhibitory nodes obviously inhibitors?

Roles from `roles_pilot.jsonl` (leave-one-out, share of a_pos). **L>0 excitatory, L<0 inhibitory.** `act@anchor` = mean activation at the target's anchor on its activating contexts; `max act / max ctr` = mean per-sequence max on activating / contrast contexts. Logit effect = decoder direction × final-norm gain × unembedding (first-order, not causal).


---

## Target 0.mlp.16000  (31 nodes; release L_C -0.060)

- **peak tokens:** 'tra'×26, 'obser'×19, 'exer'×6, 'searching'×5 (consistency 41%)
- activating: …, speaking exercises (speaking [[exer]]
- activating: …, speaking exercises (speaking [[exer]]
- activating: …, speaking exercises (speaking [[exer]]
- contrast: …value of an art collection by ensuring its appropriate use.<s> Digitalization has brought about a significant shift in the way
- contrast: …ulf to Twitter - as it changes and evolves in such an astonishing manner that it would make even amateurish speak
- contrast: …es where policymakers need to make their own decisions without damaging individual interests.<s> A Pareto Im
- target logit effect: + ['pping', 'ppers', 'inction', 'oring', 'oration', 'ovy'] | − ['the', 'mund', 'side', 'man', 'Med', 'inger']


### inhibitory (most negative L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.attn.15350 | 1.63 | -0.069 | -0.041 | -0.044 | -2.778 | inh | 0.11 | 0.64 / 0.58 | '.'×64 (100%) | …....... [[.]] | 'dist', 'rod', 'cart', 'division' / 'operators', 'min', 'HL', 'Command' |
| 0.attn.6546 | 1.03 | +0.016 | -0.023 | -0.016 | +2.413 | exc | 0.18 | 0.64 / 0.64 | 'symbols'×14; 'probability'×10; 'events'×6 (22%) | … [[symbols]] | 'Sold', 'arta', 'docs', 'normal' / 'em', 'scientific', 'per', 'L' |
| 0.attn.10377 | 1.33 | -0.011 | -0.008 | -0.010 | -0.758 | inh | 0.08 | 0.32 / 0.30 | 'efficiency'×17; 'efficient'×15; 'optim'×11 (27%) | …speed with write efficiency to maximize system [[efficiency]] | 'Jazz', 'allocate', 'Tower', 'optimized' / 'label', 'rating', 'par', 'labels' |

### excitatory (most positive L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.attn.5663 | 0.82 | -0.434 | +0.194 | +0.148 | -39.354 | inh | 3.41 | 4.20 / 4.07 | 'symbols'×19; 'symbol'×10; 'graph'×4 (30%) | … [[symbols]] | 'pass', 'int', ',', 'fl' / 'master', 'ffen', 'service', 'Scala' |
| 0.attn.25728 | 2.48 | +0.089 | +0.081 | +0.093 | +0.156 | exc | 0.07 | 0.10 / 0.00 | 'pros'×64 (100%) | … [[pros]] | 'pros', 'small', 'ac', 'ros' / 'Mul', 'gg', 'cano', 'entertain' |
| 0.attn.23318 | 0.95 | +0.049 | +0.049 | +0.050 | +0.764 | exc | 0.06 | 0.07 / 0.00 | 'exer'×64 (100%) | …tests include listening [[exer]] | 'Fre', 'Uncle', 'pond', 'icks' / 'ton', 'co', 'fix', 'acc' |
| 0.attn.38658 | 0.99 | +0.048 | +0.040 | +0.048 | +1.057 | exc | 0.07 | 0.11 / 0.00 | 'searching'×64 (100%) | … [[searching]] | 'Co', 'ear', 'Ros', 'echo' / 'cycles', 'volumes', 'attach', 'severe' |
| 0.attn.40291 | 0.88 | +0.037 | +0.025 | +0.036 | +0.439 | exc | 0.08 | 0.11 / 0.04 | 'exer'×54; 'cis'×10 (84%) | …speaking exercises), reading exer [[cis]] | 'gress', 'dou', 'vers', 'Graf' / 'flows', 'transport', 'annel', 'cloth' |
| 0.attn.24589 | 1.16 | +0.030 | +0.023 | +0.030 | +0.581 | exc | 0.04 | 0.11 / 0.00 | 'searches'×64 (100%) | … [[searches]] | 'searches', 'rot', 'SQL', 'Di' / 'OR', 'pi', 'pit', 'pid' |

### G and L_C disagree

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.attn.712 | 0.88 | +0.004 | +0.003 | +0.012 | -4.892 | inh | 1.20 | 2.56 / 2.62 | 'like'×12; '1'×10; '('×9 (19%) | … [[1]] | 'aw', 'n', 'ch', 'credit' / 'onnées', '%.', '².', '\\).' |
| 0.attn.6125 | 2.08 | -0.009 | -0.003 | +0.004 | -1.236 | inh | 0.12 | 0.39 / 0.44 | 'was'×34; 'began'×9; 'had'×8 (53%) | … [[was]] | 'Russian', 'Ford', 'prote', 'timer' / 'sun', 'akh', 'occup', 'oku' |
| 0.attn.36957 | 1.03 | +0.000 | -0.001 | +0.000 | -0.061 | inh | 0.01 | 0.01 / 0.00 | 'ly'×17; 'transformations'×13; 'maps'×12 (27%) | …of the central aspects of linear [[maps]] | 'Ant', 'Stop', 'ann', 'pin' / 'heid', 'ople', 'ibles', 'Martin' |
| 0.attn.32818 | 1.01 | +0.000 | -0.001 | +0.000 | -0.152 | inh | 0.01 | 0.04 / 0.00 | 'ob'×64 (100%) | …actual [[ob]] | 'Dem', 'ob', 'automat', 'Gray' / 'Mississippi', 'thy', 'Night', 'shorter' |

---

## Target 1.mlp.8495  (123 nodes; release L_C +0.082)

- **peak tokens:** 'multiple'×60, 'several'×2, 'Several'×2 (consistency 94%)
- activating: …'s shove various multiple [[multiple]]
- activating: …<s> Utilizing [[several]]
- activating: …, particularly significant when finding the least common [[multiple]]
- contrast: …adjustments must be made to account for differences in growth rates caused by industry conditions and type of businesses.<s> Pre
- contrast: …options assets, regardless of whether downside risk is present or not.<s> Peer group analysis and market multiples are
- contrast: …complex statements, other key ratios are used: liquidity (which measures the strength and complexity of existing financial statements
- target logit effect: + ['ova', 'ort', 'infant', 'Taylor', 'colon', 'reward'] | − ['body', 'made', 'med', 'lets', 'currently', 'log']


### inhibitory (most negative L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.resid.6299 | 1.15 | +0.006 | -0.026 | -0.038 | -6.386 | inh | 5.09 | 7.55 / 8.97 | 'inter'×10; 'new'×6; 'strateg'×4 (16%) | …research must incorporate [[inter]] | 'benef', 'vital', 'capac', 'relevant' / 'ini', 'old', 'Lad', 'Sc' |
| 0.attn.712 | 1.01 | +0.548 | -0.024 | -0.030 | -3.128 | inh | 1.66 | 2.44 / 2.56 | 'like'×12; '1'×10; '('×9 (19%) | … [[1]] | 'aw', 'n', 'ch', 'credit' / 'onnées', '%.', '².', '\\).' |
| 0.mlp.26380 | 1.53 | +0.178 | +0.007 | -0.029 | -1.233 | inh | 2.98 | 4.54 / 4.10 | 'adjacent'×6; 'color'×5; 'problem'×5 (9%) | …that no two adjacent vertices are the same [[color]] | 'probability', 'efficiently', 'necessary', 'feas' / 'Ha', 'Inst', 'sh', 'Pe' |
| 0.mlp.39852 | 0.99 | -0.071 | -0.021 | -0.022 | -1.439 | inh | 1.33 | 2.36 / 2.08 | 'Unit'×13; 'JavaScript'×6; 'Cloud'×6 (20%) | …interactions, while Java and C++ provide [[static]] | 'software', 'developers', 'IBM', 'functionality' / 'Fer', 'sympathy', 'ights', 'Wal' |
| 0.mlp.20249 | 1.00 | -0.011 | -0.019 | -0.017 | +0.601 | exc | 0.00 | 6.05 / 7.33 | 'uz'×25; 'Jay'×20; 'Matt'×6 (39%) | … [[selection]] | 'foot', 'reflect', 'wn', 'ut' / 'secre', 'diplom', 'Sus', 'private' |
| 0.attn.15350 | 1.60 | -0.029 | +0.008 | -0.016 | -1.525 | inh | 0.09 | 0.87 / 0.57 | '.'×64 (100%) | …....... [[.]] | 'dist', 'rod', 'cart', 'division' / 'operators', 'min', 'HL', 'Command' |

### excitatory (most positive L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.attn.18179 | 2.54 | +0.062 | +0.196 | +0.223 | -1.434 | exc | 0.01 | 3.99 / 3.98 | 'Cov'×17; 'spons'×6; 'kern'×5 (27%) | … [[tutorial]] | 'Point', 'snapshot', 'GP', 'DNS' / 'mar', 'S', 'accord', 'slow' |
| 0.resid.39987 | 1.19 | +0.369 | +0.167 | +0.166 | +7.177 | exc | 3.10 | 3.33 / 0.30 | 'multiple'×64 (100%) | …have been taken into account for these multiple [[multiple]] | 'gener', 'neighbor', 'sin', 'immediate' / 'ants', 'Argument', 'avor', 'ent' |
| 0.mlp.20000 | 0.84 | +0.184 | +0.152 | +0.157 | +7.526 | exc | 2.74 | 3.06 / 0.25 | 'multiple'×64 (100%) | …availability by introducing redundant data across [[multiple]] | 'spatial', 'arbitr', 'scientific', 'contin' / 'of', 's', 'and', '&' |
| 0.resid.32179 | 1.05 | +0.328 | +0.165 | +0.147 | +2.469 | exc | 4.97 | 6.05 / 4.51 | 'range'×13; 'types'×8; 'categories'×7 (20%) | …listic options available allows for a wide [[variety]] | 'types', 'categories', 'ranges', 'subsets' / 'integrity', 'Arm', 'Bre', 'assured' |
| 0.attn.6830 | 1.03 | +0.097 | +0.086 | +0.134 | +4.611 | exc | 0.88 | 0.96 / 0.05 | 'multiple'×64 (100%) | …'s shove various multiple [[multiple]] | 'multiple', 'fib', 'mes', 'Marcus' / 'village', 'rian', 'biz', 'villages' |
| 0.mlp.32179 | 1.19 | +0.362 | +0.093 | +0.120 | +1.936 | exc | 4.50 | 5.65 / 4.02 | 'idae'×13; 'range'×12; 'main'×12 (20%) | …Indo-European with its diverse [[range]] | 'types', 'ranges', 'categories', 'range' / 'Fail', 'insert', 'mach', 'aus' |

### G and L_C disagree

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.attn.5663 | 0.73 | +0.671 | +0.095 | +0.090 | -20.847 | inh | 3.87 | 4.31 / 4.19 | 'symbols'×19; 'symbol'×10; 'graph'×4 (30%) | … [[symbols]] | 'pass', 'int', ',', 'fl' / 'master', 'ffen', 'service', 'Scala' |
| 0.attn.27971 | 0.96 | -0.011 | +0.018 | +0.038 | -0.297 | inh | 0.06 | 0.31 / 1.03 | 'financial'×19; 'invest'×15; 'vest'×8 (30%) | … [[financial]] | 'ollar', 'funds', 'crash', 'prot' / 'profil', 'Ly', 'erg', 'Wi' |
| 0.mlp.1349 | 1.53 | +0.714 | +0.056 | +0.038 | +1.118 | inh | 5.26 | 9.93 / 7.37 | 'Bi'×10; 'Marie'×6; 'Computer'×3 (16%) | …<s> [[Civil]] | 'repository', 'vital', 'necessity', 'mighty' / 'or', 'with', 'Ras', 'others' |
| 0.resid.12868 | 1.00 | +0.002 | +0.007 | +0.035 | +0.018 | inh | 0.08 | 1.96 / 4.18 | 'economy'×13; 'economic'×12; 'aggregate'×7 (20%) | …policies, such as increased government expend [[iture]] | 'policy', 'economic', 'conom', 'dollars' / 'Martin', 'person', 'oo', 'God' |

---

## Target 3.attn.35598  (201 nodes; release L_C +0.006)

- **peak tokens:** "'"×26, 'an'×17, 'Shakespeare'×8, 'Elizabeth'×4 (consistency 41%)
- activating: …subtlety of Shakespeare's [[Elizabeth]]
- activating: …subtlety of Shakespeare's [[Elizabeth]]
- activating: …with the human condition. veritable Shakespeare [[an]]
- contrast: …value of an art collection by ensuring its appropriate use.<s> Digitalization has brought about a significant shift in the way
- contrast: …ulf to Twitter - as it changes and evolves in such an astonishing manner that it would make even amateurish speak
- contrast: …es where policymakers need to make their own decisions without damaging individual interests.<s> A Pareto Im
- target logit effect: + ['Ald', 'mac', 'profile', 'az', 'rov', 'acter'] | − ['abs', 'po', 'uncertain', 'sentences', 'try', 'previous']


### inhibitory (most negative L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.mlp.5646 | 1.17 | +0.095 | -0.082 | -0.078 | -4.453 | inh | 2.44 | 4.71 / 4.03 | 'Lock'×20; 'temperature'×7; 'Bol'×6 (31%) | …ers, [[scient]] | '&', 'lean', 'Wright', 'ower' / 'site', 'roy', 'non', 'large' |
| 0.attn.712 | 1.16 | +0.841 | -0.020 | -0.055 | -12.561 | inh | 1.98 | 2.79 / 2.62 | 'like'×12; '1'×10; '('×9 (19%) | … [[1]] | 'aw', 'n', 'ch', 'credit' / 'onnées', '%.', '².', '\\).' |
| 0.mlp.1982 | 1.13 | -0.012 | -0.049 | -0.053 | -2.489 | inh | 0.00 | 8.49 / 9.11 | 'ampa'×11; 'Germania'×6; 'iglia'×6 (17%) | … [[�]] | 'story', 'DE', 'AN', 'Tra' / 'isted', 'Arist', 'steam', 'Duke' |
| 1.attn.34495 | 0.68 | -0.030 | -0.021 | -0.048 | -2.996 | inh | 2.18 | 7.40 / 7.39 | 'is'×5; 'imm'×5; 'interaction'×4 (8%) | … [[scale]] | 'cks', 'lywood', 'ished', 'ipping' / 'sed', 'plan', 'Cart', 'others' |
| 0.mlp.1349 | 0.95 | +0.083 | -0.061 | -0.039 | -4.270 | inh | 2.64 | 8.72 / 9.18 | 'Bi'×10; 'Marie'×6; 'Computer'×3 (16%) | …<s> [[Civil]] | 'repository', 'vital', 'necessity', 'mighty' / 'or', 'with', 'Ras', 'others' |
| 0.resid.1982 | 1.08 | +0.033 | -0.038 | -0.038 | -1.457 | inh | 0.00 | 8.39 / 9.02 | 'ampa'×11; 'Germania'×6; '________'×6 (17%) | … [[Germania]] | 'story', 'DE', 'Ac', 'AN' / 'isted', 'Arist', 'po', 'Po' |

### excitatory (most positive L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.resid.10434 | 5.69 | +0.018 | +0.014 | +0.385 | +12.796 | exc | 0.00 | 2.41 / 2.75 | 'es'×2; 'ium'×2; 'effic'×2 (3%) | …café [[estava]] | 'long', 'halb', 'ared', 'bridge' / 'types', 'Cent', "'", 'prov' |
| 0.mlp.13164 | 1.24 | +0.127 | +0.150 | +0.380 | +14.670 | exc | 1.17 | 6.85 / 0.10 | 'Shakespeare'×64 (100%) | …A Bard's Palette: [[Shakespeare]] | 'Hot', 'Memorial', 'yer', 'ika' / 'pe', 'of', 'g', 'ang' |
| 0.attn.20152 | 1.16 | +0.291 | +0.107 | +0.348 | +10.264 | exc | 1.00 | 1.72 / 0.01 | 'Shakespeare'×64 (100%) | … [[Shakespeare]] | 'Shakespeare', 'ze', 'Luther', 'otti' / 'substr', 'parad', 'exterior', 'land' |
| 2.mlp.30344 | 4.53 | +0.000 | -0.000 | +0.262 | +10.412 | exc | 0.00 | 1.58 / 1.70 | 'LO'×6; 'TR'×6; 'IN'×4 (9%) | …ECTS AND MODIFICATIONS [[IN]] | 'Mul', 'hy', 'oby', 'berry' / 'types', 'structure', 'processing', 'composition' |
| 0.resid.13164 | 1.39 | +0.410 | +0.178 | +0.220 | +7.609 | exc | 4.45 | 7.23 / 0.08 | 'Shakespeare'×64 (100%) | … [[Shakespeare]] | 'Shakespeare', 'key', 'sett', 'Sen' / 'Gaussian', 'rim', 'ativity', 'while' |
| 1.resid.3776 | 1.50 | +0.219 | +0.189 | +0.169 | +5.206 | exc | 5.13 | 8.02 / 0.12 | 'Shakespeare'×64 (100%) | … [[Shakespeare]] | 'Shakespeare', 'Tod', 'sett', 'chron' / 'blocks', 'Eastern', 'regime', 'ischen' |

### G and L_C disagree

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.mlp.406 | 1.04 | +0.091 | +0.073 | +0.093 | +1.404 | inh | 0.00 | 9.55 / 9.72 | 'in'×17; 'the'×4; 'dah'×4 (27%) | … [[coefficients]] | 'ex', 'gel', 'Schmidt', 'posed' / 'Asian', 'verte', 'shell', 'SC' |
| 0.attn.18179 | 1.03 | +0.069 | -0.226 | +0.047 | -2.394 | inh | 0.04 | 4.09 / 3.99 | 'Cov'×17; 'spons'×6; 'kern'×5 (27%) | … [[tutorial]] | 'Point', 'snapshot', 'GP', 'DNS' / 'mar', 'S', 'accord', 'slow' |
| 2.mlp.14460 | 0.97 | -0.066 | -0.038 | +0.040 | -0.016 | inh | 1.77 | 3.78 / 3.85 | 'ens'×11; 'neur'×7; 'R'×7 (17%) | …. involves antigenic stage stages where [[foreign]] | 'types', 'structure', 'composition', 'processing' / 'Mul', 'hy', 'oby', 'next' |
| 1.resid.39697 | 1.18 | +0.091 | +0.018 | +0.039 | -0.589 | inh | 1.97 | 6.30 / 6.77 | 's'×11; 'male'×6; 'lad'×4 (17%) | …� [[�]] | 'reserve', 'aw', 'pr', 'rid' / 'NP', 'Ras', 'agi', 'GP' |

---

## Target 4.mlp.20758  (304 nodes; release L_C -0.252)

- **peak tokens:** 'modern'×52, 'Modern'×12 (consistency 81%)
- activating: …<s> [[Modern]]
- activating: …<s> [[Modern]]
- activating: …<s> Our [[modern]]
- contrast: …value of an art collection by ensuring its appropriate use.<s> Digitalization has brought about a significant shift in the way
- contrast: …ulf to Twitter - as it changes and evolves in such an astonishing manner that it would make even amateurish speak
- contrast: …es where policymakers need to make their own decisions without damaging individual interests.<s> A Pareto Im
- target logit effect: + ['ization', 'izable', 'rev', 'smaller', 'simpler', 'izing'] | − ['crew', 'inn', 'inen', 'actions', 'hands', 'Mo']


### inhibitory (most negative L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.mlp.39049 | 1.03 | -0.002 | +0.047 | -0.056 | +0.150 | inh | 0.00 | 4.65 / 10.02 | 'ouc'×6; 'campion'×6; 'buf'×4 (9%) | … [[rand]] | 'cast', 'thy', 'Pl', 'c' / 'owe', 'Pap', 'Om', 'margin' |
| 0.mlp.21893 | 1.02 | +0.037 | -0.022 | -0.049 | +0.084 | exc | 0.00 | 5.11 / 9.88 | 'ing'×20; 'etten'×6; 'egin'×4 (31%) | … [[où]] | 'iders', 'ears', 'ng', 'rs' / 'shell', 'bomb', 'block', 'shell' |
| 0.mlp.15578 | 0.98 | -0.048 | -0.003 | -0.048 | -0.024 | inh | 0.00 | 4.21 / 8.18 | 'vare'×34; 'pc'×6; 'Parti'×4 (53%) | … [[std]] | 'NER', 'angen', 'ely', 'person' / 'Julia', 'Water', 'Aur', 'Chinese' |
| 2.mlp.28752 | 1.39 | +0.116 | -0.036 | -0.048 | -1.297 | inh | 2.00 | 2.94 / 3.25 | 'devices'×4; 'songs'×2; 'used'×2 (6%) | …the importance of compression in shaping popular [[songs]] | 'ch', 'core', 'eth', 'di' / 'ame', 'halb', 'ared', 'ess' |
| 0.mlp.3223 | 0.78 | +0.019 | +0.004 | -0.037 | -0.543 | inh | 0.00 | 4.58 / 11.12 | 'ampa'×11; 'campion'×6; 'buf'×4 (17%) | … [[std]] | 'pip', 'omb', 'it', 'inu' / 'extrem', 'PR', 'micro', 'ory' |
| 0.resid.16354 | 1.01 | -0.044 | -0.015 | -0.030 | +0.560 | inh | 0.00 | 6.29 / 9.67 | 'Kü'×10; 'Bild'×8; 'gesamt'×6 (16%) | … [[Bild]] | '-', 'os', 'on', '–' / 'tel', 'nob', 'maximum', 'ggplot' |

### excitatory (most positive L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.attn.18179 | 2.51 | -0.075 | +0.361 | +0.632 | -1.045 | inh | 0.01 | 4.04 / 3.99 | 'Cov'×17; 'spons'×6; 'kern'×5 (27%) | … [[tutorial]] | 'Point', 'snapshot', 'GP', 'DNS' / 'mar', 'S', 'accord', 'slow' |
| 0.mlp.9606 | 1.83 | -0.008 | +0.165 | +0.411 | +4.836 | exc | 2.34 | 2.55 / 0.16 | 'modern'×38; 'contemporary'×24; 'Modern'×2 (59%) | … [[modern]] | 'day', 'society', 'soci', 'America' / 'Transfer', 'Prop', 'Prem', 'Policy' |
| 0.attn.26395 | 1.19 | +0.048 | +0.141 | +0.405 | +4.393 | exc | 0.84 | 0.88 / 0.02 | 'modern'×57; 'contemporary'×7 (89%) | … [[modern]] | 'modern', 'contemporary', 'Jesus', 'aci' / '&', 'flow', 'horizon', 'acc' |
| 0.resid.9606 | 1.65 | +0.041 | +0.184 | +0.321 | +4.409 | exc | 2.58 | 2.80 / 0.16 | 'modern'×42; 'contemporary'×20; 'Modern'×2 (66%) | … [[modern]] | 'day', 'modern', 'contemporary', 'soci' / 'Transfer', 'winner', 'rap', 'Turner' |
| 0.mlp.21996 | 1.00 | -0.077 | +0.128 | +0.250 | +3.503 | exc | 2.97 | 3.31 / 0.14 | 'Modern'×39; 'modern'×25 (61%) | …1700, the Early [[Modern]] | 'business', 'understanding', 'American', 'amer' / 'keep', 'ke', 'well', 'arts' |
| 3.resid.5335 | 1.47 | +0.410 | +0.217 | +0.188 | +6.467 | exc | 8.82 | 9.08 / 0.27 | 'contemporary'×27; 'modern'×25; 'Modern'×12 (42%) | …creativity and revolutionary movements.<s> [[Modern]] | 'approaches', 'uses', 'forms', 'HO' / 'uelle', 'ôle', 'ount', 'iale' |

### G and L_C disagree

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.mlp.40345 | 1.19 | -0.050 | +0.024 | +0.111 | +1.208 | inh | 0.00 | 6.10 / 10.33 | 'to'×26; 'arante'×10; 'ing'×7 (41%) | … [[to]] | 'concurrent', 'system', 'uti', 'sort' / 'Head', 'PR', 'Current', 'outer' |
| 0.attn.2835 | 1.30 | -0.011 | -0.066 | +0.106 | +0.887 | inh | 0.02 | 1.63 / 1.66 | 'kern'×19; 'LOCK'×13; 'Sta'×9 (30%) | … [[headers]] | 'inson', 'ham', 'ap', 'Cass' / 'motion', 'non', 'gen', 'total' |
| 0.mlp.37034 | 0.84 | -0.028 | -0.012 | +0.086 | +0.902 | inh | 1.54 | 2.18 / 0.22 | 'current'×43; 'future'×21 (67%) | …ive succession plan includes understanding current and [[future]] | 'batch', 'extent', 'degrees', 'kinds' / 'es', 'and', 'ants', 'cules' |
| 1.resid.431 | 1.29 | -0.056 | +0.075 | +0.086 | +0.458 | inh | 0.00 | 8.00 / 11.63 | ','×44; 'that'×10; 'doesn'×3 (69%) | … [[,]] | 'wn', 'ims', 'pet', 'iders' / 'bio', 'delta', 'cort', 'training' |

---

## Target 6.attn.8246  (382 nodes; release L_C +0.101)

- **peak tokens:** ','×32, 'and'×27, '"),'×2, '),'×2 (consistency 50%)
- activating: …s primarily categorize them into coordinating [[,]]
- activating: …s primarily divides them into coordinating [[,]]
- activating: …s primarily divides them into coordinating [[,]]
- contrast: …clauses to an independent one. diverse functionality of conjunctions empowers writers and speakers to express complex relationships
- contrast: …and 'or' unite equally-structured phrases or clauses, while subordinating conjunctions such
- contrast: …the two.ordishing conjunction (int) clause(s), which provide dependent (subordinate)clauses that
- target logit effect: + ['equ', 'aggregate', 'Co', 'Hoff', 'ouse', 'ip'] | − ['white', 'find', 'prem', 'principles', 'products', 'ed']


### inhibitory (most negative L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.attn.5663 | 1.21 | +0.897 | -0.016 | -0.020 | -7.292 | inh | 3.76 | 4.19 / 4.07 | 'symbols'×19; 'symbol'×10; 'graph'×4 (30%) | … [[symbols]] | 'pass', 'int', ',', 'fl' / 'master', 'ffen', 'service', 'Scala' |
| 1.attn.34495 | 0.99 | +0.079 | -0.018 | -0.015 | -1.007 | inh | 1.73 | 7.50 / 7.32 | 'is'×5; 'imm'×5; 'interaction'×4 (8%) | … [[scale]] | 'cks', 'lywood', 'ished', 'ipping' / 'sed', 'plan', 'Cart', 'others' |
| 0.mlp.27056 | 1.33 | +0.036 | -0.011 | -0.009 | -0.359 | exc | 0.00 | 10.27 / 7.25 | 'with'×25; 'the'×17; 'ing'×8 (39%) | … [[ing]] | 'controllers', 'Brig', 'тел', 'ograph' / 'gray', 's', 'TB', 'Paulo' |
| 0.mlp.21893 | 1.29 | +0.029 | +0.001 | -0.007 | +0.450 | inh | 0.00 | 10.86 / 8.26 | 'ing'×20; 'etten'×6; 'egin'×4 (31%) | … [[où]] | 'iders', 'ears', 'ng', 'rs' / 'shell', 'bomb', 'block', 'shell' |
| 0.mlp.9162 | 0.98 | +0.023 | -0.002 | -0.007 | +0.008 | inh | 0.00 | 11.28 / 8.42 | 'ing'×51; 'of'×8; 'the'×2 (80%) | … [[ing]] | 'imb', 'rett', 'ap', '-' / 'photograph', 'littérature', 'channel', 'paper' |
| 0.resid.9162 | 0.91 | +0.004 | +0.003 | -0.006 | -0.116 | inh | 0.00 | 11.47 / 8.49 | 'ing'×55; 'the'×5; 'of'×2 (86%) | … [[ing]] | 'imb', 'Alexand', 'ap', 'alk' / 'photograph', 'Record', 'littérature', 'channel' |

### excitatory (most positive L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.resid.39300 | 4.09 | +0.376 | +0.223 | +0.242 | +4.040 | exc | 2.68 | 12.29 / 11.52 | 'unction'×33; 'ating'×31 (52%) | …ative, each with distinct roles.ordin [[ating]] | 'conj', 'sentences', 'attach', 'declarations' / 'ving', 'ogen', 'green', 'Green' |
| 1.resid.38996 | 2.06 | +0.050 | +0.051 | +0.104 | +1.897 | inh | 1.05 | 4.13 / 3.70 | 'conj'×31; 'unction'×29; 'ordin'×4 (48%) | …ating conjunctions, subordinating [[conj]] | 'conne', 'clause', 'cla', 'unction' / 'Sam', 'Gal', 'pat', 'customers' |
| 0.attn.18179 | 2.99 | +0.176 | +0.042 | +0.093 | +0.928 | exc | 0.01 | 3.90 / 4.03 | 'Cov'×17; 'spons'×6; 'kern'×5 (27%) | … [[tutorial]] | 'Point', 'snapshot', 'GP', 'DNS' / 'mar', 'S', 'accord', 'slow' |
| 2.resid.7752 | 3.17 | +0.012 | +0.055 | +0.090 | +1.267 | inh | 1.26 | 3.66 / 3.22 | 's'×16; 'unction'×13; 'nor'×8 (25%) | …,' join independent clauses whereas subordin [[ating]] | 'conne', 'parallel', 'arrow', 'cro' / 'simulations', 'visitors', 'customer', 'fans' |
| 4.resid.15863 | 2.82 | +0.119 | +0.058 | +0.064 | +0.831 | exc | 1.73 | 6.84 / 7.10 | 'partici'×16; 'Simple'×13; 'past'×8 (25%) | …actions [[completed]] | 'sentences', 'constru', 'verb', 'pron' / 'resistance', 'control', 'advers', 'economic' |
| 3.mlp.7785 | 2.03 | +0.051 | +0.046 | +0.052 | +1.756 | inh | 1.09 | 5.93 / 4.30 | 'ating'×58; 'uses'×4; 'ound'×2 (91%) | …or more independent clauses joined by coordin [[ating]] | 'eld', 'ordin', 'scenes', 'see' / 'abs', 'Mal', 'buff', 'ber' |

### G and L_C disagree

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.resid.38697 | 1.36 | +0.049 | +0.024 | +0.023 | +1.173 | inh | 2.23 | 13.38 / 7.21 | 'ating'×49; 'ated'×15 (77%) | …<s> The basics of coordin [[ating]] | 'efforts', 'tasks', 'task', 'coordin' / 'black', 'accept', 'drag', 'dry' |
| 0.mlp.5468 | 0.88 | -0.023 | +0.011 | +0.019 | +0.758 | inh | 1.30 | 5.59 / 1.57 | 'ating'×35; 'ated'×25; 'ator'×3 (55%) | …<s> Prior to coordin [[ated]] | 'cla', 'avig', 'Francis', 'definit' / 'cks', 'que', 'mountains', 'privile' |
| 0.resid.15578 | 1.36 | -0.008 | -0.001 | +0.015 | +0.905 | inh | 0.00 | 10.32 / 7.62 | 'ute'×30; 'pc'×6; 'Parti'×4 (47%) | … [[std]] | 'angen', 'thinking', 'print', 'endor' / 'Julia', 'Earl', 'Aur', 'Giorg' |
| 0.mlp.15578 | 0.97 | +0.009 | +0.001 | +0.013 | +0.929 | inh | 0.00 | 10.05 / 7.65 | 'vare'×34; 'pc'×6; 'Parti'×4 (53%) | … [[std]] | 'NER', 'angen', 'ely', 'person' / 'Julia', 'Water', 'Aur', 'Chinese' |

---

## Target 7.mlp.15696  (345 nodes; release L_C -0.303)

- **peak tokens:** 'dr'×64 (consistency 100%)
- activating: …and involves a rotor-driven [[dr]]
- activating: …and involves a rotor-driven [[dr]]
- activating: …which used compressed air to drive the [[dr]]
- contrast: …which a complex mixture of different uses has multiple applications, including cooling and lubricating parts of the drill bit
- contrast: …driller has access to a range of options, including subterranean walls and barriers. can also use in
- contrast: …the various aspects of drilling methodologies including drill techniques using an approach that involves extremes.<s> Over the course
- target logit effect: + ['lawyer', 'Anna', 'Tol', 'ina', 'flesh', 'reader'] | − ['fre', 'tag', 'quad', 'ax', 'Action', 'nt']


### inhibitory (most negative L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.attn.33630 | 1.16 | +0.102 | +0.029 | -0.013 | -1.174 | inh | 4.03 | 8.70 / 8.88 | 'ED'×6; 'H'×6; 'S'×6 (9%) | …<s> ECONOMIC DE [[VE]] | 'Sciences', 'environmental', 'reset', 'sciences' / 'min', '7', 'Fa', 'H' |
| 0.mlp.4544 | 0.84 | +0.005 | +0.013 | -0.013 | -0.249 | inh | 0.00 | 6.40 / 5.69 | 'mode'×17; 'campion'×6; 'overline'×3 (27%) | … [[std]] | 'ically', 'ize', 'ake', 'do' / 'pixel', 'riv', 'kens', 'generate' |
| 0.resid.6299 | 1.76 | +0.949 | +0.021 | -0.012 | -2.353 | exc | 1.12 | 7.46 / 7.37 | 'inter'×10; 'new'×6; 'strateg'×4 (16%) | …research must incorporate [[inter]] | 'benef', 'vital', 'capac', 'relevant' / 'ini', 'old', 'Lad', 'Sc' |
| 0.mlp.39049 | 0.54 | +0.036 | +0.011 | -0.012 | -0.107 | exc | 0.00 | 9.41 / 9.05 | 'ouc'×6; 'campion'×6; 'buf'×4 (9%) | … [[rand]] | 'cast', 'thy', 'Pl', 'c' / 'owe', 'Pap', 'Om', 'margin' |
| 0.mlp.27056 | 1.42 | +0.057 | +0.003 | -0.012 | +0.482 | inh | 0.00 | 8.57 / 7.95 | 'with'×25; 'the'×17; 'ing'×8 (39%) | … [[ing]] | 'controllers', 'Brig', 'тел', 'ograph' / 'gray', 's', 'TB', 'Paulo' |
| 0.mlp.37468 | 1.73 | +0.077 | +0.008 | -0.011 | -0.606 | exc | 1.19 | 2.64 / 2.46 | 'modal'×33; 'tempor'×4; 'itative'×4 (52%) | …, and subsequent evolutions to quantified [[modal]] | 'frameworks', 'parameters', 'responses', 'approaches' / 'dream', 'trip', 'dust', 'sle' |

### excitatory (most positive L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.resid.22367 | 2.91 | +0.009 | -0.090 | +0.401 | +5.474 | exc | 5.04 | 5.83 / 4.57 | 'face'×29; 'ation'×10; 'ical'×7 (45%) | …GSE) and analyzing subsur [[face]] | 'Site', 'survey', 'engineering', 'Engineering' / 'concurrent', 'suggestion', 'electrons', 'Latin' |
| 6.resid.32156 | 4.57 | +0.216 | +0.225 | +0.219 | +1.431 | exc | 15.70 | 17.31 / 11.36 | 'dr'×64 (100%) | …rotary drilling, percussive [[dr]] | 'ought', 'ays', 'illing', 'ift' / 'proceed', 'O', 'René', 'por' |
| 3.resid.10677 | 3.88 | +0.003 | -0.113 | +0.216 | +3.729 | exc | 4.25 | 5.44 / 4.69 | 'face'×42; 'bore'×5; 'site'×5 (66%) | …GSE) and analyzing subsur [[face]] | 'Site', 'engineering', 'site', 'Engine' / 'concurrent', 'neg', 'Latin', 'lambda' |
| 2.mlp.36952 | 3.07 | +0.001 | -0.076 | +0.167 | +3.549 | exc | 3.76 | 4.36 / 3.40 | 'ations'×16; 'ation'×13; 'ical'×8 (25%) | …, from which they direct design of found [[ations]] | 'bore', 'Site', 'engineering', 'dr' / 'den', 'ners', 'concurrent', 'electrons' |
| 3.resid.18979 | 4.59 | +0.003 | +0.000 | +0.151 | +3.829 | inh | 4.05 | 6.17 / 4.65 | 'illing'×60; 'operations'×4 (94%) | …illing fluid systems, also known as dr [[illing]] | 'dr', 'flush', 'illing', 'oil' / 'Greek', 'patterns', 'electronic', 'displays' |
| 5.resid.39647 | 3.54 | +0.174 | +0.124 | +0.133 | +1.894 | exc | 12.10 | 13.15 / 10.25 | 'dr'×64 (100%) | …<s> The beginning of [[dr]] | 'uw', 'chn', 'min', 'ift' / 'directions', 'only', 'ach', 'dem' |

### G and L_C disagree

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.resid.18706 | 2.57 | +0.004 | -0.004 | +0.120 | +4.356 | inh | 9.38 | 10.85 / 8.05 | 'oge'×19; 'iness'×4; 'ell'×4 (30%) | …ogeographic Biomes.<s> Bi [[oge]] | 'is', "'", 'can', 'itself' / 'ou', 'slow', 'driv', 'ternoon' |
| 0.mlp.28050 | 1.46 | +0.063 | -0.029 | +0.073 | +2.426 | inh | 2.25 | 3.80 / 2.67 | 'oil'×34; 'illing'×24; 'recovery'×2 (53%) | …oil extraction process involves dr [[illing]] | 'sector', 'industry', 'illing', 'role' / 'PR', 'Greg', 'guard', 'hist' |
| 5.resid.28753 | 2.75 | +0.002 | -0.022 | +0.070 | +1.264 | inh | 13.72 | 14.98 / 10.95 | 'dr'×64 (100%) | …different soil conditions, along with percussion [[dr]] | 'ki', 'Hart', 'machine', 'screens' / 'ester', 'prov', 'onen', 'We' |
| 2.resid.7488 | 1.91 | +0.018 | -0.012 | +0.059 | +2.701 | inh | 2.27 | 3.40 / 2.83 | 'recovery'×31; 'voir'×20; 'erves'×4 (48%) | …optimal recovery from subsurface reser [[voir]] | 'oil', 'voir', 'mine', 'extra' / 'odes', 'yet', 'ors', 'diffusion' |

---

## Target 9.attn.21759  (667 nodes; release L_C -0.784)

- **peak tokens:** 'are'×18, 'these'×14, 'as'×7, 'or'×5 (consistency 28%)
- activating: …creation and economic growth. key feature of [[these]]
- activating: …creation and economic growth. key feature of [[these]]
- activating: …<s> The concept of economic zones, which [[are]]
- contrast: …value of an art collection by ensuring its appropriate use.<s> Digitalization has brought about a significant shift in the way
- contrast: …ulf to Twitter - as it changes and evolves in such an astonishing manner that it would make even amateurish speak
- contrast: …es where policymakers need to make their own decisions without damaging individual interests.<s> A Pareto Im
- target logit effect: + ['zone', 'coast', 'zones', 'Zone', 'district', 'cycle'] | − ['rep', 'racing', 'arm', 'odd', 'Oxford', 'comparison']


### inhibitory (most negative L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.resid.3989 | 3.35 | -0.009 | +0.002 | -0.263 | -18.922 | exc | 0.31 | 2.11 / 2.91 | 'language'×6; 'prevent'×4; 'measurement'×4 (9%) | …tombs utilized simple yet efficient [[light]] | 'enemy', 'lung', 'blood', 'screen' / 'exempl', 'include', 'includes', 'sees' |
| 0.attn.712 | 1.03 | +0.684 | +0.015 | -0.255 | -19.639 | inh | 1.51 | 2.55 / 2.62 | 'like'×12; '1'×10; '('×9 (19%) | … [[1]] | 'aw', 'n', 'ch', 'credit' / 'onnées', '%.', '².', '\\).' |
| 1.resid.31417 | 2.64 | -0.003 | +0.025 | -0.220 | -4.851 | inh | 0.54 | 2.28 / 1.47 | 'LE'×7; 'RE'×6; 'P'×5 (11%) | …<s> ENERGY [[ME]] | 'simultane', 'ug', 'Lock', 'N' / 'manifold', 'trouble', 'considerable', 'destination' |
| 4.attn.30415 | 4.59 | -0.005 | +0.012 | -0.191 | -17.901 | inh | 0.17 | 1.33 / 1.34 | 'the'×7; 'ing'×4; 'ancient'×4 (11%) | …algorithms have evolved remarkably since the [[dawn]] | 'icles', 'which', 'endpoint', 'secret' / 'fav', 'remains', 'wider', 'Stanley' |
| 0.mlp.12366 | 1.05 | +0.021 | -0.008 | -0.176 | -1.500 | inh | 0.84 | 3.09 / 2.99 | 'Gem'×7; 'cul'×7; 'lob'×6 (11%) | …uss.<s> Glen Cook.<s> David [[Gem]] | 'Description', 'crete', 'Qual', 'persons' / 'faster', 'rapid', 'expon', 'que' |
| 0.mlp.11297 | 1.15 | +0.011 | +0.033 | -0.172 | -17.707 | inh | 0.25 | 3.89 / 3.98 | 'variables'×6; 'modules'×5; 'patterns'×5 (9%) | …states; substitute words for qualities like [[attributes]] | 'like', 'such', 'cape', 'pace' / 's', 'descri', 'lives', 'su' |

### excitatory (most positive L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.resid.21398 | 3.70 | -0.008 | -0.001 | +0.946 | -100.041 | inh | 0.10 | 3.19 / 3.17 | 'electric'×2; 'effort'×2; 'orph'×2 (3%) | …inputs. equations often capture the dynamics of [[electric]] | 'coffee', 'oby', 'zo', 'bridge' / 'types', 'itself', 'mi', "'" |
| 2.resid.10434 | 4.09 | -0.009 | -0.004 | +0.946 | -116.777 | inh | 0.31 | 2.59 / 2.75 | 'es'×2; 'ium'×2; 'effic'×2 (3%) | …café [[estava]] | 'long', 'halb', 'ared', 'bridge' / 'types', 'Cent', "'", 'prov' |
| 8.resid.5991 | 2.83 | +0.733 | +0.676 | +0.937 | +32.114 | exc | 3.37 | 26.36 / 0.06 | 'zones'×60; 'zone'×2; 'Zone'×2 (94%) | …areas.<s> Governments often use economic [[zones]] | 'zones', 'zone', 'fires', 'Islands' / 'Anal', 'phr', 'conjug', 'Cr' |
| 7.resid.21603 | 2.79 | +0.545 | +0.695 | +0.914 | +37.247 | exc | 1.89 | 22.17 / 0.04 | 'zones'×50; 'zone'×14 (78%) | …ical zones [[zone]] | 'zones', 'zone', 'fires', 'Capital' / 'quantity', 'Hor', 'Anal', 'Cr' |
| 1.resid.277 | 0.94 | -0.035 | +0.003 | +0.858 | -58.711 | inh | 0.60 | 2.09 / 2.01 | '�'×5; 'on'×4; 'es'×3 (8%) | …О [[н]] | 'iale', 'halb', 'long', 'ign' / 'Char', 'Cent', 'k', 'g' |
| 0.mlp.27810 | 1.17 | -0.023 | +0.486 | +0.805 | +51.491 | exc | 1.18 | 5.08 / 0.00 | 'zones'×60; 'zone'×4 (94%) | …to the warm tropical zones, each climate [[zone]] | 'zones', 'zone', 'curl', 'radiation' / 'Regin', 'lis', 'fi', 'surve' |

### G and L_C disagree

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.attn.18179 | 0.89 | +0.780 | +0.127 | +0.692 | -37.950 | inh | 0.01 | 4.03 / 3.99 | 'Cov'×17; 'spons'×6; 'kern'×5 (27%) | … [[tutorial]] | 'Point', 'snapshot', 'GP', 'DNS' / 'mar', 'S', 'accord', 'slow' |
| 1.mlp.17688 | 3.24 | +0.000 | -0.003 | +0.213 | -24.787 | inh | 0.23 | 1.13 / 1.07 | 'ON'×9; '-'×5; 'IC'×5 (14%) | …CDegoTAXDEBRANEN [[ON]] | 'just', 'vet', 'und', 'bird' / 'Cent', ',', 'er', 'pro' |
| 0.resid.35895 | 1.25 | +0.020 | +0.001 | +0.190 | +7.377 | inh | 1.08 | 3.07 / 2.61 | 'habitat'×11; 'water'×9; 'habit'×6 (17%) | …catch in fisheries, pollution from [[ocean]] | 'weather', 'aqu', 'habitat', 'tropical' / 'execution', 'letter', 'performance', 'Gu' |
| 2.mlp.14460 | 1.91 | +0.006 | -0.094 | -0.171 | +4.801 | exc | 0.65 | 3.98 / 3.85 | 'ens'×11; 'neur'×7; 'R'×7 (17%) | …. involves antigenic stage stages where [[foreign]] | 'types', 'structure', 'composition', 'processing' / 'Mul', 'hy', 'oby', 'next' |

---

## Target 10.mlp.21748  (376 nodes; release L_C +0.228)

- **peak tokens:** 'such'×64 (consistency 100%)
- activating: …using metaphors [[such]]
- activating: …using metaphors [[such]]
- activating: …symbolism within stories.<s> Symbols [[such]]
- contrast: …formative stage (forming, storming, norming and performing), theoretical perspectives can be drawn that explain
- contrast: …outlook and cultural evolution can be observed in many ways throughout history. individuals may see transformation as a chance for growth and
- contrast: …ional und human beings relationship for emotions.<s> By delving into the psychological aspects of character development, readers
- target logit effect: + ['path', 'ab', 'as', 're', 'distribution', 'pro'] | − ['acker', 'IM', 'erved', 'cookies', 'fri', 'Name']


### inhibitory (most negative L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8.resid.20013 | 2.66 | +0.497 | -0.076 | -0.028 | -1.642 | exc | 25.11 | 27.52 / 9.28 | 'such'×64 (100%) | …, when faced with such circumstances [[such]] | 'applications', 'as', 'that', 'scenarios' / 'endo', 'Mol', 'skip', 'Bowl' |
| 0.attn.5663 | 1.00 | +1.115 | -0.003 | -0.013 | -68.538 | inh | 3.91 | 4.29 / 4.37 | 'symbols'×19; 'symbol'×10; 'graph'×4 (30%) | … [[symbols]] | 'pass', 'int', ',', 'fl' / 'master', 'ffen', 'service', 'Scala' |
| 8.resid.30800 | 2.11 | +0.095 | -0.019 | -0.012 | -1.542 | inh | 4.83 | 9.82 / 2.54 | 'such'×64 (100%) | …rams led to the development of such devices [[such]] | 'as', 'figures', 'events', 'gi' / 'hen', 'bounded', 'multi', 'Cl' |
| 9.resid.7581 | 0.54 | -0.022 | -0.009 | -0.010 | -0.397 | inh | 5.90 | 8.60 / 1.57 | 'such'×64 (100%) | …method. tricks are employed also, [[such]] | 'reverse', 'where', 'cool', 'tack' / 'icons', 'metrics', 'goals', 'Cultural' |
| 9.resid.21452 | 0.84 | -0.014 | -0.008 | -0.009 | -0.563 | inh | 8.32 | 10.29 / 2.63 | 'such'×64 (100%) | …aminants [[such]] | 'previous', 'past', 'pop', 'pain' / 'equilibrium', 'stable', 'ordering', 'ove' |
| 0.attn.15350 | 1.04 | -0.011 | -0.004 | -0.008 | -1.423 | inh | 0.26 | 0.61 / 0.62 | '.'×64 (100%) | …....... [[.]] | 'dist', 'rod', 'cart', 'division' / 'operators', 'min', 'HL', 'Command' |

### excitatory (most positive L_C)

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.resid.7055 | 1.38 | +0.005 | +0.961 | +1.039 | +14.469 | exc | 3.29 | 3.77 / 1.43 | 'such'×64 (100%) | …with such circumstances such emergencies, [[such]] | 'as', 'AS', 'wiel', 'que' / 'ult', 'come', 'ownership', 'welcome' |
| 0.resid.14756 | 1.69 | +0.038 | +0.827 | +1.039 | +12.979 | exc | 3.72 | 4.05 / 1.49 | 'such'×64 (100%) | …gencies, such as such at times [[such]] | 'as', 'Asp', 'j', 'Ash' / 'ols', 'OT', 'ogn', 'our' |
| 2.resid.19422 | 2.37 | +0.567 | +1.090 | +1.039 | +17.613 | exc | 5.95 | 6.43 / 2.32 | 'such'×64 (100%) | …, when faced with such circumstances [[such]] | 'as', 'rule', 'thing', 'arrangement' / 'Hor', 'ult', 'Reserve', 'onic' |
| 1.resid.29733 | 2.07 | +0.060 | +1.061 | +1.039 | +14.603 | exc | 3.60 | 3.93 / 1.54 | 'such'×64 (100%) | …gencies, such as such at times [[such]] | 'wiel', 'as', 'associations', 'dy' / 'to', 'we', 'w', 'Mus' |
| 0.mlp.7055 | 1.67 | +0.036 | +1.090 | +1.039 | +21.799 | exc | 4.32 | 4.93 / 1.82 | 'such'×64 (100%) | …with such circumstances such emergencies, [[such]] | 'as', 'AS', 'conven', 'occasions' / 'ult', 'ov', 'old', '(' |
| 0.attn.4743 | 1.21 | -0.003 | +0.488 | +0.852 | +10.724 | exc | 0.33 | 0.39 / 0.13 | 'such'×64 (100%) | …gencies, such as such at times [[such]] | 'such', 'Bet', 'rollo', 'elf' / 'Mon', 'cons', 'miss', 'Sig' |

### G and L_C disagree

| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.resid.23777 | 1.77 | +0.039 | +0.066 | +0.211 | +6.055 | inh | 4.38 | 4.70 / 1.61 | 'such'×64 (100%) | …compressed technology such ASCIi designs [[such]] | 'as', 'Joseph', 'rapid', 'wie' / 'ourse', 'rew', 'Classic', 'teen' |
| 2.attn.4806 | 3.22 | -0.014 | +0.035 | +0.160 | +7.315 | inh | 0.18 | 2.23 / 2.91 | 'istic'×2; 'co'×2; 'culture'×2 (3%) | …purpose, including creating expressive futur [[istic]] | '.', '".', '.\\', '."' / 'yz', 'ize', 'arily', 'develop' |
| 8.resid.28363 | 2.52 | +0.542 | +0.057 | +0.068 | +1.249 | inh | 36.70 | 39.51 / 13.03 | 'such'×64 (100%) | …when developing an effective methods such meditations [[such]] | 'as', 'As', 'Associ', 'wie' / 'launch', '', '', 'tabular' |
| 1.resid.2159 | 1.31 | +0.018 | +0.023 | +0.063 | +3.502 | inh | 3.45 | 3.65 / 1.22 | 'such'×64 (100%) | …gencies, such as such at times [[such]] | 'clos', 'as', 'gradient', 'pine' / 'policy', 'Policy', 'cli', 'ownership' |