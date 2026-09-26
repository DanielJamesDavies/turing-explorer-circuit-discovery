# Running example under protocol v1: 3.resid.35381

**Target.** Peak tokens 'temperature'×61, 'atures'×3 (consistency 95%). Logits + ['threshold', 'change', 'Day', 'quant', 'changes', 'causes'] | − ['Bé', 'vet', 'pped', 'wider', 'ptop', 'cious']
- …critical point is reached at a specific [[temperature]]and pressure,
- …<s> Internal energy of a system is strongly dependent on [[temperature]]and pressure (
- …<s> Internal energy of a system is strongly dependent on [[temperature]]and pressure (
- …the Curie [[temperature]]on average shows
- …(U = T) was not present), S implies that [[temperature]]was present and

## Scores (held-out; activation read)

| arm | held | nodes | free0 | freeM_topk | freeN_topk | necessity | sufficiency to induce |
|---|---|---|---|---|---|---|---|
| B | strong | 294 | 0.986 | 1.209 | 1.228 | 0.982 | 0.974 |
| B | mid | 294 | 1.028 | 1.424 | 1.303 | 0.971 | 1.602 |
| Bw | strong | 301 | 0.979 | 1.192 | 1.191 | 1.000 | 1.039 |
| Bw | mid | 301 | 1.085 | 1.373 | 1.320 | 0.900 | 1.707 |

## Specificity (held-out strongest; pre-activation faithfulness)

| arm | ablation | target | siblings (median) | control (median) | n siblings | lifted above cut (circuit / empty) | target in Top-K |
|---|---|---|---|---|---|---|---|
| B | Z | - | -0.019 | 0.723 | 47 | 10710.0 / 40589.0 | 1.000 |
| B | A | 1.208 | 0.926 | 1.364 | 47 | 1246.0 / 5.0 | 1.000 |
| B | C | 1.228 | 0.884 | 1.201 | 47 | 1275.0 / 2.0 | 1.000 |
| Bw | Z | - | 0.033 | 0.289 | 47 | 10935.0 / 40589.0 | 0.938 |
| Bw | A | 1.193 | 0.806 | 1.337 | 47 | 1322.0 / 5.0 | 1.000 |
| Bw | C | 1.191 | 0.725 | 1.124 | 47 | 1187.0 / 2.0 | 1.000 |

## Top 15 nodes, arm B (by consensus attribution; direct-to-target share Z/A/C 0.23 / 0.27 / 0.25)

| node | α | attribution (min over Z/A/C, share) | sign | peak tokens (consistency) | e.g. | on target ctx (anchor / seq max) | on contrast (seq max) | logits + |
|---|---|---|---|---|---|---|---|---|
| R2/29808 | 1.29 | 0.024 | + | 'atures'×44; 'ature'×17; 'temperature'×3 (69%) | …is more obvious that minimum temper [[atures]]are reduced at | 4.64 / 5.82 | 0.25 | 'causes', 'measured', 'eros', 'conditions' |
| R2/23176 | 1.16 | 0.023 | + | 'price'×49; 'temperature'×15 (77%) | …Third Law of Thermodynamics holds that at zero [[temperature]](or almost | 5.30 / 5.54 | 0.03 | 'points', 'threshold', 'nodes', 'camp' |
| R0/39652 | 1.40 | 0.021 | + | 'atures'×56; 'temperature'×8 (88%) | …measures temperature relative to temperature; movement of particles at higher temper [[atures]]due to motion | 2.85 / 3.20 | 0.01 | 'pool', 'hood', '-', 'burn' |
| R2/31405 | 1.32 | 0.017 | + | 'atures'×64 (100%) **string detector?** | …maintaining temper [[atures]]below a specific | 2.96 / 4.82 | 0.15 | 'temper', 'ranges', 'atures', 'extreme' |
| R2/3261 | 1.26 | 0.016 | + | 'temper'×35; 'war'×25; 'temperature'×4 (55%) | …viscosity is one of the fundamental properties that vary with [[temperature]]; in therm | 3.30 / 4.03 | 0.16 | 'bo', 'temper', 'weapons', 'temperature' |
| R2/12030 | 1.03 | 0.015 | + | 'temperature'×64 (100%) **string detector?** | …temperature applications, temperature is a crucial factor due to the [[temperature]]-dependent magnet | 5.75 / 6.16 | 0.03 | 'AL', 'of', 'bud', 'board' |
| R2/22529 | 1.26 | 0.013 | + | 'heat'×38; 'thermal'×26 (59%) | …understanding of [[thermal]]energy's | 3.71 / 4.13 | 1.66 | 'properties', 'diss', 'stress', 'profile' |
| R2/8106 | 1.30 | 0.013 | + | 'temper'×32; 'temperature'×32 (50%) | …regions with more seasonal temperatures include the tropical, [[temper]]ate or temper | 3.45 / 4.01 | 0.09 | 'Turn', 'imo', 'yard', 'Plan' |
| R1/10359 | 1.31 | 0.013 | + | 'atures'×64 (100%) **string detector?** | …temper [[atures]]result in increased | 3.20 / 4.80 | 0.02 | 'disput', 'turn', 'ats', 'atures' |
| R2/16195 | 1.68 | 0.012 | + | 'thermal'×19; 'ing'×12; 'temperature'×11; 'heat'×8 (30%) | …heat or even [[thermal]]expansion are significant | 2.57 / 3.29 | 1.52 | 'heat', 'thermal', 'cold', 'temper' |
| R1/27935 | 1.47 | 0.012 | + | 'temperature'×64 (100%) **string detector?** | … [[temperature]]below Tg | 3.15 / 3.50 | 0.10 | 'atin', 'threshold', 'C', 'treat' |
| M1/4282 | 1.29 | 0.011 | + | 'temperature'×48; 'ature'×8; 'warm'×8 (75%) | …of a system is strongly dependent on temperature and pressure (temper [[ature]]is essentially arbitrary | 2.32 / 2.44 | 0.00 | 'streets', 'eries', 'ird', 'nodes' |
| M3/17686 | 1.02 | 0.011 | + | 'temperature'×53; 'atures'×6; 'ature'×5 (83%) | …of molecules in a chemical reaction; temperature increases with increasing [[temperature]]and collision frequency | 3.24 / 3.47 | 0.16 | 'VP', 'Times', 'IN', 'modes' |
| R1/34680 | 1.10 | 0.009 | + | 'parameter'×30; 'temperature'×21; 'percentage'×12; 'gau'×1 (47%) | …location [[parameter]]specifies the | 2.96 / 3.14 | 0.03 | 'returns', 'ratings', 'click', 'eries' |
| R1/16840 | 1.18 | 0.009 | + | 'price'×57; 'prices'×4; 'taste'×2; 'demand'×1 (89%) | …sensitivity measures the variation in demand for varying goods due to [[price]]changes, while | 2.41 / 2.64 | 0.10 | 'bud', 'ier', 'charts', 'ulen' |

## Top 15 nodes, arm Bw (by consensus attribution; direct-to-target share Z/A/C 0.26 / 0.34 / 0.28)

| node | α | attribution (min over Z/A/C, share) | sign | peak tokens (consistency) | e.g. | on target ctx (anchor / seq max) | on contrast (seq max) | logits + |
|---|---|---|---|---|---|---|---|---|
| R2/29808 | 1.43 | 0.024 | + | 'atures'×44; 'ature'×17; 'temperature'×3 (69%) | …is more obvious that minimum temper [[atures]]are reduced at | 4.64 / 5.82 | 0.25 | 'causes', 'measured', 'eros', 'conditions' |
| R0/39652 | 1.35 | 0.023 | + | 'atures'×56; 'temperature'×8 (88%) | …measures temperature relative to temperature; movement of particles at higher temper [[atures]]due to motion | 2.85 / 3.20 | 0.01 | 'pool', 'hood', '-', 'burn' |
| R2/23176 | 1.12 | 0.018 | + | 'price'×49; 'temperature'×15 (77%) | …Third Law of Thermodynamics holds that at zero [[temperature]](or almost | 5.30 / 5.54 | 0.03 | 'points', 'threshold', 'nodes', 'camp' |
| R2/31405 | 1.32 | 0.017 | + | 'atures'×64 (100%) **string detector?** | …maintaining temper [[atures]]below a specific | 2.96 / 4.82 | 0.15 | 'temper', 'ranges', 'atures', 'extreme' |
| R2/3261 | 1.30 | 0.017 | + | 'temper'×35; 'war'×25; 'temperature'×4 (55%) | …viscosity is one of the fundamental properties that vary with [[temperature]]; in therm | 3.30 / 4.03 | 0.16 | 'bo', 'temper', 'weapons', 'temperature' |
| R2/12030 | 1.04 | 0.015 | + | 'temperature'×64 (100%) **string detector?** | …temperature applications, temperature is a crucial factor due to the [[temperature]]-dependent magnet | 5.75 / 6.16 | 0.03 | 'AL', 'of', 'bud', 'board' |
| R2/22529 | 1.33 | 0.013 | + | 'heat'×38; 'thermal'×26 (59%) | …understanding of [[thermal]]energy's | 3.71 / 4.13 | 1.66 | 'properties', 'diss', 'stress', 'profile' |
| M0/16799 | 1.55 | 0.013 | − | 'heat'×24; 'temperature'×15; 'thermal'×12; 'therm'×4 (38%) | …flow during thermal temperature increase.ing latent heat and specific [[heat]]is crucial | 3.68 / 4.32 | 2.22 | 'therm', 'heat', 'temper', 'temperature' |
| R0/4856 | 1.13 | 0.013 | + | 'temperature'×64 (100%) **string detector?** | … [[temperature]]dependence on temperature | 2.70 / 3.05 | 0.03 | 'control', 'flux', 'parameters', 'threshold' |
| M3/17686 | 1.21 | 0.012 | + | 'temperature'×53; 'atures'×6; 'ature'×5 (83%) | …of molecules in a chemical reaction; temperature increases with increasing [[temperature]]and collision frequency | 3.24 / 3.47 | 0.16 | 'VP', 'Times', 'IN', 'modes' |
| R1/10359 | 1.17 | 0.011 | + | 'atures'×64 (100%) **string detector?** | …temper [[atures]]result in increased | 3.20 / 4.80 | 0.02 | 'disput', 'turn', 'ats', 'atures' |
| M1/4282 | 1.23 | 0.011 | + | 'temperature'×48; 'ature'×8; 'warm'×8 (75%) | …of a system is strongly dependent on temperature and pressure (temper [[ature]]is essentially arbitrary | 2.32 / 2.44 | 0.00 | 'streets', 'eries', 'ird', 'nodes' |
| R1/27935 | 1.26 | 0.011 | + | 'temperature'×64 (100%) **string detector?** | … [[temperature]]below Tg | 3.15 / 3.50 | 0.10 | 'atin', 'threshold', 'C', 'treat' |
| R2/8106 | 1.21 | 0.011 | + | 'temper'×32; 'temperature'×32 (50%) | …regions with more seasonal temperatures include the tropical, [[temper]]ate or temper | 3.45 / 4.01 | 0.09 | 'Turn', 'imo', 'yard', 'Plan' |
| M0/24314 | 1.59 | 0.010 | − | 'paint'×14; 'prints'×7; 'gla'×6; 'esso'×6 (22%) | …s like transferring watercolor or photo-etching onto [[pap]]ier-m | 1.79 / 2.91 | 2.15 | 'technique', 'printing', 'stro', 'prints' |