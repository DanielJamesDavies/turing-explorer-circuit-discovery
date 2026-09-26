# Is the 11.mlp.30743 relay one feature? (random corpus sample: 4096 sequences, 262144 positions, 115 "iv" tokens)

## Behaviour per latent on the random sample

| latent | fires (positions) | precision on "iv" | recall of "iv" | top tokens it fires on (count) |
|---|---|---|---|---|
| R2/1164 | 233 | 0.489 | 0.991 | 'iv'×114, 'in'×29, 'ov'×7, 'v'×7, 'Comput'×5, 'estic'×4 |
| R3/21272 | 201 | 0.557 | 0.974 | 'iv'×112, 'id'×7, 'v'×6, 'is'×6, '2'×5, 'ur'×4 |
| R4/22454 | 205 | 0.537 | 0.957 | 'iv'×110, 'dens'×8, 'av'×6, 'uls'×5, 'Rel'×5, 'emp'×3 |
| R5/38187 | 194 | 0.588 | 0.991 | 'iv'×114, 'riv'×5, 'in'×5, 'iti'×3, 'ac'×2, 'ers'×2 |
| R6/1448 | 195 | 0.585 | 0.991 | 'iv'×114, 'v'×5, 'av'×3, 'ist'×3, 'ov'×2, 'oc'×2 |
| R7/14260 | 218 | 0.495 | 0.939 | 'iv'×108, 'opol'×5, 'in'×5, 'st'×4, 'ic'×3, 'ig'×2 |
| R9/37889 | 264 | 0.383 | 0.878 | 'iv'×101, 'Buddh'×11, 'it'×10, 'inn'×7, 'ict'×5, 'uls'×5 |
| R10/36122 | 5881 | 0.006 | 0.296 | '-'×68, 'cru'×57, 'inter'×46, 'in'×41, 'character'×38, 'pre'×36 |
| M11/30743 (target) | 116 | 0.681 | 0.687 | 'iv'×79, 'cru'×2, 'complex'×2, 'vi'×2, 'gu'×2, 'for'×1 |

## Co-firing between consecutive chain latents (and each with the target)

| pair | Jaccard of firing positions | Pearson r of activations |
|---|---|---|
| R2/1164 → R3/21272 | 0.387 | 0.946 |
| R3/21272 → R4/22454 | 0.386 | 0.950 |
| R4/22454 → R5/38187 | 0.430 | 0.931 |
| R5/38187 → R6/1448 | 0.468 | 0.943 |
| R6/1448 → R7/14260 | 0.386 | 0.874 |
| R7/14260 → R9/37889 | 0.296 | 0.895 |
| R2/1164 → M11/30743 | 0.297 | 0.759 |
| R3/21272 → M11/30743 | 0.338 | 0.753 |
| R4/22454 → M11/30743 | 0.332 | 0.808 |
| R5/38187 → M11/30743 | 0.348 | 0.723 |
| R6/1448 → M11/30743 | 0.352 | 0.621 |
| R7/14260 → M11/30743 | 0.295 | 0.811 |
| R9/37889 → M11/30743 | 0.254 | 0.743 |
| R2/1164 → R9/37889 | 0.291 | 0.833 |

## Dictionary geometry

| pair | decoder cos | rank of downstream among its dictionary (by decoder cos) | read-write cos (enc_d · dec_u) |
|---|---|---|---|
| R2/1164 → R3/21272 | 0.382 | 2 of 40960 | 0.232 |
| R3/21272 → R4/22454 | 0.471 | 1 of 40960 | 0.358 |
| R4/22454 → R5/38187 | 0.455 | 1 of 40960 | 0.331 |
| R5/38187 → R6/1448 | 0.683 | 1 of 40960 | 0.470 |
| R6/1448 → R7/14260 | 0.578 | 2 of 40960 | 0.420 |
| R7/14260 → R9/37889 | 0.504 | 1 of 40960 | 0.392 |

Random cross-dictionary pairs: median |cos| 0.023, 99th percentile 0.093.

## Census: the best "iv" detector in each dictionary (F1 on the random sample)

| site | best latent | precision | recall | F1 | in the circuit's chain? |
|---|---|---|---|---|---|
| R0 | 19408 | 0.917 | 0.670 | 0.774 | no chain node at this site |
| R1 | 9353 | 0.685 | 0.887 | 0.773 | no chain node at this site |
| R2 | 1164 | 0.489 | 0.991 | 0.655 | yes |
| R3 | 21272 | 0.557 | 0.974 | 0.709 | yes |
| R4 | 22454 | 0.537 | 0.957 | 0.688 | yes |
| R5 | 38187 | 0.588 | 0.991 | 0.738 | yes |
| R6 | 1448 | 0.585 | 0.991 | 0.735 | yes |
| R7 | 14260 | 0.495 | 0.939 | 0.649 | yes |
| R8 | 40171 | 0.409 | 0.878 | 0.558 | no chain node at this site |
| R9 | 11663 | 0.410 | 0.896 | 0.563 | no (circuit has 37889, F1 0.53) |
| R10 | 5650 | 0.587 | 0.730 | 0.651 | no chain node at this site |
| R11 | 15015 | 0.445 | 0.991 | 0.615 | no chain node at this site |
| M11 | 30743 | 0.681 | 0.687 | 0.684 | yes |