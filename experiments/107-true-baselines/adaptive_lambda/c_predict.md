| policy | pass rate | median size (passing) | size vs oracle (passing) |
|---|---|---|---|
| fixed λ 0.00025 | 0.61 | 957 | 3.26 |
| fixed λ 0.0005 | 0.50 | 543 | 2.45 |
| fixed λ 0.001 | 0.46 | 402 | 1.88 |
| fixed λ 0.002 | 0.37 | 261 | 1.42 |
| fixed λ 0.004 | 0.34 | 184 | 1.00 |
| oracle (largest passing λ) | 0.65 | 277 | 1.00 |
| A: ladder down from 4e-3, select on training-context Z (mean 1.8 fits) | 0.38 | 201 | 1.00 |
| A: ladder from 1e-3 up / down on training-context Z (mean 2.9 fits) | 0.38 | 201 | 1.00 |
| C: GBM, static, shift 0 | 0.48 ± 0.01 | 283 | 1.00 |
| C: GBM, static, shift 1 | 0.49 ± 0.01 | 389 | 1.48 |
| C: RF, static, shift 0 | 0.46 ± 0.00 | 284 | 1.00 |
| C: RF, static, shift 1 | 0.49 ± 0.01 | 364 | 1.43 |
| C: GBM, static+probe, shift 0 | 0.47 ± 0.01 | 275 | 1.00 |
| C: GBM, static+probe, shift 1 | 0.51 ± 0.01 | 365 | 1.44 |
| C: RF, static+probe, shift 0 | 0.45 ± 0.01 | 281 | 1.00 |
| C: RF, static+probe, shift 1 | 0.50 ± 0.01 | 362 | 1.43 |
