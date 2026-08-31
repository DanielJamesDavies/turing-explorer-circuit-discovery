# error-node-mode (2026-07-23)

*README generated from the scripts' docstrings; the scripts are the record.*

## `smoke.py`

Error-node mode smoke: do non-member SAE errors free-ride in our φ numbers?

Historically every φ preserves the SAE reconstruction error at EVERY upstream
site — error terms are invisible free members of every circuit. The new mode
(CircuitOnlyPatcher.keep_error_sites + collect_site_error_means) makes them
ablatable nodes, SFC-style. This measures, on L2 and L9 with a real
abl-ig_mean PA circuit:

  a_e0            empty circuit, errors preserved   (the historical anchor)
  a_e0_noerr      empty circuit, ALL errors zeroed  (denominator question:
                                                     did errors prop up empty?)
  free0           kept latents, errors preserved    (historical metric)
  free0_esites    errors kept ONLY at member sites, zeroed elsewhere
  free0_emean     errors kept ONLY at member sites, mean-filled elsewhere

If free0_esites ~= free0, non-member errors are inert at the latent endpoint
and the historical numbers stand as-is. If it drops, error nodes are
load-bearing members our circuits have been silently granted.

  PYTHONPATH=src python experiments/004-error-node-mode/smoke.py

## Result files

`smoke_rows.jsonl`
