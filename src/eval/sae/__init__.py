"""SAE quality evaluation: data extraction and figures for the paper appendix.

Modules
-------
training_logs       Parse sae-system training logs (final loss/EV/dead features).
latent_density      Alive fractions and firing-rate densities from latent_stats.pt.
reconstruction_eval GPU pass: fresh explained variance (and optional CE recovered)
                    for all 36 SAEs.
figures             Render the SAE quality appendix figure from the CSVs above.
report              Orchestrate the CPU-only steps and the figure.

All outputs go to a new output directory (default ``analysis-restyled/sae-eval``);
source directories (sae-system, Runs) are only ever read.
"""
