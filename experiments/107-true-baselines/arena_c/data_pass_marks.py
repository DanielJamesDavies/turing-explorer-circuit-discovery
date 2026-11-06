"""Arena C (DAN-194): 069's data pass (experiments/069-public-data-pass/run_data_pass.py, unchanged) for the Marks
family, so latent targets get protocol contexts (circuit.protocol_contexts.build_record reads its stores).

It registers the family (wcm_marks.register(patch_bank_dispatch=True): SAEBank(...) builds MarksSAEBank) and lets
run_data_pass's --family accept it: its argparse lists only pythia-70m / gpt2-small, so the choices of that one option
are widened for this process [ours: no file change]. Everything else is 069's code.

Default stages pass1,negctx: what build_record needs (latent_stats, top_ctx, mid_ctx, seq_repr, neg_ctx). pass2
(top_coactivation) is left out by default: its PMI derives the token count as active_count.sum() // k, which assumes a
Top-K dictionary (exactly k active latents per token); with k = d_sae on a ReLU dictionary that count is wrong.

  HF_HUB_OFFLINE=1 ./.venv/bin/python experiments/107-true-baselines/arena_c/data_pass_marks.py \
      --data data_public/pythia-70m --out experiments/107-true-baselines/arena_c/out/datapass --batch-size 64
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "069-public-data-pass"))
import wcm_marks as WM  # noqa: E402

WM.register(patch_bank_dispatch=True)
_orig_add = argparse.ArgumentParser.add_argument


def _add(self, *names, **kw):
    if "--family" in names and "choices" in kw:
        kw["choices"] = list(kw["choices"]) + [WM.FAMILY]
    return _orig_add(self, *names, **kw)


argparse.ArgumentParser.add_argument = _add
import run_data_pass as R  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--family" not in argv:
        argv = ["--family", WM.FAMILY] + argv
    if not any(a.startswith("--stages") for a in argv):
        argv += ["--stages", "pass1,negctx"]
    R.main(argv)
