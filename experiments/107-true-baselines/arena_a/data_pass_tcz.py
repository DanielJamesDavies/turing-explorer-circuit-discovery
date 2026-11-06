"""Arena A (item A5): 069's data pass (experiments/069-public-data-pass/run_data_pass.py, unchanged) for the
gpt2-small-tcz family, so latent targets get protocol contexts (circuit.protocol_contexts.build_record reads its
stores). As arena C's data_pass_marks.py.

It registers the family (wcm_tcz.register(patch_bank_dispatch=True): SAEBank(...) builds TCZSAEBank) and widens the
choices of run_data_pass's --family option for this process [ours: no file change]. Everything else is 069's code.
The stores are allocated on the family grid: 12 layers x 3 kinds (z, z2, tc) = 36 components x 24,576 latents; the
(L, z2) components at the 10 narrow layers are inert NullSAE sites (never active, so empty stores).

Default stages pass1,negctx: what build_record needs (latent_stats, top_ctx, mid_ctx, seq_repr, neg_ctx). pass2
(top_coactivation) is left out: its PMI derives the token count as active_count.sum() // k, which assumes a Top-K
dictionary; with k = d_sae on a ReLU dictionary that count is wrong.

  HF_HUB_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python experiments/107-true-baselines/arena_a/data_pass_tcz.py \
      --data data_public/gpt2-small --out experiments/107-true-baselines/arena_a/out/pilot/datapass \
      --batch-size 64 --mid-warmup 4
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "069-public-data-pass"))
if os.environ.get("ARENA_A_BOS", "1") != "1":
    raise SystemExit("arena A is settled on ARENA_A_BOS=1 (got %r)" % os.environ["ARENA_A_BOS"])
import wcm_tcz as WT  # noqa: E402

WT.register(patch_bank_dispatch=True)
_orig_add = argparse.ArgumentParser.add_argument


def _add(self, *names, **kw):
    if "--family" in names and "choices" in kw:
        kw["choices"] = list(kw["choices"]) + [WT.FAMILY]
    return _orig_add(self, *names, **kw)


argparse.ArgumentParser.add_argument = _add
import run_data_pass as R  # noqa: E402

# seq_repr [ours]: pass 1 stores each sequence's mean-pooled LAST-LAYER "resid" site input (the close contrast
# selector's similarity space) and looks the kind up as bank.kinds.index("resid"); this family has no resid site. The
# stand-in is (11, tc)'s input = the full ln_2 output of the last block (LayerNorm of the last mid-block residual
# stream, d_model 768), the nearest residual-stream read the family captures. Only pass 1's lookup is redirected
# (a list subclass for the duration of run_first_pass); the bank's kind order is unchanged.
SEQ_REPR_KIND = "tc"


class _Kinds(list):
    def index(self, value, *args):
        return list.index(self, SEQ_REPR_KIND if value == "resid" else value, *args)


def _wrap_first_pass():
    import pipeline.distributed  # noqa: F401  (run_data_pass's import order: avoids a runtime <-> distributed cycle)
    import pipeline.first_pass as FP
    from pipeline.runtime import get_runtime
    orig = FP.run_first_pass

    def run_first_pass(*a, **kw):
        bank = get_runtime().bank
        kinds = bank.kinds
        bank.kinds = _Kinds(kinds)
        try:
            return orig(*a, **kw)
        finally:
            bank.kinds = kinds
    FP.run_first_pass = run_first_pass


_configure = R.common.configure


def _configure_then_wrap(*a, **kw):
    """pipeline.first_pass imports store.*, which must follow 069's configure(): wrap right after it."""
    cfg = _configure(*a, **kw)
    _wrap_first_pass()
    return cfg


R.common.configure = _configure_then_wrap

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--family" not in argv:
        argv = ["--family", WT.FAMILY] + argv
    if not any(a.startswith("--stages") for a in argv):
        argv += ["--stages", "pass1,negctx"]
    R.main(argv)
