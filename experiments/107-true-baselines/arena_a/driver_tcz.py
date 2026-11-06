"""Arena A (DAN-195 / DAN-197, item A3): 062's production driver (experiments/062-h100-protocol-v1/driver.py, unchanged)
on the gpt2-small-tcz family (wcm_tcz.py: Kissane z-SAEs + Dunefsky transcoders), for latent targets under
WCM_VERSION=v4 (the default here; IMPLEMENTATION_PLAN section 0). As arena C's driver_marks.py: the family is
registered (and SAEBank(...)'s dispatch pointed at TCZSAEBank) before the driver is imported, because the driver
configures the family's stores at import.

One runtime change to the driver's behaviour [ours, settled]: arena A runs float32 with no TF32 (README section 5),
while 062's public_setup() sets torch.set_float32_matmul_precision("high"). driver.public_setup is wrapped so the
precision is put back to "highest" right after it; nothing else differs. ARENA_A_TF32=1 keeps the driver's TF32.

  FAMILY=gpt2-small-tcz DATA=data_public/gpt2-small STORES=<data_pass_tcz.py --out> OUT=<dir> CTX_DIR=<records> \
  TARGETS=<file of L.kind.i keys> MODE=main WCM_VERSION=v4 ORDER=list SKIP_SPEC=1 SAE_PRELOAD=0 DEEP_BS=1 \
  HF_HUB_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python experiments/107-true-baselines/arena_a/driver_tcz.py
Env of wcm_tcz: ARENA_A_BOS (1, settled), ARENA_A_SCOPE (a1 = z + z2 + tc, default; a2 = tc only).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import wcm_tcz as WT  # noqa: E402

if os.environ.get("FAMILY", WT.FAMILY) != WT.FAMILY:
    raise SystemExit("driver_tcz.py runs FAMILY=%s only" % WT.FAMILY)
os.environ["FAMILY"] = WT.FAMILY
os.environ.setdefault("WCM_VERSION", "v4")
os.environ.setdefault("DATA", str(ROOT / "data_public" / "gpt2-small"))
if os.environ.get("ARENA_A_BOS", "1") != "1":
    raise SystemExit("arena A is settled on ARENA_A_BOS=1 (got %r)" % os.environ["ARENA_A_BOS"])
WT.register(patch_bank_dispatch=True)
sys.path.insert(0, str(ROOT / "experiments" / "062-h100-protocol-v1"))
import driver  # noqa: E402
import torch  # noqa: E402

_public_setup = driver.public_setup


def _public_setup_fp32():
    G = _public_setup()
    if os.environ.get("ARENA_A_TF32") != "1":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return G


driver.public_setup = _public_setup_fp32

# AA_RANK_W (2026-11-05, A13 tuning; TUNING.md): the rank-keep weight gamma_R of this process's WCM version, in every
# mode (062's driver reaches it only through SWEEP_ARMS _r in MODE=sweep). On this substrate k = d_sae (ReLU), so no
# latent is ever censored and rank-keep's Top-K motivation does not apply. Unset = the version's own (v4: 1e-1),
# unchanged. A run under an override marks OUT/AA_RANK_W and refuses an OUT fitted with another weight or without it.
AA_RANK_W = os.environ.get("AA_RANK_W")
if AA_RANK_W:
    _w = float(AA_RANK_W)
    driver.WCM_RANK[driver.WCM_VERSION] = _w
    driver.RANK = _w
    _m = driver.OUT / "AA_RANK_W"
    if _m.exists():
        if float(_m.read_text()) != _w:
            raise SystemExit("OUT %s was fitted with AA_RANK_W=%s, this process runs %g" % (driver.OUT, _m.read_text().strip(), _w))
    else:
        if any((driver.OUT / d / "circuits").is_dir() and any((driver.OUT / d / "circuits").rglob("*.pt"))
               for d in ("main", "sweep")):
            raise SystemExit("OUT %s already holds circuits fitted without AA_RANK_W" % driver.OUT)
        driver.OUT.mkdir(parents=True, exist_ok=True)
        _m.write_text("%g\n" % _w)

if __name__ == "__main__":
    driver.log("arena A: family %s, scope %s, BOS %s, matmul precision after setup: %s" % (
        WT.FAMILY, WT.scope(), WT.bos_prefix(), "TF32 (ARENA_A_TF32=1)" if os.environ.get("ARENA_A_TF32") == "1"
        else "highest"))
    driver.main()
