"""PER-FAMILY CONSTANTS for the 062 analysis scripts (merge.py, pass_rule.py, curves.py, paper_sweep.py), DAN-40/41 B4.

The analysis was written for TuringLLM: Top-K k = 128, 3 site kinds x 12 layers, depth bands L0-4 / L5-7 / L8-11 (merge,
paper_sweep) and layers 0-3 / 4-7 / 8-11 (curves, pass_rule). FAMILY (env, the same name driver.py's FAMILY takes)
selects the table row; the default is turingllm, whose row holds exactly the old constants, so every TuringLLM output
is unchanged (checked byte for byte on out_full, 2026-10-23).

  turingllm   k 128, 12 layers, attn / mlp / resid; bands L0-4 / L5-7 / L8-11; curve bands 0-3 / 4-7 / 8-11; 15,046
  gpt2-small  k 32, 12 layers, attn / resid_mid / mlp / resid; bands and curve bands as TuringLLM (same depth); 1,034
  pythia-70m  k 16, 6 layers, attn / mlp / resid; bands AND curve bands in thirds L0-1 / L2-3 / L4-5 (D7); 1,024

n_layer, kinds and k of the public families come from src/model/families.py (the adapters' own table). BANDS (env)
overrides both band sets, e.g. BANDS=0-1,2-3,4-5 or BANDS=0,1,2,3,4,5 (one band per layer); K_TOPK (env) overrides k.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from model.families import FAMILIES, TURINGLLM  # noqa: E402

TWELVE = ((0, 4), (5, 7), (8, 11))                  # merge.py / paper_sweep.py depth bands on a 12-layer model
TWELVE_CURVES = ((0, 3), (4, 7), (8, 11))           # curves.py / pass_rule.py bands on a 12-layer model
THIRDS_6 = ((0, 1), (2, 3), (4, 5))                 # Pythia-70m (6 layers), both uses (D7)
TABLE = {
    "turingllm": dict(k=128, bands=TWELVE, curve_bands=TWELVE_CURVES, n_full=15046),
    "gpt2-small": dict(bands=TWELVE, curve_bands=TWELVE_CURVES, n_full=1034),
    "pythia-70m": dict(bands=THIRDS_6, curve_bands=THIRDS_6, n_full=1024),
}


def parse_bands(text):
    """'0-1,2-3,4-5' -> ((0, 1), (2, 3), (4, 5)); a bare '3' is the one-layer band (3, 3)."""
    out = []
    for part in (p.strip() for p in text.split(",") if p.strip()):
        lo, _, hi = part.partition("-")
        out.append((int(lo), int(hi or lo)))
    return tuple(out)


def family(name=None):
    """The analysis constants of one family: name, k, n_layer, kinds, bands, curve_bands (tuples of (lo, hi) layers,
    inclusive), n_full (the planned run's target count, for merge.py's projection)."""
    name = name or os.environ.get("FAMILY", TURINGLLM)
    if name not in TABLE:
        raise ValueError("unknown FAMILY %r; known: %s" % (name, sorted(TABLE)))
    fam, row = FAMILIES[name], dict(TABLE[name])
    k = row.pop("k", None) or fam.sae.k
    if os.environ.get("K_TOPK"):
        k = int(os.environ["K_TOPK"])
    if os.environ.get("BANDS"):
        row["bands"] = row["curve_bands"] = parse_bands(os.environ["BANDS"])
    for bands in (row["bands"], row["curve_bands"]):
        covered = [L for lo, hi in bands for L in range(lo, hi + 1)]
        if covered != list(range(fam.n_layer)):
            raise ValueError("bands %s do not tile layers 0-%d of %s in order" % (bands, fam.n_layer - 1, name))
    return dict(name=name, k=int(k), n_layer=fam.n_layer, kinds=tuple(fam.kinds), **row)


def band_label(lo, hi, fmt="L%d-%d"):
    """'L0-4' for a band, 'L3' for a one-layer band (same fmt rule with any prefix)."""
    return fmt % (lo, hi) if hi != lo else fmt.split("%d")[0] + "%d" % lo
