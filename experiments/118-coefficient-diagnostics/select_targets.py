"""118 target list and CPU-side prep (no GPU).

1. Target sample: scored, non-vacuous TuringLLM production targets (WCM v4, T3 main, held-out strongest), N_PER per
   depth band x site kind cell (3 x 3), drawn uniformly at random with a fixed seed. Not pass/fail-enriched, so the
   sample pass rate estimates the production rate within each cell (cells are equal-sized, so the pooled rate is a
   cell-balanced rate, not the production-weighted one).
2. The scoring OUT for the driver: version markers copied from the production OUT (no circuits copied; run118.py loads
   the production circuits read-only).
3. The natural-maximum reference: outputs/top_ctx.pt ctx_seq_val[comp, latent, 0] (each latent's largest per-sequence
   maximum over the corpus pass) and [..., 63] (its 64th largest), as float32 out/natmax.npy [2, n_comp, d_sae].

  python -I -X utf8 experiments/118-coefficient-diagnostics/select_targets.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "109-pod-run-analysis"))
import lib109 as L  # noqa: E402

SEED = 20261115
N_PER = 24
ARCH = ROOT / "experiments" / "109-pod-run-analysis" / "out" / "archives"
OUT_V4 = ARCH / "pub-turingllm-T3-main" / "root/turing-tl/experiments/094-tl-v2-records/out_v2"


def production_rows():
    dirs = [d / "main" for d in L.find("turingllm", "out") if (d / "main").exists()]
    df = L.frame(L.read_rows(dirs, "eval"))
    df = df[df.held == "strong"].copy()
    df["vacuous"] = L.vacuous(df).values
    df["passes"] = (L.passes(df) & ~df["vacuous"]).values
    return L.add_site(df, "turingllm")


def main():
    (HERE / "out").mkdir(exist_ok=True)
    df = production_rows()
    ok = df[~df.vacuous]
    print("production rows %d, non-vacuous %d, pass %.1f%%" % (len(df), len(ok), 100 * ok.passes.mean()))
    rng = np.random.default_rng(SEED)
    pick = []
    for band, _, _ in L.bands_of("turingllm"):
        for kind in ("attn", "mlp", "resid"):
            pool = ok[(ok.band == band) & (ok.kind == kind)].sort_values("seed").seed.tolist()
            pick += [pool[i] for i in sorted(rng.choice(len(pool), size=min(N_PER, len(pool)), replace=False))]
    sel = ok.set_index("seed").loc[sorted(pick)]
    (HERE / "targets_tl.txt").write_text("\n".join(sel.index) + "\n")
    cols = ["layer", "kind", "band", "n", "passes"] + L.HEAD + [L.NEC, "a_pos_tk"]
    sel[cols].to_csv(HERE / "out" / "targets_tl.csv")
    print("picked %d (%d pass in production)" % (len(sel), sel.passes.sum()))
    print(sel.groupby(["band", "kind"]).passes.agg(["size", "mean"]))

    rs = HERE / "out" / "rs"
    (rs / "main").mkdir(parents=True, exist_ok=True)
    for m in ("WCM_VERSION", "WCM_R"):
        shutil.copyfile(OUT_V4 / m, rs / m)

    nm = HERE / "out" / "natmax.npy"
    if not nm.exists():
        import torch
        t = torch.load(ROOT / "outputs" / "top_ctx.pt", weights_only=False, map_location="cpu")["ctx_seq_val"]
        np.save(nm, np.stack([t[..., 0].float().numpy(), t[..., 63].float().numpy()]).astype(np.float32))
        print("natmax", nm)


if __name__ == "__main__":
    main()
