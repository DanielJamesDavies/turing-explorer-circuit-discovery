"""129 B (CPU): member composition of 117's and 120's fitted circuits (read-only .pt files) by site kind and layer,
plus alpha statistics and, for attention members, the layer band.

  PYTHONPATH=src ./.venv/bin/python -I experiments/129-known-circuits-diagnosis/b_members.py
  -> out/b_members.json, out/b_members.md
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402
from eval.wcm_eval import circuit_alphas  # noqa: E402

DIRS = {"117": ROOT / "experiments" / "117-known-circuits-gpt2" / "out" / "wcm",
        "120": ROOT / "experiments" / "120-known-circuits-errors-zeroed" / "out" / "wcm"}
KINDS = ("attn", "resid_mid", "mlp", "resid")


def main():
    out = {}
    md = ["# 129 B: circuit members by site kind / layer", ""]
    for tag, d in DIRS.items():
        rows = []
        for cname, sub in (("corpus", "main"), ("task", "task")):
            for p in sorted((d / sub / "circuits").glob("*.pt")):
                key = p.stem
                tl = int(key.split(".")[0])
                c = torch.load(p, weights_only=False, map_location="cpu")
                if c is None:
                    continue
                al = circuit_alphas(c)
                byk = defaultdict(int); byl = defaultdict(int); last = 0; att_band = defaultdict(int)
                amp = defaultdict(list)
                for (l, k), dct in al.items():
                    byk[k] += len(dct)
                    byl["%d|%s" % (l, k)] += len(dct)
                    amp[k] += list(dct.values())
                    if l == tl - 1 or (l == tl):
                        last += len(dct)
                    if k == "attn":
                        att_band["L0-2" if l <= 2 else ("L3-5" if l <= 5 else "L6+")] += len(dct)
                n = sum(byk.values())
                rows.append(dict(key=key, circuit=cname, n=n, by_kind=dict(byk), by_layer=dict(byl),
                                 adjacent_layers=last, attn_band=dict(att_band),
                                 alpha_median={k: float(torch.tensor(v).median()) for k, v in amp.items()}))
        out[tag] = rows
        md += ["## %s" % tag, "", "| target | circuit | n | attn | resid_mid | mlp | resid | attn L0-2 / L3-5 / L6+ | "
               "members in target layer & layer below |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            b = r["by_kind"]; ab = r["attn_band"]
            md.append("| %s | %s | %d | %d | %d | %d | %d | %d / %d / %d | %d |" % (
                r["key"], r["circuit"], r["n"], b.get("attn", 0), b.get("resid_mid", 0), b.get("mlp", 0),
                b.get("resid", 0), ab.get("L0-2", 0), ab.get("L3-5", 0), ab.get("L6+", 0), r["adjacent_layers"]))
        tot = defaultdict(int)
        for r in rows:
            for k, v in r["by_kind"].items():
                tot[k] += v
        T = sum(tot.values())
        md += ["", "Pooled member share by kind: " + ", ".join("%s %.3f" % (k, tot[k] / T) for k in KINDS), ""]
        att = defaultdict(int)
        for r in rows:
            for k, v in r["attn_band"].items():
                att[k] += v
        A = sum(att.values()) or 1
        md += ["Pooled attention members by band: " + ", ".join("%s %.3f" % (k, att[k] / A) for k in ("L0-2", "L3-5", "L6+")), ""]
    (HERE / "out" / "b_members.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (HERE / "out" / "b_members.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
