"""Arena D pod stage (plan D8): two CPU cross-checks of pod_stage_d.sh, repo .venv, no GPU, no download.

  weights  google/gemma-2-2b (fp32 originals, pinned c5ebcd40) cast to bf16 vs the unsloth/gemma-2-2b bf16 mirror
           (pinned 25319945), tensor by tensor (IMPLEMENTATION_PLAN §1.2): per tensor, bitwise equal after the cast?
           max |diff|. Both read lazily from the HF cache (local_files_only). Tells whether the local bf16 results
           (smokes, timings) carry over to the pod's fp32 runs. Informative: never fails the gate on a mismatch, only on
           a missing / unreadable file.
  means    caples_means.py's pod file (fp32, google weights) vs the local bf16 SMOKE file: per layer max |diff| of the
           "theirs" code / err means, the mean-token prompts equal?, and their LDs. Informative (bf16 vs fp32).

  ./.venv/bin/python -B pod_checks_d.py weights --out W.json [--google DIR --unsloth DIR]
  ./.venv/bin/python -B pod_checks_d.py means --pod POD.pt --local LOCAL.pt --out M.json
Exit: 0 written (whatever the comparison found), 1 an input is missing or unreadable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GOOGLE = ("google/gemma-2-2b", "c5ebcd40d208330abc697524c919956e692655cf")
UNSLOTH = ("unsloth/gemma-2-2b", "25319945f7fd83b8b903e12081777b7eef2ba993")


def snapshot(repo_rev, given):
    if given:
        return Path(given)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_rev[0], revision=repo_rev[1], local_files_only=True))


def tensor_index(folder: Path):
    """name -> file, over every *.safetensors of a snapshot folder."""
    from safetensors import safe_open
    idx = {}
    for f in sorted(folder.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                idx[k] = f
    return idx


def weights(a) -> dict:
    import torch
    from safetensors import safe_open
    g, u = snapshot(GOOGLE, a.google), snapshot(UNSLOTH, a.unsloth)
    gi, ui = tensor_index(g), tensor_index(u)
    if not gi or not ui:
        raise FileNotFoundError("no safetensors in %s or %s" % (g, u))
    rows, n_eq, worst = [], 0, 0.0
    for k in sorted(set(gi) & set(ui)):
        with safe_open(str(gi[k]), framework="pt") as h:
            tg = h.get_tensor(k)
        with safe_open(str(ui[k]), framework="pt") as h:
            tu = h.get_tensor(k)
        if tg.shape != tu.shape:
            rows.append(dict(name=k, shape_google=list(tg.shape), shape_unsloth=list(tu.shape), equal=False))
            continue
        cast = tg.to(torch.bfloat16)
        eq = bool(torch.equal(cast, tu.to(torch.bfloat16)))
        d = float((cast.float() - tu.float()).abs().max()) if not eq else 0.0
        n_eq += eq
        worst = max(worst, d)
        rows.append(dict(name=k, shape=list(tg.shape), dtype_google=str(tg.dtype), dtype_unsloth=str(tu.dtype),
                         equal=eq, max_abs_diff=d))
    only_g, only_u = sorted(set(gi) - set(ui)), sorted(set(ui) - set(gi))
    return dict(check="google fp32 -> bf16 vs unsloth bf16, tensor by tensor", google=str(g), unsloth=str(u),
                n_common=len(rows), n_bitwise_equal=n_eq, max_abs_diff=worst, only_google=only_g,
                only_unsloth=only_u, all_equal=(n_eq == len(rows) and not only_g and not only_u), tensors=rows)


def means(a) -> dict:
    import torch
    for p in (a.pod, a.local):
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    P, L = (torch.load(p, map_location="cpu", weights_only=False) for p in (a.pod, a.local))
    out = dict(check="caples_means: pod file vs local file", pod=a.pod, local=a.local,
               pod_label=P["meta"]["label"], local_label=L["meta"]["label"],
               mean_tokens_equal=bool(torch.equal(P["mean_tokens"], L["mean_tokens"])), layers={}, ld={})
    for l in P["theirs"]:
        if l not in L["theirs"]:
            out["layers"][str(l)] = "missing in local"
            continue
        r = {}
        for src in ("theirs", "exact"):
            for q in ("code", "err"):
                x, y = P[src][l][q].float(), L[src][l][q].float()
                r["%s_%s_max_abs" % (src, q)] = float((x - y).abs().max()) if x.shape == y.shape else "shape"
                r["%s_%s_scale" % (src, q)] = float(x.abs().max())
        out["layers"][str(l)] = r
    for k in P["ld"]:
        out["ld"][k] = {n: dict(pod=sum(v) / len(v), local=(sum(L["ld"][k][n]) / len(L["ld"][k][n]))
                                if n in L["ld"].get(k, {}) else None) for n, v in P["ld"][k].items()}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("weights", "means"))
    p.add_argument("--google", default=None, help="google snapshot folder (default: the pinned HF cache snapshot)")
    p.add_argument("--unsloth", default=None)
    p.add_argument("--pod", default=None)
    p.add_argument("--local", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    try:
        res = weights(a) if a.cmd == "weights" else means(a)
    except Exception as e:  # noqa: BLE001  (a missing input: report and exit 1)
        print("pod_checks_d %s: %s: %s" % (a.cmd, type(e).__name__, e))
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    brief = {k: v for k, v in res.items() if k not in ("tensors", "layers", "ld")}
    print(json.dumps(brief, default=str)[:1500])
    if a.cmd == "means":
        print(json.dumps(res["layers"], default=str)[:1500])
    print("written %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
