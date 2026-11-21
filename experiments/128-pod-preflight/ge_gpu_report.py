"""Exp 128 ge_gpu: Ge et al. (run_ge.py, unchanged) on the local GPU (~/venvs/ge107gpu, torch 2.8.0+cu128) against the
2026-11-05 CPU smoke (arena A out/ge_smoke_up: ge107 torch 2.3.1+cpu), same 3 targets x 2 prompts x 30 tau.

  python -I ge_gpu_report.py --gpu <run_ge out> --cpu experiments/107-true-baselines/arena_a/out/ge_smoke_up --out ge_gpu.md

Per circuit (target, prompt): seconds GPU / CPU (status.jsonl "seconds", one circuit = 30 tau hierarchical + post-hoc +
A11 re-runs), a_t, G1 gap, the hierarchical and post-hoc sizes over the tau grid (equal, or the max |difference| and
how many tau differ), and from the circuit files the node-set mismatches per tau and the max attribution difference
(as check_ge_smoke.py). Floating-point order differs on the GPU, so a node at a threshold edge may flip.
"""
import argparse
import json
from pathlib import Path

import torch


def status(d):
    out = {}
    p = Path(d) / "status.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ok":
                out[(r["key"], r["j"])] = r
    return out


def files_cmp(a, b):
    fa, fb = Path(a), Path(b)
    if not fa.exists() or not fb.exists():
        return None
    x, y = torch.load(fa, weights_only=False), torch.load(fb, weights_only=False)
    mism, attr = 0, 0.0
    for hx, hy in zip(x["hier"], y["hier"]):
        if not torch.equal(hx["nodes"], hy["nodes"]):
            mism += 1
            continue
        if hx["attr"].numel():
            attr = max(attr, float((hx["attr"].float() - hy["attr"].float()).abs().max()))
    return dict(set_mismatch_tau=mism, n_tau=len(x["hier"]), max_attr_diff=attr)


def sizes(a, b):
    if a is None or b is None:
        return "-"
    if a == b:
        return "equal"
    d = [abs(p - q) for p, q in zip(a, b)]
    return "%d of %d tau differ, max |dn| %d" % (sum(1 for v in d if v), len(d), max(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G, C = status(a.gpu), status(a.cpu)
    L = ["# Exp 128 ge_gpu: Ge on the RTX 5070 Ti (torch 2.8.0+cu128) vs the CPU smoke (torch 2.3.1+cpu)", "",
         "| circuit | GPU s | CPU s | speed-up | a_t GPU / CPU | G1 gap GPU | n_hier tau0..: GPU vs CPU | n_posthoc | node sets (files) |",
         "|---|---|---|---|---|---|---|---|---|"]
    tg = tc = 0.0
    for k in sorted(set(G) | set(C)):
        g, c = G.get(k), C.get(k)
        fc = files_cmp(Path(a.gpu) / "circuits" / k[0] / ("p%d.pt" % k[1]), Path(a.cpu) / "circuits" / k[0] / ("p%d.pt" % k[1]))
        if g and c:
            tg += g["seconds"]
            tc += c["seconds"]
        L.append("| %s p%d | %s | %s | %s | %s / %s | %s | %s (GPU n %s) | %s | %s |" % (
            k[0], k[1], "%.1f" % g["seconds"] if g else "-", "%.1f" % c["seconds"] if c else "-",
            "%.1fx" % (c["seconds"] / g["seconds"]) if g and c and g["seconds"] else "-",
            "%.6g" % g["a_t"] if g else "-", "%.6g" % c["a_t"] if c else "-", "%.2g" % g["g1_rel_gap"] if g else "-",
            sizes(g and g.get("n_hier"), c and c.get("n_hier")), (g or {}).get("n_hier", ["-"])[0],
            sizes(g and g.get("n_posthoc"), c and c.get("n_posthoc")),
            "-" if fc is None else "%d of %d tau differ, max attr diff %.2g" % (fc["set_mismatch_tau"], fc["n_tau"], fc["max_attr_diff"])))
    L += ["", "Total over circuits on both: GPU %.0f s, CPU %.0f s (%s). Per circuit = 30 tau (hierarchical + post-hoc + "
          "A11 re-runs); start-up (model + 24 dictionaries) is not in these seconds: see the job log's wall time." % (
              tg, tc, "%.1fx" % (tc / tg) if tg else "-")]
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
