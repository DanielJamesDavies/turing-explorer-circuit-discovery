"""Exp 126 arm D (CPU): copies of fitted WCM circuits with every member's amplitude (alpha) clipped to [1/c, c].

No refit: the membership is the baseline fit's; only node.metadata["amplitude"] of the non-seed nodes changes (a node
without one is unweighted, alpha 1, and stays as it is). Read back through eval.wcm_eval.circuit_alphas, as
score_arena_a.wcm_units reads every WCM circuit.

Outputs: OUT/c<c>/main/circuits/<key>.pt (a score_arena_a --wcm dir; label it c<c>: a label that does not start with
"l" gets knob_value None in wcm_units, so the lambda parse is not involved), OUT/clip_stats.jsonl (one row per circuit
and c: n members, alpha quantiles before, n raised / lowered, alpha quantiles after). Existing outputs are kept.

  WSL, repo root:
  PYTHONPATH=src CUDA_VISIBLE_DEVICES= ARENA_A_SCOPE=a2 ./.venv/bin/python -B experiments/126-wcm-overshoot/make_clipped.py \
      --src experiments/126-wcm-overshoot/out/fit/s400/wcm/a2_l1e-3 --out experiments/126-wcm-overshoot/out/clip \
      [--c 1.5,2,3] [--keys k1,k2]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A = ROOT / "experiments" / "107-true-baselines" / "arena_a"
for p in (str(ROOT / "src"), str(A)):
    if p not in sys.path:
        sys.path.insert(0, p)


def quant(v):
    if not v:
        return None
    v = sorted(v)
    q = lambda f: v[min(len(v) - 1, int(f * len(v)))]  # noqa: E731
    return dict(min=v[0], p10=q(0.1), p50=q(0.5), p90=q(0.9), max=v[-1], mean=sum(v) / len(v))


def cname(c):
    return "c%g" % c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a WCM fit OUT (main/circuits/*.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--c", default="1.5,2,3")
    ap.add_argument("--keys", default="")
    a = ap.parse_args()
    import torch
    import wcm_tcz as WT
    WT.register()                                         # the circuits unpickle with the family's FeatureID
    from eval.wcm_eval import circuit_alphas, circuit_key, seed_of
    cs = [float(x) for x in a.c.split(",") if x]
    keys = set(k for k in a.keys.split(",") if k)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    srcs = sorted((Path(a.src) / "main" / "circuits").glob("*.pt"))
    if not srcs:
        raise SystemExit("no circuits in %s/main/circuits" % a.src)
    stats_path = out / "clip_stats.jsonl"
    done = set()
    if stats_path.exists():
        for line in open(stats_path):
            try:
                r = json.loads(line)
                done.add((r["key"], r["c"]))
            except ValueError:
                continue
    fh = open(stats_path, "a")
    n_w = 0
    for p in srcs:
        for c in cs:
            dst = out / cname(c) / "main" / "circuits" / p.name
            circ = torch.load(p, weights_only=False, map_location="cpu")
            if circ is None:
                continue
            key = circuit_key(circ)
            if keys and key not in keys:
                break
            if dst.exists() and (key, c) in done:
                continue
            seed = seed_of(circ)
            before, after, n_up, n_down, n_none = [], [], 0, 0, 0
            lo, hi = 1.0 / c, c
            for n in circ.nodes.values():
                if n is seed:
                    continue
                if "amplitude" not in n.metadata:
                    n_none += 1
                    continue
                v = float(n.metadata["amplitude"])
                w = min(hi, max(lo, v))
                n_up += w > v
                n_down += w < v
                before.append(v)
                after.append(w)
                n.metadata["amplitude"] = w
            al = circuit_alphas(circ)                    # the scorer's read-back
            got = sorted(x for d in al.values() for x in d.values())
            assert all(lo - 1e-9 <= x <= hi + 1e-9 for x in got) or n_none, (key, c)
            circ.metadata = dict(circ.metadata or {}, exp126_clip=dict(c=c, lo=lo, hi=hi, src=str(p)))
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                tmp = dst.with_suffix(".tmp")
                torch.save(circ, tmp)
                tmp.replace(dst)
                n_w += 1
            row = dict(key=key, c=c, label=cname(c), src=str(p), dst=str(dst), n_members=len(before) + n_none,
                       n_no_amplitude=n_none, n_raised=n_up, n_lowered=n_down, alpha_before=quant(before),
                       alpha_after=quant(after))
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print("%-12s c=%-4g n %4d raised %4d lowered %4d | alpha p90 %.2f max %.2f -> p90 %.2f max %.2f" % (
                key, c, row["n_members"], n_up, n_down, row["alpha_before"]["p90"], row["alpha_before"]["max"],
                row["alpha_after"]["p90"], row["alpha_after"]["max"]), flush=True)
    print("make_clipped: %d circuit files written under %s" % (n_w, out))


if __name__ == "__main__":
    main()
