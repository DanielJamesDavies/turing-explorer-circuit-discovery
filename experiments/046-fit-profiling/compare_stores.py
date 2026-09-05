"""EQUIVALENCE CHECK between two circuit stores (same seeds, same config,
different engine code): membership Jaccard and amplitude max-abs-diff per
seed. The gate for any performance change: memberships identical (or
float-noise: Jaccard >= 0.98) and alphas within ~1e-2.

  python experiments/046-fit-profiling/compare_stores.py A.pt B.pt
"""
import sys

import torch

sys.path.insert(0, "src")   # stores pickle project classes


def load(path):
    cs = torch.load(path, weights_only=False)
    out = {}
    for c in cs.values():
        md = c.metadata
        key = (md.get("seed_comp"), md.get("seed_latent"), md.get("discovery_method"))
        mem = {}
        for n in c.nodes.values():
            if n.metadata.get("role") == "seed":
                continue
            f = n.metadata["feature_id"]
            mem[(f.layer, f.kind, f.index)] = n.metadata.get("amplitude")
        out[key] = mem
    return out


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(a) | set(b))
    worst_j, worst_amp = 1.0, 0.0
    for k in keys:
        ma, mb = a.get(k), b.get(k)
        if ma is None or mb is None:
            print("seed %s: only in %s" % (k, "A" if ma is not None else "B"))
            worst_j = 0.0
            continue
        inter = set(ma) & set(mb)
        j = len(inter) / max(len(set(ma) | set(mb)), 1)
        amp_diff = max((abs((ma[i] or 0) - (mb[i] or 0)) for i in inter),
                       default=0.0)
        worst_j, worst_amp = min(worst_j, j), max(worst_amp, amp_diff)
        print("seed c%s/%s: |A|=%d |B|=%d jaccard %.4f | max |dalpha| %.4f"
              % (k[0], k[1], len(ma), len(mb), j, amp_diff))
    verdict = "EQUIVALENT" if worst_j >= 0.98 and worst_amp <= 2e-2 else "DIFFERENT"
    print("%s | worst jaccard %.4f | worst |dalpha| %.4f | %d seeds"
          % (verdict, worst_j, worst_amp, len(keys)))
    sys.exit(0 if verdict == "EQUIVALENT" else 1)


if __name__ == "__main__":
    main()
