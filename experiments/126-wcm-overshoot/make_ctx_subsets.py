"""Exp 126 arm B (CPU): copies of the smoke context records with fewer TRAINING rows, everything else unchanged.

Per target and per size n (default 8, 16): rec["D_train"] -> a seeded random subset of n of its 32 strongest-train
indices, rec["B_mid"] -> n / 2 of its 16 mid-band indices (scaled with D_train: 32 + 16 at baseline). The subsets
are NESTED (one seeded permutation per target and pool, the first n taken; n8 is a subset of n16) and keep the
original index order. Unchanged: strong / mid pools (tokens, peaks, train / held splits), so the held-out set
(strong["held"]), the A-ablation means (strong["train"], 48), the contrast rows (neg) and the ho_xdoc reference are
identical to the baseline. score_arena_a's "home" set follows the record (prompts_for: the 8 highest-peak rows of
D_train), so on n8 home = all 8 training-strong rows.

Outputs, per size: OUT/targets_n<n>/ctx/<key>.pt ({key: rec}, the driver's CTX_DIR), targets.txt (the keys: run_pod.sh
make_lists checks AA_KEYS against it), SUBSETS.json (the indices kept, seed, sources). Never overwrites a record.

  WSL, repo root:
  PYTHONPATH=src CUDA_VISIBLE_DEVICES= ./.venv/bin/python -B experiments/126-wcm-overshoot/make_ctx_subsets.py \
      --src experiments/107-true-baselines/arena_a/out/targets_smoke --out experiments/126-wcm-overshoot/out \
      --keys 0.tc.1257,... [--sizes 8,16] [--seed 126] [--dry-run]
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))


def subset(idx, n, rng):
    """the first n of a seeded permutation of idx, in idx's own order"""
    idx = list(idx)
    if n >= len(idx):
        return idx
    perm = list(range(len(idx)))
    rng.shuffle(perm)
    keep = set(perm[:n])
    return [idx[j] for j in range(len(idx)) if j in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="make_targets output dir (ctx/, targets.txt)")
    ap.add_argument("--out", required=True, help="parent dir; writes targets_n<n>/ under it")
    ap.add_argument("--keys", required=True)
    ap.add_argument("--sizes", default="8,16")
    ap.add_argument("--seed", type=int, default=126)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import torch
    src = Path(a.src)
    keys = [k for k in a.keys.split(",") if k]
    sizes = [int(x) for x in a.sizes.split(",") if x]
    have = set((src / "targets.txt").read_text().split())
    miss = [k for k in keys if k not in have or not (src / "ctx" / ("%s.pt" % k)).exists()]
    if miss:
        raise SystemExit("not in %s (targets.txt / ctx): %s" % (src, miss))
    plan = {n: {} for n in sizes}
    from circuit.protocol_contexts import training_arm
    kept, skipped = [], []
    for key in keys:
        rec = torch.load(src / "ctx" / ("%s.pt" % key), weights_only=False, map_location="cpu")[key]
        if rec.get("strong") is None or training_arm(rec)[0] != "B":
            # arm A (no mid-band pool): the fit uses all 48 strong-train rows, which are also the A-means rows, so a
            # D_train subset would change nothing; such a target is left out of arm B
            skipped.append(key)
            print("%s: training arm A (no mid-band pool): left out of arm B" % key)
            continue
        kept.append(key)
        d, b = list(rec["D_train"]), list(rec.get("B_mid") or [])
        # one permutation per (target, pool): string seeds hash deterministically in random.Random (sha512)
        for n in sizes:
            dn = subset(d, n, random.Random("%d|%s|D_train" % (a.seed, key)))
            bn = subset(b, n // 2, random.Random("%d|%s|B_mid" % (a.seed, key))) if b else b
            assert set(dn) <= set(d) and set(bn) <= set(b) and len(dn) == min(n, len(d))
            plan[n][key] = dict(D_train=dn, B_mid=bn, n_D_train_src=len(d), n_B_mid_src=len(b))
            if a.dry_run:
                print("%s n%d: D_train %d -> %d %s | B_mid %d -> %d %s" % (key, n, len(d), len(dn), dn, len(b), len(bn), bn))
                continue
            od = Path(a.out) / ("targets_n%d" % n) / "ctx"
            od.mkdir(parents=True, exist_ok=True)
            p = od / ("%s.pt" % key)
            if p.exists():
                old = torch.load(p, weights_only=False, map_location="cpu")[key]
                if list(old["D_train"]) != dn or list(old.get("B_mid") or []) != bn:
                    raise SystemExit("%s exists with another subset (move it away)" % p)
                continue
            new = dict(rec)
            new["D_train"], new["B_mid"] = dn, bn
            new["exp126_subset"] = dict(n=n, seed=a.seed, D_train_src=d, B_mid_src=b, src=str(src / "ctx"))
            tmp = p.with_suffix(".tmp")
            torch.save({key: new}, tmp)
            tmp.replace(p)
    if a.dry_run:
        return
    for n in sizes:
        o = Path(a.out) / ("targets_n%d" % n)
        (o / "targets.txt").write_text("".join(k + "\n" for k in kept))
        (o / "SUBSETS.json").write_text(json.dumps(dict(
            n=n, n_mid=n // 2, seed=a.seed, src=str(src), rule="nested seeded subset of D_train (n) and B_mid (n/2); "
            "everything else in the record unchanged", skipped_arm_A=skipped, targets=plan[n]), indent=1))
        print("wrote %s: %d records (%d arm-A targets left out: %s)" % (o, len(kept), len(skipped), skipped))


if __name__ == "__main__":
    main()
