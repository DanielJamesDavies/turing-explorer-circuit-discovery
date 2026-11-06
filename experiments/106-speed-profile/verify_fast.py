"""106: the T2 record verify (094 build_records.verify_one, CPU part) with a cached 8-gram predicate.

verify_one calls 086 rebuild._c8_vs(h, O) ~28k times per record; each call rebuilds the 8-gram set of every row of O
from numpy slices in pure Python. The prototype memoises each row's 8-gram list / set (keyed by the row's bytes) and
builds them from Python ints; the predicate (the share of h's 8-gram positions found in o's 8-gram set) and its
arithmetic are unchanged, so the failure lists must be identical. CPU only (FIRE=0: the GPU firing re-check is not
touched).

  PYTHONPATH=src python experiments/106-speed-profile/verify_fast.py
  env: REC_DIR (default 094/out/smoke10/ctx_v2)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "094-tl-v2-records"))
sys.path.insert(0, str(REPO / "experiments" / "086-heldout-rebuild"))
os.environ.setdefault("MODE", "verify")
os.environ["FIRE"] = "0"
REC_DIR = Path(os.environ.get("REC_DIR", str(REPO / "experiments" / "094-tl-v2-records" / "out" / "smoke10" / "ctx_v2")))
OUT = HERE / "out" / "verify_fast.json"

import torch  # noqa: E402
import build_records as BR  # noqa: E402
import rebuild as RB  # noqa: E402

_ORIG_C8 = RB._c8_vs
_G, _S = {}, {}


def _grams(row):
    k = row.tobytes()
    g = _G.get(k)
    if g is None:
        r = row.tolist()
        g = [tuple(r[j:j + 8]) for j in range(len(r) - 7)]
        _G[k] = g
        _S[k] = set(g)
    return k, g


def c8_fast(h, O):
    _, g = _grams(np.ascontiguousarray(h))
    out = []
    for o in O:
        k, _ = _grams(np.ascontiguousarray(o))
        s = _S[k]
        out.append(sum(x in s for x in g) / len(g))
    return np.asarray(out)


_CHK = {"calls": 0, "mismatch": 0}


def c8_check(h, O):
    """Both predicates on every call; counts calls whose outputs are not exactly equal."""
    a, b = _ORIG_C8(h, O), c8_fast(h, O)
    _CHK["calls"] += 1
    _CHK["mismatch"] += int(not np.array_equal(a, b))
    return a


def main():
    from config import config
    cfg = config.discovery.context_v1
    corpus = RB.Corpus(Path(os.environ.get("CORPUS_DIR", str(REPO / "experiments" / "085-heldout-audit" / "out"))))
    res = {}
    for p in sorted(REC_DIR.glob("*.pt")):
        key = p.stem
        rec = torch.load(p, weights_only=False)[key]
        row = {}
        for name, fn in (("orig", _ORIG_C8), ("fast", c8_fast)):
            RB._c8_vs = fn
            _G.clear()
            _S.clear()
            t = time.perf_counter()
            f = BR.verify_one(key, rec, corpus, None, cfg, warn=[])
            row[name] = dict(s=round(time.perf_counter() - t, 3), failures=f)
        RB._c8_vs = c8_check                                   # every call's output compared exactly
        _G.clear()
        _S.clear()
        n0, m0 = _CHK["calls"], _CHK["mismatch"]
        BR.verify_one(key, rec, corpus, None, cfg, warn=[])
        row["c8_calls_checked"], row["c8_calls_mismatched"] = _CHK["calls"] - n0, _CHK["mismatch"] - m0
        RB._c8_vs = _ORIG_C8
        row["identical"] = row["orig"]["failures"] == row["fast"]["failures"] and row["c8_calls_mismatched"] == 0
        row["speedup"] = round(row["orig"]["s"] / max(row["fast"]["s"], 1e-9), 2)
        res[key] = row
        print("%-16s orig %.2fs fast %.2fs x%.1f identical %s (%d failures)" % (
            key, row["orig"]["s"], row["fast"]["s"], row["speedup"], row["identical"], len(row["orig"]["failures"])),
            flush=True)
    tot_o = sum(r["orig"]["s"] for r in res.values())
    tot_f = sum(r["fast"]["s"] for r in res.values())
    summ = dict(n=len(res), orig_s_per_record=round(tot_o / max(len(res), 1), 3),
                fast_s_per_record=round(tot_f / max(len(res), 1), 3), speedup=round(tot_o / max(tot_f, 1e-9), 2),
                all_identical=all(r["identical"] for r in res.values()),
                c8_calls_checked=sum(r["c8_calls_checked"] for r in res.values()),
                c8_calls_mismatched=sum(r["c8_calls_mismatched"] for r in res.values()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(summary=summ, per_record=res), indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
