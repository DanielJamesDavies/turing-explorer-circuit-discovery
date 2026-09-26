"""SPECIFICITY ON THE PROTOCOL-V1 CIRCUITS (059): do equal ablation-term weights change the amplifier rate?

Runs this directory's diagnostic (specificity.score) on 059's circuits for the protocol's training set, fitted with
the old weights (B, B2: gamma 0.25, lambda 1e-3) and the new ones (Bw, B2w: gamma 1, lambda 2e-3), each pair being
two stratified samples of the same 32 + 16 recipe. Contexts are 059's cache (held-out strongest; siblings from the
48 strongest-train contexts; close contrast set), injected as in 059, so the scores match 059's evaluation.

  PYTHONPATH=src python experiments/056-specificity/spec059.py
      -> results/v059/specificity.jsonl, results/v059/summary.md
  env: ARMS (B,B2,Bw,B2w)  ONLY (comma targets, smoke)
"""
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
EXP = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(EXP / "059-context-pool")); sys.path.insert(0, str(EXP / "049-circuit-graph"))
ARMS = [a for a in os.environ.get("ARMS", "B,B2,Bw,B2w").split(",") if a]
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]
OUTDIR = HERE / "results" / "v059"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TAG", "spec059_unused")
    os.environ["ARMS"] = ",".join(ARMS)
    import specificity as S
    S.RES = OUTDIR; S.OUT = OUTDIR / "specificity.jsonl"; S.ARMS = ARMS
    if os.environ.get("SUMMARY") == "1":
        S.summarise(); return
    import amp_eval_pass_v2 as V
    import protocol_harness as H
    G = V.setup()
    CleanTap, CircuitTap = S.make_patchers(G)
    ctx = torch.load(H.P.CTX, weights_only=False)
    tg = [x for x in (H.P.E055 / "seeds.txt").read_text().split() if x]
    tg = [x for x in tg if x in ONLY] if ONLY else tg
    done = set()
    if S.OUT.exists():
        done = {(r["arm"], r["seed"]) for r in map(json.loads, open(S.OUT)) if "pi" in r}
    with open(S.OUT, "a") as fh:
        for arm in ARMS:
            p = H.P.HERE / ("data_%s" % arm) / "discovered_circuits.shard0.pt"
            if not p.exists():
                print("arm %s: no circuits" % arm, flush=True); continue
            byk = {V.key_of(c): c for c in torch.load(p, weights_only=False, map_location="cpu").values()}
            for key in tg:
                if (arm, key) in done or key not in byk or key not in ctx:
                    continue
                H.patch_eval_contexts(G, ctx[key], "strong")
                try:
                    rows = S.score(G, byk[key], V, CleanTap, CircuitTap, arm)
                except Exception as e:  # noqa: BLE001
                    rows = [dict(arm=arm, seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))]
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                r0 = next((r for r in rows if r.get("pi") == "C"), rows[0])
                print("  %-4s %-15s %s" % (arm, key, r0.get("error") or "faith %s sib %s switched %s" % (
                    r0["target_faith_pre"], r0["sibling_faith_median"], r0["switched_on_circuit"])), flush=True)
    S.summarise()


if __name__ == "__main__":
    main()
