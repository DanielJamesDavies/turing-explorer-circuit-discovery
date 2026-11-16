"""116 §2: case-study checks on the exp-110 production circuits (DAN-214 §2). TuringLLM, WCM v4, fresh records.

Targets and groups are exp 110's, read from 110 out/case/<key>.json (the groups as restricted to the members present in
the v4 circuit, and the story list); contribution ranks from 110 out/members/<key>.csv (063's recipe).

For each tested latent set S (a story group, the union of the story groups, ...):

  1. SUBSET, PRODUCTION COEFFICIENTS (no refit). The production scorer (062 Runner.score -> 049 score_circuit,
     held-out strongest, activation read) with override_alphas = S at their fitted alpha: Z / A / C of S alone.
  2. KNOCK-OUT IN THE UNMODIFIED MODEL. From the same scorer row: necessity's live-stream run with only S zeroed,
     phi_sup_blind_raw_tk = the target's mean activation on the held-out strongest contexts with S set to 0 in the
     full, unablated model (every other latent and every SAE error live, everything downstream recomputed).
     drop = 1 - a_KO / a_pos (a_pos = the clean mean, same contexts, same read).
  3. SUFFICIENCY, COEFFICIENTS REFITTED ON S. The production fit (WCM v4, driver.Runner.fit) restricted to S, with no
     sparsity penalty and gates fully open (062 random_null.install_support_wrapper, the stable-core recipe), then
     scored like any circuit: Z / A / C / necessity.
  Controls, per S:
     - rank-matched random sets (063 recipe: for each member of S one non-named, non-generic member within +-RANK_WIN
       contribution ranks; N_DRAWS draws for 1-2, N_REFIT of them also refitted for 3),
     - the |S| largest contributors of the circuit (1-3).
  Reference rows: the full circuit (production coefficients; its necessity = all members knocked out together).

  wsl: PYTHONPATH=src .venv/bin/python -X utf8 experiments/116-error-ablated-and-case-checks/case_checks.py
  env: KEYS (9.resid.13945,9.resid.20419)  N_DRAWS (8)  N_REFIT (3)  RANK_WIN (15)  SKIP_REFIT=1
       REFIT_SETS ("plasma confinement|all story groups": the sets that get step 3)
  -> out/case/rows.jsonl (one row per (key, set, variant, draw); resumable), out/case/circuits/...
"""
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments"
ARCH = EXP / "109-pod-run-analysis" / "out" / "archives"
PROD_OUT = ARCH / "pub-turingllm-T3-main/root/turing-tl/experiments/094-tl-v2-records/out_v2"
CTX = ARCH / "pub-turingllm-T2-records/root/turing-tl/experiments/082-public-runs/out_turingllm/ctx_fresh"
E110 = EXP / "110-case-studies-v4" / "out"
OUTC = HERE / "out" / "case"
OUTC.mkdir(parents=True, exist_ok=True)
os.environ.update(MODE="main", WCM_VERSION="v4", OUT=str(OUTC / "rs"), CTX_DIR=str(CTX), SKIP_SPEC="1")
os.environ.setdefault("TAG", "case116_unused")
for p in (str(ROOT / "src"), str(EXP / "062-h100-protocol-v1"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import driver  # noqa: E402
import random_null as RN  # noqa: E402

KEYS = [k for k in os.environ.get("KEYS", "9.resid.13945,9.resid.20419").split(",") if k]
N_DRAWS = int(os.environ.get("N_DRAWS", 8))
N_REFIT = int(os.environ.get("N_REFIT", 3))
RANK_WIN = int(os.environ.get("RANK_WIN", 15))
SKIP_REFIT = os.environ.get("SKIP_REFIT") == "1"
# the sets whose coefficients are refitted (the paper's claims: fusion's ten plasma-confinement latents and its 24
# physics-group latents; antiquity's 19 story latents); "|"-separated, every other set gets steps 1-2 only
REFIT_SETS = set(os.environ.get("REFIT_SETS", "plasma confinement|all story groups").split("|"))
# 063 validate.GENERIC: members that fire about equally on target and contrast in nearly every circuit
GENERIC = {"0.mlp.1349", "0.resid.6299", "0.attn.5663", "0.attn.712", "1.attn.13899", "1.attn.34495", "0.mlp.26380",
           "1.mlp.621", "1.attn.16130", "1.resid.39697", "5.attn.33630"}
COLS = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "phi_sup_blind_raw_tk", "a_pos_tk",
        "e0_tk", "eM_topk_tk", "eN_topk_tk", "vacuous_tk", "n"]


def parse(k):
    l, kd, i = k.split(".")
    return int(l), kd, int(i)


def alpha_map(c):
    out = defaultdict(dict)
    for n in c.nodes.values():
        md = n.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out[(int(f.layer), str(f.kind))][int(f.index)] = float(md.get("amplitude", 1.0))
    return dict(out)


def only(alphas, keys):
    keep = {parse(k) for k in keys}
    out = {s: {i: a for i, a in d.items() if (s[0], s[1], i) in keep} for s, d in alphas.items()}
    return {s: d for s, d in out.items() if d}


def sets_for(key):
    spec = json.loads((E110 / "case" / ("%s.json" % key)).read_text(encoding="utf-8"))
    groups = {g: v["members"] for g, v in spec["groups"].items() if not g.startswith("_")}
    story = spec["story"]
    sets = {g: groups[g] for g in story}
    sets["all story groups"] = [m for g in story for m in groups[g]]
    named = {m for ms in groups.values() for m in ms}
    return sets, named


def main():
    import pandas as pd
    import torch
    path = OUTC / "rows.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r:
                done.add((r["key"], r["set"], r["variant"], r["draw"]))
    for m in ("WCM_VERSION", "WCM_R"):
        (OUTC / "rs").mkdir(parents=True, exist_ok=True)
        (OUTC / "rs" / m).write_bytes((PROD_OUT / m).read_bytes())
    RN.install_support_wrapper()
    R = driver.Runner()
    M = None if SKIP_REFIT else R.method(driver.GAMMA, driver.LAM, True)
    fh, ftr = open(path, "a"), open(OUTC / "train.jsonl", "a")
    t0 = time.time()

    def emit(key, set_name, variant, draw, members, row, extra=None):
        out = dict(key=key, set=set_name, variant=variant, draw=draw, n_set=len(members), members=sorted(members))
        out.update({k: row.get(k) for k in COLS})
        if row.get("a_pos_tk") and row.get("phi_sup_blind_raw_tk") is not None:
            out["ko_drop"] = 1.0 - float(row["phi_sup_blind_raw_tk"]) / float(row["a_pos_tk"])
        out.update(extra or {})
        fh.write(json.dumps(out) + "\n"); fh.flush()
        print("  %-16s %-34s %-10s %2s n=%3d Z %.2f A %.2f C %.2f nec %s drop %s  (%.0fs)" % (
            key, set_name[:34], variant, draw, len(members),
            *(float("nan") if row.get(c) is None else float(row[c]) for c in COLS[:3]),
            "-" if row.get("phi_sup_blind_tk") is None else "%.2f" % row["phi_sup_blind_tk"],
            "-" if out.get("ko_drop") is None else "%.2f" % out["ko_drop"], time.time() - t0), flush=True)

    for key in KEYS:
        try:
            c = torch.load(PROD_OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
            rec = R.contexts(key)
            arm, _ = R.train_arm(rec)
            base = alpha_map(c)
            present = {"%d.%s.%d" % (s[0], s[1], i) for s, d in base.items() for i in d}
            mem = pd.read_csv(E110 / "members" / ("%s.csv" % key))
            assert set(mem.node) == present, "%s: members table does not match the circuit" % key
            rank = {n: r for r, n in enumerate(mem.node)}
            sets, named = sets_for(key)
            pool = [n for n in mem.node if n not in named and n not in GENERIC]
            if (key, "full", "prod", 0) not in done:
                emit(key, "full", "prod", 0, present, R.score(c, rec, "strong"))
            rng = random.Random("116-%s" % key)
            plan = []
            for sname, ms in sets.items():
                missing = [m for m in ms if m not in present]
                assert not missing, "%s %s: not in the circuit: %s" % (key, sname, missing)
                plan.append((sname, "named", 0, ms))
                plan.append((sname, "top", 0, list(mem.node[:len(ms)])))
                for d in range(N_DRAWS):
                    pick = []
                    for m in ms:
                        cand = [n for n in pool if abs(rank[n] - rank[m]) <= RANK_WIN and n not in pick]
                        if cand:
                            pick.append(rng.choice(cand))
                    plan.append((sname, "random", d, pick))
            for sname, kind, d, ms in plan:
                # 1 + 2: subset at production coefficients; knock-out of the subset in the unmodified model
                v = "%s_prod" % kind
                if (key, sname, v, d) not in done:
                    emit(key, sname, v, d, ms, R.score(c, rec, "strong", alphas=only(base, ms)))
                # 3: coefficients refitted on the subset alone
                v = "%s_refit" % kind
                if SKIP_REFIT or sname not in REFIT_SETS or (kind == "random" and d >= N_REFIT) \
                        or (key, sname, v, d) in done:
                    continue
                sup = defaultdict(list)
                for m in ms:
                    l, kd, i = parse(m)
                    sup[(l, kd)].append(i)
                RN._SUPPORT.clear(); RN._SUPPORT.update(sup)
                tag = "%s__%s__%d" % (sname.replace(" ", "_").replace("/", "-").replace(",", ""), kind, d)
                try:
                    cc, tf = R.fit(M, rec, key, arm, OUTC / "circuits" / key / ("%s.pt" % tag), train_fh=ftr,
                                   label=dict(stage="116", set=sname, variant=kind, draw=d))
                finally:
                    RN._SUPPORT.clear()
                if cc is None:
                    emit(key, sname, v, d, ms, {}, dict(skip="rejected"))
                    continue
                emit(key, sname, v, d, ms, R.score(cc, rec, "strong"),
                     dict(t_fit=round(tf, 1), n_refit=driver.n_nodes(cc),
                          alpha_refit=[[nd.metadata["feature_id"].layer, nd.metadata["feature_id"].kind,
                                        int(nd.metadata["feature_id"].index), float(nd.metadata.get("amplitude", 1.0))]
                                       for nd in cc.nodes.values() if nd.metadata.get("role") != "seed"]))
        except Exception as e:  # noqa: BLE001
            RN._SUPPORT.clear()
            fh.write(json.dumps(dict(key=key, set="error", variant="error", draw=-1,
                                     error="%s: %s" % (type(e).__name__, str(e)[:300]))) + "\n"); fh.flush()
            traceback.print_exc()
    print("DONE %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
