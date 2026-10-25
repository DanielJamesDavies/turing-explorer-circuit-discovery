"""079 — WCM with a fourth training ablation: donor-resample backgrounds (pilot; local RTX).

073 found production circuits pass the resample test 0.1% of the time (0.92 for resid targets only when members are
PINNED to natural values: closure, not node choice). This refits pilot targets with the production objective PLUS a
term that trains under real donor backgrounds (rs_engine.py), then scores every circuit both ways.

Arms (all protocol v1: triple floor gamma_C = gamma_A = 0.25, rank-keep 3e-3, lambda 1e-3, 400 steps, train-only means):
  prod     the 15k run's production circuit (fitted on H100; scored here, not refitted)
  W0       refit through the PATCHED engine with the resample weight 0 = production config on this GPU (the in-run
           baseline: separates the new term's effect from RTX-vs-H100 and refit noise)
  RS<w>    refit with resample weight w (WEIGHTS, default 0.25 and 1.0), donors = DONORS

Donors (training) per target, DONORS env:
  mix (default)  the 48 train contrast contexts (the C term's own contexts) + 48 random corpus contexts from a TRAINING
                 pool (seed 20261017) disjoint by corpus id from 073's evaluation pool
  contrast | random   one of the two
Evaluation per circuit, on held-out strongest:
  production scorer   driver.Runner.score (the 15k run's rows: Z / A / C / necessity / induce, pass rule)
  resample (073)      strict_eval.score_target with C donors = the 16 HELD-OUT contrast contexts (never a training
                      donor) and R donors = 073's evaluation pool (disjoint from the training pool); A donors = the
                      strongest-train contexts, as in 073. Plus 074's all-position metrics.

Targets: PER_CELL per band x kind (default 3, so 27) from 073's order (the first of 073's diag targets), or TARGETS;
CELL_OFFSET skips that many per cell first (the confirmation run: CELL_OFFSET=3 PER_CELL=11, 99 targets).

  PYTHONPATH=src python experiments/079-resample-training/fit_score.py
  env: PER_CELL (3)  TARGETS  LIMIT  WEIGHTS (0.25,1.0)  DONORS (mix)  ARMS (W0,RS)  OUT79  STEPS (400)
       REPORT=1 (summary; no GPU)
"""
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "073-strict-eval"))
sys.path.insert(0, str(HERE))
import strict_eval as SE  # noqa: E402  (imports 062's driver; OUT = the 15k run)
import rs_engine  # noqa: E402

PER_CELL = int(os.environ.get("PER_CELL", 3))
# skip the first CELL_OFFSET targets of every cell: CELL_OFFSET=3 gives targets disjoint from the pilot's 27
CELL_OFFSET = int(os.environ.get("CELL_OFFSET", 0))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
WEIGHTS = [float(x) for x in os.environ.get("WEIGHTS", "0.25,1.0").split(",") if x]
DONORS = os.environ.get("DONORS", "mix")
ARMS = [a for a in os.environ.get("ARMS", "W0,RS").split(",") if a]
OUT79 = Path(os.environ.get("OUT79", str(HERE / "out")))
# circuits already fitted under the same name elsewhere (e.g. the pilot's out_smoke4): loaded instead of refitted
REUSE_FROM = Path(os.environ["REUSE_FROM"]) if os.environ.get("REUSE_FROM") else None


def parse_arms(names):
    """Arm names -> [(name, resample weight, silent weight)]. W0 | RS<w> | RS<w>_S<v> | W0_S<v>; the bare name "RS"
    expands to RS<w> for w in WEIGHTS (the pilot's form)."""
    out = []
    for a in names:
        if a == "RS":
            out += [("RS%g" % w, w, 0.0) for w in WEIGHTS]
            continue
        head, _, tail = a.partition("_S")
        s = float(tail) if tail else 0.0
        if head == "W0":
            w = 0.0
        elif head.startswith("RS"):
            w = float(head[2:])
        else:
            raise ValueError("arm %r: expected W0, RS<w>, RS<w>_S<v> or W0_S<v>" % a)
        name = (head if head == "W0" else "RS%g" % w) + ("_S%g" % s if s else "")
        out.append((name, w, s))
    return out
STEPS = int(os.environ.get("STEPS", SE.driver.STEPS))
TRAIN_POOL_SEED = 20261017
N_RANDOM_DONORS = 48


def pick_targets():
    if os.environ.get("TARGETS"):
        keys = [t for t in Path(os.environ["TARGETS"]).read_text().split() if t]
    else:
        from collections import defaultdict
        cells = defaultdict(list)
        for k in SE.order_targets(SE.default_targets()):
            l, kd, _ = k.split(".")
            cells[(SE.band(int(l)), kd)].append(k)
        keys = [k for i in range(CELL_OFFSET, CELL_OFFSET + PER_CELL) for c in sorted(cells) for k in cells[c][i:i + 1]]
    return keys[:LIMIT] if LIMIT else keys


def train_pool(loader, exclude_ids):
    """N_RANDOM_DONORS x 4 random corpus contexts (>= 60 non-pad tokens), none in 073's evaluation pool."""
    from circuit.protocol_contexts import _load
    ranges = [(a, b) for a, b in loader.shard_id_ranges if a >= 0]
    rng = random.Random(TRAIN_POOL_SEED)
    ids = set()
    while len(ids) < N_RANDOM_DONORS * 6:
        a, b = rng.choice(ranges)
        i = rng.randint(a, b)
        if i not in exclude_ids:
            ids.add(i)
    pos, _, got = _load(loader, sorted(ids))
    keep = [j for j in range(pos.shape[0]) if int((pos[j] != 0).sum()) >= 60 and got[j] not in exclude_ids]
    return pos[keep], [got[j] for j in keep]


def donor_tokens(rec, pool, key):
    from circuit.protocol_contexts import eval_probe
    from eval.wcm_eval import split_n
    nt = rec["neg"][:64]
    nt_tr = nt[:split_n(int(nt.shape[0]))]
    g = torch.Generator().manual_seed(TRAIN_POOL_SEED + (sum(map(ord, key)) % 9973))
    rnd = pool[torch.randperm(pool.shape[0], generator=g)[:N_RANDOM_DONORS]]
    if DONORS == "contrast":
        return nt_tr
    if DONORS == "random":
        return rnd
    return torch.cat([nt_tr, rnd.to(nt_tr.device)])


def main():
    OUT79.mkdir(parents=True, exist_ok=True)
    mod = rs_engine.install()
    R = SE.driver.Runner()
    G = dict(R.G)
    G["R"] = R
    inference, bank, dev = G["inference"], G["bank"], G["device"]
    from eval.ablation_faithfulness import upstream_sites

    ev_pool_meta = torch.load(SE.OUT73 / "random_pool.pt")
    eval_pool = ev_pool_meta["tokens"].to(dev)
    tp_path = OUT79 / "train_pool.pt"
    if tp_path.exists():
        tp = torch.load(tp_path)
    else:
        toks, ids = train_pool(G["M0"].probe_builder.loader, set(int(i) for i in ev_pool_meta["ids"]))
        tp = dict(tokens=toks, ids=ids, seed=TRAIN_POOL_SEED, excluded=len(ev_pool_meta["ids"]))
        torch.save(tp, tp_path)
    tpool = tp["tokens"]
    assert not set(tp["ids"]) & set(int(i) for i in ev_pool_meta["ids"]), "train pool overlaps 073's eval pool"

    arms = parse_arms(ARMS)
    keys = pick_targets()
    done = set()
    rows_path = OUT79 / "rows.jsonl"
    if rows_path.exists():
        for line in open(rows_path):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "error" not in r:
                done.add((r["seed"], r["arm"]))
    print("079: %d targets x arms %s (+ prod), donors %s, steps %d; %d rows done; reuse %s" % (
        len(keys), [a for a, _, _ in arms], DONORS, STEPS, len(done), REUSE_FROM), flush=True)
    reuse_rows = {}
    if REUSE_FROM is not None and (REUSE_FROM / "rows.jsonl").exists():
        for line in open(REUSE_FROM / "rows.jsonl"):
            r = json.loads(line)
            if "error" not in r:
                reuse_rows[(r["seed"], r["arm"])] = r
    fh, ftr = open(rows_path, "a"), open(OUT79 / "train.jsonl", "a")
    SE.RAND_DIR = OUT79 / "_no_random_null"       # random nulls were scored in 073; skip them here
    for n_, key in enumerate(keys):
        rec = R.contexts(key)
        l, kd, _ = key.split(".")
        sites = sorted(upstream_sites(bank, int(l), kd))
        for arm, w, sw in [("prod", None, None)] + arms:
            if (key, arm) in done:
                continue
            if (key, arm) in reuse_rows and not (OUT79 / "circuits" / arm / ("%s.pt" % key)).exists():
                r = dict(reuse_rows[(key, arm)], reused_from=str(REUSE_FROM))   # same circuit, same deterministic eval
                fh.write(json.dumps(r) + "\n"); fh.flush()
                continue
            t0 = time.time()
            row = dict(seed=key, arm=arm, weight=w, silent_weight=sw, donors=DONORS if w else None,
                       band=SE.band(int(l)), kind=kd)
            try:
                if arm == "prod":
                    cdir = SE.driver.OUT / "main" / "circuits"
                else:
                    cdir = OUT79 / "circuits" / arm
                    path = cdir / ("%s.pt" % key)
                    reuse = REUSE_FROM / "circuits" / arm / ("%s.pt" % key) if REUSE_FROM else None
                    if not path.exists() and reuse is not None and reuse.exists():
                        cdir = reuse.parent
                        row["reused_from"] = str(REUSE_FROM)
                    elif not path.exists():
                        mod.RESAMPLE["weight"] = float(w)
                        if sw:
                            from eval.wcm_eval import site_encoder
                            We, be = site_encoder(bank, int(l), kd)
                            ti = int(key.split(".")[2])
                            sfn = rs_engine.make_silent_fn(inference, bank, (int(l), kd), ti, We[ti], be[ti], sw)
                            mod.SILENT.update(weight=float(sw), fn=sfn)
                        if w > 0:
                            toks = donor_tokens(rec, tpool, key).to(dev)
                            codes = SE.capture_codes(inference, bank, toks, sites)
                            codes = {s: (a.cpu(), i.cpu()) for s, (a, i) in codes.items()}
                            floors = rs_engine.DonorFloors(codes, bank.d_sae, dev, seed=TRAIN_POOL_SEED)
                            mod.RESAMPLE["make"] = rs_engine.make_factory(mod, floors)
                            row["n_donors"] = int(toks.shape[0])
                        M = R.method(SE.driver.GAMMA, SE.driver.LAM, True, steps=STEPS)
                        if os.environ.get("RS_MICRO"):
                            # the donor term keeps a dense donor floor per site for its backward (~2x one term's
                            # memory): a smaller micro-batch with gradient accumulation (effective batch unchanged)
                            from config import config
                            config.discovery.learned_mask.deep_site_threshold = 0
                            config.discovery.learned_mask.deep_batch_size = int(os.environ["RS_MICRO"])
                        train_arm, _ = R.train_arm(rec)
                        c, t_fit = R.fit(M, rec, key, train_arm, path, train_fh=ftr,
                                         label=dict(arm079=arm, weight=w, silent_weight=sw, donors=DONORS))
                        row["t_fit"] = round(t_fit, 1)
                        if w > 0:
                            row["donor_draws"] = floors.n_draws
                        if sw:
                            row["silent_calls"] = dict(sfn.stats)
                        mod.RESAMPLE.update(weight=0.0, make=None)
                        mod.SILENT.update(weight=0.0, fn=None)
                c = torch.load(cdir / ("%s.pt" % key), weights_only=False)
                if c is None:
                    row["skip"] = "rejected"
                else:
                    row["n"] = SE.driver.n_nodes(c)
                    prow = R.score(c, rec, "strong")
                    row["prod_scorer"] = {k: prow.get(k) for k in (
                        "free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "phi_cf_alpha_blind_tk",
                        "vacuous_tk", "a_pos_tk", "alpha_median", "alpha_p90", "n_members")}
                    SE.MAIN_DIR = cdir
                    srow = SE.score_target(G, key, eval_pool, c_held=True)
                    row["resample"] = {k: v for k, v in srow.items() if k.startswith(("rs_", "ap_")) and
                                       not k.endswith("_draws")}
            except Exception as e:  # noqa: BLE001
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                traceback.print_exc()
                mod.RESAMPLE.update(weight=0.0, make=None)
                mod.SILENT.update(weight=0.0, fn=None)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row["secs"] = round(time.time() - t0, 1)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            ps, rs = row.get("prod_scorer") or {}, row.get("resample") or {}
            print("[%s] %3d/%d %-16s %-7s %s n=%s Z %s A %s C %s nec %s | rsC %s rsR %s | fit %ss total %.0fs" % (
                time.strftime("%H:%M"), n_ + 1, len(keys), key, arm, row.get("error", row.get("skip", "ok")),
                row.get("n"), *["%.2f" % ps[k] if ps.get(k) is not None else "-" for k in
                                ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk")],
                *["%.2f" % rs[k] if rs.get(k) is not None else "-" for k in ("rs_C_wcm", "rs_R_wcm")],
                row.get("t_fit", "-"), row["secs"]), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    if os.environ.get("REPORT"):
        import report
        report.main()
    else:
        main()
