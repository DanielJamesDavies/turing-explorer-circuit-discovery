"""133 runner (GPU): IOI WCM sweeps on GPT-2 small + OAI v5 32k SAEs, scored with 081's unchanged scorer.

Each config (a JSON list in CONFIGS) is either
  {"name", "rescore": "<circuit json path>"}            score an existing circuit (081 out_v2 / out_v4)
  {"name", "arm": W|U|W_CF, "lam", "lr", "wd", "steps", "sched": geometric|constant, "warmup", "R": gamma_R,
   "pair": bool, "off_w", "off_mode": hinge|sq, "sites": all|attn_mlp}   fit then score
Defaults = 081 v4 (lr 0.1575 peak, geometric, warm-up 0.1, wd 0.05, R 1.0 random donors, 400 steps, all 48 sites).

Scoring (081 engine.score = eval.wcm_eval.score_task, coefficients applied, A / C = the whole training / contrast pool
means; engine.score_cf = the minimal-pair read) on two held-out sets:
  dis  the first N_DIS of 081's held-out items (names / places / objects from the DISJOINT list halves; = 081's rule)
  sn   N_SN fresh items from the TRAINING list halves (make_sn.py; same names, no training item)
The scorer's ablation scope = the config's site set (attn_mlp: resid / resid_mid sites are never edited).

  CONFIGS=experiments/133-ioi-wcm-sweep/configs/<job>.json OUT=experiments/133-ioi-wcm-sweep/out \
  MODEL=gpt2-small WCM_VERSION=v4 PYTHONPATH=src .venv/bin/python -X utf8 -u experiments/133-ioi-wcm-sweep/run133.py
  env: N_DIS (1000) N_SN (998) STOP_EPOCH (no new config after it) EST_MIN (minutes a config needs; with STOP_EPOCH)
       EVAL_BS (64)
Rows: OUT/results.jsonl (one per config; configs already there are skipped), circuits OUT/circuits/<name>.json.
"""
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
E81 = HERE.parent / "081-task-circuits"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(E81))
os.environ.setdefault("WCM_VERSION", "v4")
os.environ.pop("HELDOUT_N", None)
import task_common as C  # noqa: E402

OUT = Path(os.environ.get("OUT", str(HERE / "out")))
CONFIGS = Path(os.environ["CONFIGS"])
N_DIS = int(os.environ.get("N_DIS", 1000))
N_SN = int(os.environ.get("N_SN", 998))
STOP_EPOCH = float(os.environ["STOP_EPOCH"]) if os.environ.get("STOP_EPOCH") else None
EST_MIN = float(os.environ.get("EST_MIN", 8))
DEF = dict(arm="W", lr=0.1575, wd=0.05, steps=400, sched="geometric", warmup=0.1, R=1.0, pair=False, off_w=0.0,
           off_mode="hinge", sites="all")
FIELDS = ("free0", "freeM_topk", "freeN_topk", "nec", "acc_free0", "pass", "free0_a1", "full", "e0")


def log(s):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), s), flush=True)


def site_set(all_sites, which):
    if which == "all":
        return list(all_sites)
    if which == "attn_mlp":
        return [s for s in all_sites if s[1] in ("attn", "mlp")]
    raise ValueError(which)


def pick(sc, mp, pre):
    out = {}
    for f in FIELDS:
        out["%s_%s" % (pre, f)] = sc.get("task_%s" % f)
    if mp:
        for f in ("free0", "freeM_topk", "freeN_topk"):
            out["%s_mp_%s" % (pre, f)] = mp.get("mp_task_%s" % f)
        out["%s_mp_d_full" % pre] = mp.get("mp_task_d_full")
        out["%s_mp_d_free0" % pre] = mp.get("mp_task_d_free0")
        out["%s_mp_cf_acc_free0" % pre] = mp.get("mp_task_cf_acc_free0")
    return out


def main():
    import torch
    import engine as E
    import engine133 as P
    from eval import wcm_eval as W
    mod = P.install()
    configs = json.loads(CONFIGS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "circuits").mkdir(exist_ok=True)
    p_res = OUT / "results.jsonl"
    done = set()
    if p_res.exists():
        for line in open(p_res):
            r = json.loads(line)
            if "error" not in r:
                done.add(r["name"])
    todo = [c for c in configs if c["name"] not in done]
    log("%d configs, %d to do | out %s | N_DIS %d N_SN %d" % (len(configs), len(todo), OUT, N_DIS, N_SN))
    if not todo:
        return
    eng = E.Engine(C.MODEL)
    ALL = list(eng.sites)
    T = json.loads((HERE / "tasks" / "ioi_sn.json").read_text())
    tr = list(T["split"]["train"])
    v_dis = dict(T, split=dict(train=tr, held=list(T["split"]["held"])[:N_DIS]))
    v_sn = dict(T, split=dict(train=tr, held=list(T["split"]["held_sn"])[:N_SN]))
    B_dis = eng.batch(v_dis)
    B_sn = eng.batch(v_sn)
    N_TR = int(os.environ.get("N_TR", 0))          # >0: also score the first N_TR TRAINING items (in-sample, "tr")
    B_tr = eng.batch(dict(T, split=dict(train=tr, held=tr[:N_TR]))) if N_TR else None
    t0 = time.time()
    eng.means(B_dis)
    B_sn["means"] = B_dis["means"]
    if B_tr is not None:
        B_tr["means"] = B_dis["means"]
    n_tr = B_dis["n_train"]
    assert torch.equal(B_dis["tokens"][:n_tr], B_sn["tokens"][:n_tr])
    ct, ca = B_dis["cf_all"]
    g_tr, b_tr = B_dis["good"][:n_tr].cpu(), B_dis["bad"][:n_tr].cpu()

    def full_D(tok, anc):                                                      # the full model's D per training row
        lp, _, _ = W.read_rows(eng.inf, None, tok, anc, None, E.EVAL_BS)
        return W.task_metric(lp, B_dis["metric"]["task"], g_tr, b_tr).float().cpu()

    D_cf = full_D(ct[:n_tr], ca[:n_tr])                                        # on each training item's ABC prompt
    D_cl = full_D(B_dis["tokens"][:n_tr], B_dis["anchors"][:n_tr])
    log("means + full-model train D: clean %.3f, ABC %.3f (frac ABC > 0 %.2f) (%.0fs)"
        % (float(D_cl.mean()), float(D_cf.mean()), float((D_cf > 0).float().mean()), time.time() - t0))
    fh = open(p_res, "a")
    for cfg0 in todo:
        if STOP_EPOCH is not None and time.time() + 60 * EST_MIN > STOP_EPOCH:
            log("STOP: %s would end after STOP_EPOCH" % cfg0["name"])
            break
        cfg = dict(DEF, **cfg0)
        row = dict(cfg)
        t_c = time.time()
        try:
            eng.sites = site_set(ALL, cfg["sites"])
            if cfg.get("rescore"):
                d = json.loads(Path(cfg["rescore"]).read_text())
                alphas = C.alphas_from_json(d["alphas"])
                alphas = {s: v for s, v in alphas.items() if s in set(eng.sites)}
                row.update(fit_src=cfg["rescore"], lam=d.get("lam"), arm=d.get("arm"))
            else:
                a = C.ARMS[cfg["arm"]]
                cf_train = bool(a.get("cf_train", False))
                if cf_train and cfg["pair"]:
                    raise ValueError("pair with W_CF is not defined")
                support = None
                if cfg["arm"] == "RND":
                    # 081's random null: support = random latents active on the training prompts, matched to the
                    # reference circuit's per-site counts; fitted coefficients, gates open, no penalty
                    import random as _random
                    ref = C.alphas_from_json(json.loads(Path(cfg["ref"]).read_text())["alphas"])
                    if "live" not in B_dis:
                        B_dis["live"] = eng.active_latents(B_dis)
                    sup, short = C.draw_support(C.site_counts(ref), B_dis["live"],
                                                _random.Random("133-%s" % cfg["name"]))
                    support = sup
                    row["rnd_short_sites"] = short
                kw = eng.engine_kwargs(B_dis, float(cfg["lam"]), bool(a["weighted"]), int(cfg["steps"]), 0.05, "gap",
                                       support, True, cf_train)
                kw.update(lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]), lr_schedule=cfg["sched"],
                          warmup_frac=float(cfg["warmup"]), steps=int(cfg["steps"]))
                bs = int(kw["batch_size"])
                _, _, _, idx = eng.presented_rows(B_dis, int(cfg["steps"]), bs, cf_train)
                if float(cfg["R"]) > 0:
                    kw["resample_weight"] = float(cfg["R"])
                    if cfg["pair"]:
                        kw["resample_tokens"] = ct[idx]
                        assert kw["resample_tokens"].shape == kw["pos_tokens"].shape
                else:
                    kw["resample_weight"] = 0.0
                    kw.pop("resample_tokens", None)
                    kw.pop("resample_k", None)
                off_data = None
                if float(cfg["off_w"]) > 0:
                    if cf_train:
                        raise ValueError("off term with W_CF is not defined here")
                    off_data = dict(tokens=ct[idx], anchors=ca[idx], good=B_dis["good"][idx, 0],
                                    bad=B_dis["bad"][idx, 0], target=D_cf[idx.cpu()].to(ct.device))
                P.configure(pair=bool(cfg["pair"]), off_weight=float(cfg["off_w"]), off_mode=cfg["off_mode"],
                            off_data=off_data)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                tf = time.time()
                try:
                    scores, prov = mod.run_learned_mask(eng.inf, eng.bank, **kw)
                finally:
                    row["p133"] = P.stats_summary()
                    eng.reset()
                    P.configure()
                row["t_fit"] = round(time.time() - tf, 1)
                row["peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None
                if a["weighted"] and prov.get("amp_kept") is not None:
                    alphas = C.alphas_from_json(prov["amp_kept"])
                else:
                    alphas = defaultdict(dict)
                    for f in scores:
                        alphas[(int(f.layer), str(f.kind))][int(f.index)] = 1.0
                    alphas = dict(alphas)
                alphas = {s: v for s, v in alphas.items() if v}
                curve = prov.get("curve") or {}
                terms = curve.get("terms") or {}
                row["terms_first"] = {k: round(float(v[0]), 4) for k, v in terms.items() if v}
                row["terms_last40"] = {k: round(float(sum(v[-40:]) / len(v[-40:])), 4) for k, v in terms.items() if v}
                row["curve_n_every50"] = (curve.get("n_members") or [])[::50]
                for k in ("loss_initial", "loss_final", "logit_scale", "lr_sum", "lr_peak", "resample_n_donors",
                          "resample_draws", "weight_decay", "amp_stats", "mean_m_kept"):
                    row[k] = prov.get(k)
                h0, h1 = B_sn["n_train"], B_sn["n_train"] + N_SN
                row["soft"] = P.soft_eval(eng.inf, eng.bank, B_sn["tokens"][h0:h1], B_sn["anchors"][h0:h1],
                                          B_sn["good"][h0:h1, 0], B_sn["bad"][h0:h1, 0])
                (OUT / "circuits" / ("%s.json" % cfg["name"])).write_text(json.dumps(dict(
                    alphas=C.alphas_to_json(alphas), cfg=cfg, terms_last40=row["terms_last40"])) + "\n")
            row["n"] = sum(len(v) for v in alphas.values())
            row["n_by_kind"] = {k: sum(len(v) for s, v in alphas.items() if s[1] == k) for k in eng.kinds}
            ts = time.time()
            views = [("dis", B_dis, N_DIS), ("sn", B_sn, N_SN)] + ([("tr", B_tr, N_TR)] if B_tr is not None else [])
            for pre, B, nn in views:
                sc = eng.score(B, alphas)
                mp = eng.score_cf(B, alphas, nn, clean_row=sc)
                row.update(pick(sc, mp, pre))
            row["t_score"] = round(time.time() - ts, 1)
            sf = row.get("soft") or {}
            if sf and row.get("sn_full") is not None and row.get("sn_e0") is not None:
                den = row["sn_full"] - row["sn_e0"]
                for k in ("soft", "hard"):
                    if k in sf:
                        row["sn_Z_train_%s" % k] = (sf[k] - row["sn_e0"]) / den if abs(den) > 1e-9 else None
        except Exception as e:  # noqa: BLE001
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            eng.sites = ALL
        row["t_total"] = round(time.time() - t_c, 1)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        g = lambda k: ("%.2f" % row[k]) if isinstance(row.get(k), (int, float)) and not isinstance(row.get(k), bool) else "-"  # noqa: E731
        log("%-22s n=%-6s | dis Z %s A %s C %s nec %s acc %s mpZ %s pass %s | sn Z %s A %s C %s nec %s acc %s mpZ %s "
            "pass %s | tr Z %s | sn Z train-fwd soft %s hard %s | terms %s | %ss %s" % (
                cfg["name"], row.get("n"), g("dis_free0"), g("dis_freeM_topk"), g("dis_freeN_topk"), g("dis_nec"),
                g("dis_acc_free0"), g("dis_mp_free0"), row.get("dis_pass"), g("sn_free0"), g("sn_freeM_topk"),
                g("sn_freeN_topk"), g("sn_nec"), g("sn_acc_free0"), g("sn_mp_free0"), row.get("sn_pass"),
                g("tr_free0"), g("sn_Z_train_soft"), g("sn_Z_train_hard"), row.get("terms_last40"), row["t_total"], row.get("error") or ""))
    fh.close()
    log("DONE")


if __name__ == "__main__":
    main()
