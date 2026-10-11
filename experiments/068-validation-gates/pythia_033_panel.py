"""DAN-36 GATE: the 033 Pythia-70m panel re-fitted through the PRODUCTION ENGINE, scored by 033's own scorer.

033 (experiments/033-cross-sae-topk) is the verified standalone implementation: 30 analysed seeds (5 layers x 3
kinds x 2), triamp400 = triple floor + free amplitudes, 400 steps, lambda 1e-3; headline 22/30 ALL-PASS, fitted
random nulls 0/146. Pass rule (029/031/033 "ALL-PASS"): ampF0 and ampFM both in [0.8, 1.25] and sup > 0.9, held-out
16 contexts; seeds with held-out a_pos < 1.0 excluded. Per-seed rows: 033/rows.jsonl (arm "triamp400", nulls
"null0".."null4").

Contexts: 033's own windows. Six seeds (resid L2 / L4) were fitted from 033/scan.pt (run.log); the panel run
(run_panel.log, scan_panel.pt) skipped the four of them it shared as already done, so those four keep scan.pt's
negatives. The panel's 30 analysed seeds are therefore 6 from scan.pt + 24 from scan_panel.pt (two panel seeds,
mlp L2 #12294 and attn L4 #2654, fail the a_pos_ho >= 1.0 rule). Tokens: the 20,000 x 64 wikitext windows stored in
both scan files (identical). Positives: the 64 strongest windows (first 48 train, last 16 held out); negatives: 64
verified-silent windows (first 48 train / floors, last 16 held out).

Arms per seed (rows keyed (seed, arm, steps) in OUT/rows.jsonl; resumable):
  engine      _build_mode_method("ablation_gradient", "mask") -> discover() -> run_learned_mask, with 033's
              objective settings (see configure_033) and 033's contexts and anchors injected as in 062's driver.
              Scored by ref033.Seed.score, i.e. 033's scorer, on 033's held-out 16.
  standalone  033's own fit (ref033.Seed.fit, a device-agnostic transcription) on the same machine and contexts,
              same scorer. Separates "engine vs standalone" from "this machine vs 033's GPU (TF32)".
  standalone_notemp  (diagnostic, not 033) 033's fit with the gate priced as the engine prices it: L1 and the
              amplitude leak guard on sigmoid(theta) instead of sigmoid(theta / T). Isolates that one difference.
  nullD       a fitted-amplitude random null through the engine core (run_learned_mask with a fixed support, the
              gates frozen on (theta_init 40) and lambda 0, as 033's null), members drawn from 033's live pool at
              the ENGINE circuit's size with 033's rng (Random(1000 + latent)).

  wsl: PYTHONPATH=src CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TURING_NATIVE_CPU_ONLY=1 \\
       .venv/bin/python experiments/068-validation-gates/pythia_033_panel.py
  env: SEEDS (all | cpu | comma list of kind/L/latent; default cpu = the stratified CPU subset)
       ARMS (engine,standalone,null; default all three)  N_NULL (nulls per seed; default 1 on CPU, 5 on CUDA)
       STEPS (400)  DEVICE (cpu | cuda)  THREADS (torch CPU threads)  OUT (default ./out_panel)
       PART (rows file suffix: several processes may share OUT, each writing OUT/rows.<PART>.jsonl)
       REPORT=1 (rebuild OUT/report.md from the rows only)
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
E033 = HERE.parent / "033-cross-sae-topk"
sys.path.insert(0, str(HERE))
import ref033  # noqa: E402

FAMILY = "pythia-70m"
DEV = torch.device(os.environ.get("DEVICE", "cpu"))
STEPS = int(os.environ.get("STEPS", ref033.STEPS))
OUT = Path(os.environ.get("OUT", str(HERE / "out_panel")))
ARMS = [a for a in os.environ.get("ARMS", "engine,standalone,null").split(",") if a]
N_NULL = int(os.environ.get("N_NULL", 5 if DEV.type == "cuda" else 1))
PART = os.environ.get("PART", "")             # rows file suffix, for several processes sharing OUT
BAND, SUP_MIN, APOS_MIN = (0.8, 1.25), 0.9, 1.0
if os.environ.get("THREADS"):
    torch.set_num_threads(int(os.environ["THREADS"]))

# The CPU subset (L1 only: cost ~ number of upstream sites, L1 attn / mlp 3, L1 resid 5), every kind, two 033 passes
# (attn/1/31694, resid/1/6626) and two 033 fails (attn/1/32511, mlp/1/26871). Measured on the Ryzen 9700X, 8 threads,
# per upstream site and 400 steps: engine fit ~240 s, standalone ~150 s, null ~190 s on an idle CPU, ~3x that on
# 2026-10-09 with two panel processes and another agent's data pass sharing it. (Planned six; resid/1/6115 and
# mlp/2/20702 were dropped for time.)
CPU_SUBSET = ["attn/1/31694", "mlp/1/26871", "attn/1/32511", "resid/1/6626"]


def log(*a):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(x) for x in a)), flush=True)


# --------------------------------------------------------------------------------------------- the panel
def panel():
    """{key: (kind, layer, latent, pos_windows, neg_windows)} for the 30 analysed seeds, plus the token table."""
    a = torch.load(E033 / "scan.pt", weights_only=False)
    b = torch.load(E033 / "scan_panel.pt", weights_only=False)
    assert torch.equal(a["tokens"], b["tokens"]), "033 scan files hold different token tables"
    seeds = {}
    for k, s in a["seeds"].items():                        # old format "L/latent", resid only
        seeds["resid/%d/%d" % (s["layer"], s["latent"])] = ("resid", s["layer"], s["latent"],
                                                             s["pos_windows"], s["neg_windows"], "scan.pt")
    for k, s in b["seeds"].items():
        if k not in seeds:
            seeds[k] = (s["kind"], s["layer"], s["latent"], s["pos_windows"], s["neg_windows"], "scan_panel.pt")
    ref_rows = rows_033()
    analysed = {k: v for k, v in seeds.items()
                if (v[0], v[1], v[2]) in ref_rows and ref_rows[(v[0], v[1], v[2])]["a_pos_ho"] >= APOS_MIN}
    assert len(analysed) == 30, len(analysed)
    return analysed, a["tokens"]


def rows_033():
    """033's triamp400 rows keyed (kind, layer, latent), with its nulls and the pin control attached."""
    out, nulls = {}, {}
    for line in open(E033 / "rows.jsonl"):
        r = json.loads(line)
        key = (r.get("kind", "resid"), r["layer"], r["latent"])
        if r["arm"] == "triamp400":
            out[key] = dict(r)
        elif r["arm"].startswith("null"):
            nulls.setdefault(key, []).append(r)
    pins = {}
    for line in open(E033 / "rows_pin_panel.jsonl"):
        r = json.loads(line)
        if r.get("arm", "").startswith("triamp400"):
            pins[(r.get("kind", "resid"), r["layer"], r["latent"])] = r
    for k, r in out.items():
        r["nulls"] = nulls.get(k, [])
        if k in pins:
            r["pin0"], r["pinM"] = pins[k].get("pin0"), pins[k].get("pinM")
    return out


def passes(r):
    f0, fm, sup = r.get("ampF0"), r.get("ampFM"), r.get("sup")
    return (f0 is not None and fm is not None and sup is not None
            and BAND[0] <= f0 <= BAND[1] and BAND[0] <= fm <= BAND[1] and sup > SUP_MIN)


# --------------------------------------------------------------------------------------------- the engine
def configure_033(steps):
    """The engine's learned-mask config set to 033's triamp400 objective (every knob that has a config field)."""
    from config import config
    config.model.family = FAMILY
    disc, lm = config.discovery, config.discovery.learned_mask
    lm.mask_floor_source = "triple"; lm.free_amplitude = True; lm.l1_lambda = ref033.LAM
    lm.dual_floor_weight = ref033.DUAL_W; lm.triple_floor_weight = ref033.TRIPLE_W      # negctx 0.25, posctx 0.10
    lm.steps = int(steps); lm.lr = ref033.LR; lm.lr_schedule = "constant"
    lm.optimizer = "adamw"; lm.weight_decay = ref033.WEIGHT_DECAY; lm.theta_init = ref033.THETA_INIT
    lm.binarize = "anneal"; lm.keep_threshold = 0.5; lm.holdout_frac = 0.25            # 48 train / 16 held out
    lm.amp_l1 = 0.0; lm.offtarget_weight = 0.0; lm.rank_weight = 0.0                    # no rank-keep in 033
    lm.signed_amplitude = False; lm.neg_suppress_weight = 0.0; lm.margin_topk = None
    lm.floors_train_only = True                                                         # 033's means: train only
    lm.deep_site_threshold = 99; lm.code_dtype = "stream"                               # batch 4, no micro-batching
    lm.log_every = max(1, int(steps) // 4)
    disc.probe_batch_size = 4; disc.probe_sequence_count = 128; disc.eval_sequence_count = 128
    disc.eval_batch_size = 64; disc.context_protocol = "legacy"
    disc.neg_context_selection.preact_filter = False
    return config


class Engine:
    def __init__(self):
        from model.inference import Inference
        from sae.bank import SAEBank
        if DEV.type == "cuda":
            torch.set_float32_matmul_precision("high")         # production / 033 setting
        configure_033(STEPS)
        self.inference = Inference(DEV, family=FAMILY)
        self.bank = SAEBank(device=DEV, family=FAMILY, preload=False)
        self.kinds = list(self.bank.kinds)
        self.avg_acts = torch.zeros((self.bank.n_layer * len(self.kinds), self.bank.d_sae))

    def probe(self, pos, neg, anchors):
        from circuit.probe_dataset import ProbeDataset
        return ProbeDataset(pos_tokens=pos.to(DEV), target_tokens=torch.roll(pos, -1, dims=1).to(DEV),
                            neg_tokens=neg.to(DEV), pos_argmax=anchors, metadata={"n_train": ref033.N_TRAIN})

    def sites(self, layer, kind):
        from eval.ablation_faithfulness import upstream_sites
        return sorted(upstream_sites(self.bank, layer, kind), key=lambda s: (s[0], self.kinds.index(s[1])))

    def fit(self, kind, layer, latent, pos, neg, anchors):
        """discover() with the contexts injected exactly as 062 driver.Runner.fit / 070 smoke.py do."""
        from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
        configure_033(STEPS)
        M = _build_mode_method("ablation_gradient", "mask", self.inference, self.bank, self.avg_acts, None)
        pd_ = self.probe(pos, neg, anchors)
        M.build_probe_dataset = lambda comp, i, _p=pd_: _p
        M._floor_negatives = lambda probe_data, comp, i, logger: probe_data.neg_tokens
        M.last_mask_provenance = None
        c = M.discover(layer * len(self.kinds) + self.kinds.index(kind), latent)
        if c is None:
            raise RuntimeError("discover() returned no circuit")
        ma = {}
        for nd in c.nodes.values():
            if nd.metadata.get("role") == "seed":
                continue
            f = nd.metadata["feature_id"]
            ma.setdefault((f.kind, f.layer), {})[int(f.index)] = float(nd.metadata.get("amplitude", 1.0))
        return c, ma, M.last_mask_provenance or {}

    def null(self, kind, layer, latent, pos, neg, anchors, support):
        """033's null through the engine core: run_learned_mask over a fixed support, gates frozen ON."""
        from circuit.instrument.learned_mask import run_learned_mask
        sites = self.sites(layer, kind)
        sup = {(l_, k_): idx for (k_, l_), idx in support.items() if (l_, k_) in sites}
        _, prov = run_learned_mask(
            self.inference, self.bank, objective="pos", sites=sites, seed_layer=layer, seed_kind=kind,
            seed_latent_idx=latent, pos_tokens=pos.to(DEV), pos_argmax=anchors, neg_tokens=neg.to(DEV),
            mask_floor_source="triple", dual_floor_weight=ref033.DUAL_W, triple_floor_weight=ref033.TRIPLE_W,
            free_amplitude=True, l1_lambda=0.0, steps=STEPS, lr=ref033.LR, keep_threshold=0.5, batch_size=4,
            holdout_frac=0.25, theta_init=40.0, optimizer="adamw", weight_decay=ref033.WEIGHT_DECAY,
            binarize="anneal", floors_train_only=True, support=sup, log_every=0)
        ma = {}
        for site_key, d in (prov.get("amp_kept") or {}).items():
            l_, k_ = site_key.split("/")
            if d:
                ma[(k_, int(l_))] = {int(i): float(a) for i, a in d.items()}
        return ma


# --------------------------------------------------------------------------------------------- run
def all_rows():
    """Every row of every rows*.jsonl in OUT (one file per PART when several processes share OUT)."""
    out = []
    for p in sorted(OUT.glob("rows*.jsonl")):
        for line in open(p):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001  (a truncated last line after a kill)
                pass
    return out


def done_keys():
    return {(r["seed"], r["arm"], r.get("steps")) for r in all_rows() if "error" not in r}


def run():
    seeds, toks = panel()
    want = os.environ.get("SEEDS", "cpu")
    keys = (sorted(seeds, key=lambda k: (int(k.split("/")[1]), k)) if want == "all"
            else CPU_SUBSET if want == "cpu" else [k for k in want.split(",") if k])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "circuits").mkdir(exist_ok=True)
    done = done_keys()
    fh = open(OUT / ("rows.%s.jsonl" % PART if PART else "rows.jsonl"), "a")
    log("device %s | steps %d | arms %s | %d nulls | %d seeds -> %s" % (DEV, STEPS, ARMS, N_NULL, len(keys), OUT))
    R = ref033.Ref033(DEV)
    E = Engine() if ("engine" in ARMS or "null" in ARMS) else None
    meta = dict(steps=STEPS, device=str(DEV), threads=torch.get_num_threads())
    for key in keys:
        kind, layer, latent, pw, nw, src = seeds[key]
        arms_left = [a for a in ARMS if a != "null" and (key, a, STEPS) not in done]
        arms_left += ["null%d" % d for d in range(N_NULL) if "null" in ARMS and (key, "null%d" % d, STEPS) not in done]
        if not arms_left:
            continue
        pos, neg = toks[pw], toks[nw]
        t0 = time.time()
        S = R.seed(kind, layer, latent, pos.to(DEV), neg.to(DEV))
        anchors = S.anchors()
        log("%s (%s): a_pos_ho %.3f e0 %.3f eM %.3f a_base %.3f | 033 UP %d sites, engine %d | prep %.0fs"
            % (key, src, S.a_pos_ho, S.e0, S.eM, S.a_base, len(S.UP),
               len(E.sites(layer, kind)) if E else -1, time.time() - t0))
        base = dict(seed=key, kind=kind, layer=layer, latent=latent, scan=src, **meta)
        n_engine = None
        for arm in arms_left:
            t1 = time.time()
            try:
                if arm == "engine":
                    c, ma, prov = E.fit(kind, layer, latent, pos, neg, anchors)
                    t_fit = time.time() - t1
                    torch.save({key: c}, OUT / "circuits" / ("%s.pt" % key.replace("/", "_")))
                    row = S.score(ma, arm)
                    row.update(t_fit=round(t_fit, 1), loss_initial=prov.get("loss_initial"),
                               loss_final=prov.get("loss_final"), holdout_data_loss=prov.get("holdout_data_loss"),
                               members={"%s.%d" % s: sorted(d) for s, d in ma.items()})
                elif arm in ("standalone", "standalone_notemp"):
                    ma = S.fit(True, ref033.LAM, steps=STEPS, pen_temp=(arm == "standalone"))
                    t_fit = time.time() - t1
                    row = S.score(ma, arm)
                    row.update(t_fit=round(t_fit, 1), members={"%s.%d" % s: sorted(d) for s, d in ma.items() if d})
                else:                                           # nullD
                    if n_engine is None:
                        n_engine = engine_n(key)
                    if not n_engine:
                        log("  %s: no engine row yet, null skipped" % arm)
                        continue
                    live, rate = S.live_pool()
                    eng_sites = set(E.sites(layer, kind))
                    live = [(s, i) for s, i in live if (s[1], s[0]) in eng_sites]
                    d = int(arm[4:])
                    support = S.null_members(live, n_engine, d, N_NULL)
                    ma = E.null(kind, layer, latent, pos, neg, anchors, support)
                    t_fit = time.time() - t1
                    row = S.score(ma, arm)
                    row.update(t_fit=round(t_fit, 1), n_drawn=min(n_engine, len(live)), live_pool=len(live),
                               anchor_support_rate=round(rate, 5))
                row.update(base, arm=arm, secs=round(time.time() - t1, 1), passes=passes(row))
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                row = dict(base, arm=arm, error="%s: %s" % (type(e).__name__, str(e)[:300]))
            fh.write(json.dumps(row) + "\n"); fh.flush()
            log("  %-10s n=%-5s F0=%-7s FM=%-7s sup=%-6s cf=%-7s pin0=%-7s %s fit %ss"
                % (arm, row.get("n"), row.get("ampF0"), row.get("ampFM"), row.get("sup"), row.get("cf_amp"),
                   row.get("pin0"), "PASS" if row.get("passes") else "-", row.get("t_fit")))
    fh.close()


def engine_n(key):
    """The engine circuit's size for `key` (the last engine row at this STEPS), or None."""
    n = None
    for r in all_rows():
        if r["seed"] == key and r["arm"] == "engine" and r.get("steps") == STEPS and "error" not in r:
            n = r["n"]
    return n


# --------------------------------------------------------------------------------------------- report
def jaccard(a, b):
    """Jaccard of two member listings {"kind.layer": [latents]}."""
    sa = {(s, i) for s, v in a.items() for i in v}
    sb = {(s, i) for s, v in b.items() for i in v}
    return len(sa & sb) / max(len(sa | sb), 1)


def report():
    ref = rows_033()
    rows = {}
    for r in all_rows():
        if "error" not in r and r.get("steps") == STEPS:
            rows.setdefault(r["seed"], {})[r["arm"]] = r
    seeds, _ = panel()
    f = lambda v, d=3: ("%.*f" % (d, v)) if isinstance(v, (int, float)) else "-"  # noqa: E731
    yn = lambda r: "PASS" if r is not None and passes(r) else ("-" if r is not None else "")  # noqa: E731
    lines = ["# DAN-36: 033 Pythia panel through the engine (%s)\n" % time.strftime("%Y-%m-%d %H:%M"),
             "steps %d. Pass = ampF0 and ampFM in [0.8, 1.25] and sup > 0.9 (held-out 16). 033 = rows.jsonl "
             "(GPU, TF32); engine / standalone = this run, scored by 033's scorer (ref033.Seed.score).\n" % STEPS,
             "| seed | 033 n / F0 / FM / sup / cf / pass | engine n / F0 / FM / sup / cf / pin0 / pass | "
             "standalone n / F0 / FM / sup / cf / pass | standalone, engine pricing n / F0 / FM / pass | "
             "engine fit s | nulls pass (max F0) |",
             "|---|---|---|---|---|---|---|"]
    cnt = dict(seeds=0, p033=0, peng=0, pstd=0, agree_eng=0, agree_std=0, n_eng=[], n_std=[], dF0=[], dFM=[],
               nulls=0, null_pass=0, null_max=0.0, agree_eng_std=0, both=0)
    for key in sorted(rows, key=lambda k: (int(k.split("/")[1]), k)):
        if key not in seeds:
            continue
        kind, layer, latent = seeds[key][:3]
        o = ref[(kind, layer, latent)]
        e, s = rows[key].get("engine"), rows[key].get("standalone")
        nl = [v for a, v in rows[key].items() if a.startswith("null")]
        if e is None:
            continue
        cnt["seeds"] += 1
        cnt["p033"] += passes(o); cnt["peng"] += passes(e); cnt["agree_eng"] += passes(o) == passes(e)
        cnt["n_eng"].append(e["n"] / max(o["n"], 1))
        if e.get("ampF0") is not None and o.get("ampF0") is not None:
            cnt["dF0"].append(abs(e["ampF0"] - o["ampF0"]))
        if e.get("ampFM") is not None and o.get("ampFM") is not None:
            cnt["dFM"].append(abs(e["ampFM"] - o["ampFM"]))
        if s is not None:
            cnt["both"] += 1; cnt["pstd"] += passes(s); cnt["agree_std"] += passes(o) == passes(s)
            cnt["agree_eng_std"] += passes(e) == passes(s); cnt["n_std"].append(e["n"] / max(s["n"], 1))
        if s is not None and e.get("members") is not None and s.get("members") is not None:
            cnt.setdefault("jac", []).append("%s %.2f" % (key, jaccard(e["members"], s["members"])))
        for r in nl:
            cnt["nulls"] += 1; cnt["null_pass"] += passes(r)
            cnt["null_max"] = max(cnt["null_max"], r.get("ampF0") or 0.0)
        t = rows[key].get("standalone_notemp")
        lines.append("| %s | %d / %s / %s / %s / %s / %s | %d / %s / %s / %s / %s / %s / %s | %s | %s | %s | %s |" % (
            key, o["n"], f(o["ampF0"]), f(o["ampFM"]), f(o["sup"]), f(o["cf_amp"]), yn(o),
            e["n"], f(e["ampF0"]), f(e["ampFM"]), f(e["sup"]), f(e["cf_amp"]), f(e.get("pin0")), yn(e),
            ("%d / %s / %s / %s / %s / %s" % (s["n"], f(s["ampF0"]), f(s["ampFM"]), f(s["sup"]), f(s["cf_amp"]),
                                              yn(s))) if s else "",
            ("%d / %s / %s / %s" % (t["n"], f(t["ampF0"]), f(t["ampFM"]), yn(t))) if t else "",
            f(e.get("t_fit"), 0), ("%d/%d (%s)" % (sum(map(passes, nl)), len(nl),
                                                   f(max(r.get("ampF0") or 0 for r in nl)))) if nl else ""))
    med = lambda v: statistics.median(v) if v else float("nan")  # noqa: E731
    o_null = [r for k in rows if k in seeds for r in ref[tuple(seeds[k][:3])]["nulls"]]
    lines += ["", "## Summary (%d seeds)\n" % cnt["seeds"],
              "- pass: 033 %d / engine %d / standalone %d (of %d with a standalone row)" % (
                  cnt["p033"], cnt["peng"], cnt["pstd"], cnt["both"]),
              "- per-seed verdict agreement: engine vs 033 %d/%d; standalone vs 033 %d/%d; engine vs standalone %d/%d"
              % (cnt["agree_eng"], cnt["seeds"], cnt["agree_std"], cnt["both"], cnt["agree_eng_std"], cnt["both"]),
              "- size ratio n_engine / n_033: median %.2f (min %.2f, max %.2f); n_engine / n_standalone median %.2f"
              % (med(cnt["n_eng"]), min(cnt["n_eng"] or [float("nan")]), max(cnt["n_eng"] or [float("nan")]),
                 med(cnt["n_std"])),
              "- |ampF0 engine - 033| median %.3f (max %.3f); |ampFM| median %.3f (max %.3f)" % (
                  med(cnt["dF0"]), max(cnt["dF0"] or [float("nan")]), med(cnt["dFM"]),
                  max(cnt["dFM"] or [float("nan")])),
              "- engine nulls: %d / %d pass, max ampF0 %.3f (033 on these seeds: %d / %d pass, max ampF0 %.3f)" % (
                  cnt["null_pass"], cnt["nulls"], cnt["null_max"], sum(map(passes, o_null)), len(o_null),
                  max([r.get("ampF0") or 0 for r in o_null] or [0.0])),
              "- membership Jaccard engine vs standalone (same machine, same contexts; expected to differ "
              "numerically): %s" % (", ".join(cnt.get("jac", [])) or "-")]
    txt = "\n".join(lines)
    (OUT / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if os.environ.get("REPORT") != "1":
        run()
    report()
