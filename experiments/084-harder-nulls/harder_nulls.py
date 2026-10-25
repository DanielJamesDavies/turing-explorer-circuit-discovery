"""DAN-181: harder random-circuit nulls for weighted circuits (anchor-active and contribution-matched), with alpha = 1
scoring of every random set and the training-split diagnostics. Supersedes the DAN-46 tiers (see README).

Per target (the production WCM circuit's own target, its protocol-v1 contexts and training arm):
  profile    on the fit's training contexts (arm B: 32 strongest + 16 mid-band; thin targets arm A), every upstream
             latent's anchor lag (nulls_core: minimum anchor - position over positions <= anchor where it fires), its
             corpus frequency decile per site (latent_stats.active_count), and its contribution to the target
             (activation x gradient of the target's pre-activation at the anchor, summed over the causal prefix,
             |.| per context, averaged over contexts)
  nulls      anchor   members drawn from the ANCHOR-ACTIVE pool (lag <= WIN, default 8), production members included
                      (2026-10-24 final; EXCLUDE_MEMBERS lists the nulls whose pool drops them, default contrib:
                      the contrib null is a substitution test), site by site at the production circuit's per-site counts, each
                      member replaced by a pool latent of the same frequency decile (nearest decile when the bin is
                      empty); the overlap with the production members is recorded per draw
             contrib  as anchor, matched on (contribution bin, then frequency decile); the bins are the production
                      members' own contribution deciles, so the drawn set reproduces their contribution distribution
             uniform  (optional, NULLS=...,uniform) the anchor-active pool drawn uniformly: isolates the pool
             each draw is fitted EXACTLY as DAN-17 (random_null.py's support wrapper: support = the drawn members,
             lambda 0, gates fully open, the production objective, contexts, steps, lr, rank-keep; driver.Runner.fit)
  scoring    production scorer (driver.Runner.score) on held-out strongest ("held") and on 16 of the fit's strongest
             TRAINING contexts ("train"; same reference set), fitted coefficients; plus every member at alpha = 1
             on held-out strongest (the Shi et al. / CircuitKIT null: size-matched, nothing fitted)
  references wcm      the production circuit (fitted, held + train) and wcm_a1 (its members at alpha = 1)
             dan17    the DAN-17 random circuit of the same target (the first RN_DIR holding circuits/<key>.d0.pt;
                      fitted: held + train) and dan17_a1, so all three nulls are scored on one machine. A target with
                      no stored DAN-17 circuit gets one drawn and fitted here exactly as 062/random_null.py does
                      (DAN17_REFIT=1, default; nulls_core.dan17_draw; OUT84/circuits/dan17/<key>.d0.pt), so every
                      target has the reference; the row's dan17_source says which (a directory name or "refit")
Rows (one per seed x arm x draw x split, resumable): OUT84/nulls[.shard<i>].jsonl; per-target pool / coverage rows:
OUT84/profile[.shard<i>].jsonl; drawn sets OUT84/draws/<null>/<key>.d<draw>.json (written before the fit, so a resume
refits the identical set); circuits OUT84/circuits/<null>/<key>.d<draw>.pt; fit stats (no curve) OUT84/train*.jsonl.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python experiments/084-harder-nulls/harder_nulls.py
  env: TARGETS  default (TuringLLM) the frozen panel targets_turingllm_nulls.txt (make_panel.py: 126 of the 062
                sweep targets, 14 per band x kind, the 63 DAN-17 sweep targets first); a .txt list is taken in file
                order (public models: 082's targets_<model>_nulls.txt, subsample first); a .jsonl (a run's rows) in
                first-seen order; a .csv (targets.csv) or STRATIFY=1 re-draws stratified
       N_TARGETS (0 = the whole list)  N_DRAWS (1)  SUB_N (63: the first 63 of the order, 7 per cell; 64 on the
       public lists) get SUB_DRAWS (4) draws
       NULLS (anchor,contrib)  WIN (8)  A1 (1: alpha = 1 rows)  SPLITS (held,train)  REFS (wcm,dan17)
       RN_DIR (comma list of DAN-17 run dirs holding circuits/<key>.d0.pt, first hit wins; default on TuringLLM
       062/out_random,062/out_random_scale, none on public families)  DAN17_REFIT (1: draw + fit the missing ones)
       EXCLUDE_MEMBERS (contrib; 0 = none, 1 = every null, or a comma list)  MOVE_TOL (0.01)  ATTR_BS (4)  OUT84 (experiments/084-harder-nulls/out)  LIMIT
       PROFILE_ONLY=1 (profiles + coverage rows only: no fits)  REPORT=1 (report.py; no model)
       SHARD=i/k, FAMILY / DATA / STORES / LAM / STEPS as driver.py (launch.sh SCRIPT=... runs it multi-GPU)
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
E62 = EXP / "062-h100-protocol-v1"
sys.path.insert(0, str(HERE))
import nulls_core as NC  # noqa: E402

if os.environ.get("REPORT"):
    import report  # noqa: E402
    report.main()
    sys.exit(0)

import torch  # noqa: E402

os.environ.setdefault("OUT", str(E62 / "out_full" / "out"))
sys.path.insert(0, str(E62))
import random_null as RN  # noqa: E402  (imports 062's driver: FAMILY / SHARD / LAM / STEPS)
import rescore_alpha1 as RA  # noqa: E402

driver = RN.driver
OUT84 = Path(os.environ.get("OUT84", str(HERE / "out")))
N_TARGETS = int(os.environ.get("N_TARGETS", 0))               # 0 = every target of the list
N_DRAWS = int(os.environ.get("N_DRAWS", 1))
SUB_N = int(os.environ.get("SUB_N", 63))
SUB_DRAWS = int(os.environ.get("SUB_DRAWS", 4))
NULLS = [x for x in os.environ.get("NULLS", "anchor,contrib").split(",") if x]
WIN = int(os.environ.get("WIN", NC.WIN_DEFAULT))
A1 = os.environ.get("A1", "1") == "1"
SPLITS = [x for x in os.environ.get("SPLITS", "held,train").split(",") if x]
REFS = [x for x in os.environ.get("REFS", "wcm,dan17").split(",") if x]
EXCLUDE = NC.parse_exclude(os.environ.get("EXCLUDE_MEMBERS"))   # nulls whose pool drops the members (default contrib)
DAN17_REFIT = os.environ.get("DAN17_REFIT", "1") == "1"
MOVE_TOL = float(os.environ.get("MOVE_TOL", 0.01))
ATTR_BS = int(os.environ.get("ATTR_BS", 4))
REF_STEPS = int(os.environ.get("REF_STEPS", 400))        # the budget the production / DAN-17 circuits were fitted at
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
PROFILE_ONLY = os.environ.get("PROFILE_ONLY") == "1"
STRATIFY = os.environ.get("STRATIFY") == "1"
PANEL = HERE / "targets_turingllm_nulls.txt"                 # make_panel.py (frozen)
_rn = os.environ.get("RN_DIR")
RN_DIRS = ([Path(x) for x in _rn.split(",") if x] if _rn is not None
           else ([] if driver.PUBLIC else [E62 / "out_random", E62 / "out_random_scale"]))
FIT = "fit"
ONE = "one"


def log(s):
    print("[%s %s] %s" % (time.strftime("%H:%M:%S"), driver.TAG, s), flush=True)


# ---------------------------------------------------------------------------------------------- targets
def target_order(n_layer):
    """The full ordered target list (before sharding): positions decide the subsample that gets extra draws."""
    t = os.environ.get("TARGETS")
    if not t:
        if driver.PUBLIC:
            raise SystemExit("TARGETS is required on a public family (082's targets_<model>_nulls.txt)")
        if not PANEL.exists():
            raise SystemExit("no %s: run make_panel.py first" % PANEL)
        path = PANEL
    else:
        path = Path(t)
    if path.suffix == ".jsonl":
        keys = NC.unique_in_order(json.loads(l)["seed"] for l in open(path) if l.strip())
    elif path.suffix == ".csv":
        import pandas as pd
        return NC.stratified_order(list(pd.read_csv(path, index_col="seed").index), N_TARGETS, n_layer=n_layer)
    else:
        keys = [k for k in path.read_text().split() if k]
    if STRATIFY:
        return NC.stratified_order(keys, N_TARGETS or len(keys), n_layer=n_layer)
    return keys[:N_TARGETS] if N_TARGETS else keys


def dan17_path(key):
    """(path, source) of the target's stored DAN-17 circuit: the first RN_DIRS entry holding it; else, with
    DAN17_REFIT, the in-process refit's path (source 'refit'); else (None, None)."""
    for d in RN_DIRS:
        p = d / "circuits" / ("%s.d0.pt" % key)
        if p.exists():
            return p, d.name
    if DAN17_REFIT:
        return OUT84 / "circuits" / "dan17" / ("%s.d0.pt" % key), "refit"
    return None, None


# ---------------------------------------------------------------------------------------------- profile
def _contrib_instrument():
    from circuit.instrument.ig_baseline import InterpolatedCodeInstrument

    class NaturalCodeGrad(InterpolatedCodeInstrument):
        """Every upstream site replaced by decode(its own natural code) + its clean residual, with the code's nonzero
        values as gradient leaves (sparse [B, T, K]) and SFC pass-through gradients; leaf x leaf.grad is then each
        latent's activation x gradient at every position (ig_baseline's instrument at alpha = 1 with a zero floor,
        without the dense [B, T, d_sae] leaves)."""

        def transform(self, layer_idx, kind, x):
            if layer_idx == self.seed_layer and kind == self.seed_kind:
                self.seed_pre_act = x @ self.w_seed.to(device=x.device, dtype=x.dtype) + self.b_seed.to(
                    device=x.device, dtype=x.dtype)
                return x
            site = (layer_idx, kind)
            if site not in self.substitute_sites:
                return x
            ta, ti = self.natural_codes[site]
            leaf = ta.detach().float().requires_grad_(True)
            dense = torch.zeros(*ta.shape[:-1], self.bank.d_sae, dtype=torch.float32, device=ta.device)
            dense = dense.scatter_add(-1, ti.long(), leaf)
            self.anchors[site] = leaf
            self.deltas[site] = ti
            return (self.bank.decode(dense.to(x.dtype), kind, layer_idx)
                    + self.residuals[site].to(device=x.device, dtype=x.dtype) + (x - x.detach()))
    return NaturalCodeGrad


def contributions(G, key, tokens, anchors, sites, bs=ATTR_BS):
    """{site: (abs [d_sae], signed [d_sae])}: per latent, mean over contexts of |sum_{p <= anchor} a_p dpre/da_p| and of
    the signed sum (pre = the target's encoder pre-activation at the context's anchor)."""
    from circuit.instrument.ig_baseline import collect_natural_codes
    from eval.wcm_eval import site_encoder
    inference, bank = G["inference"], G["bank"]
    layer, kind, i = NC.parse(key)
    We, be = site_encoder(bank, layer, kind)
    Inst = _contrib_instrument()
    D = int(bank.d_sae)
    acc_abs = {s: torch.zeros(D, dtype=torch.float64) for s in sites}
    acc_sum = {s: torch.zeros(D, dtype=torch.float64) for s in sites}
    n = 0
    sset = set(sites)
    inference.disable_compile()
    try:
        for s0 in range(0, int(tokens.shape[0]), bs):
            tk, an = tokens[s0:s0 + bs], anchors[s0:s0 + bs]
            B, T = int(tk.shape[0]), int(tk.shape[1])
            codes, residuals = collect_natural_codes(inference, bank, tk, sset)
            inst = Inst(bank, sset, codes, residuals, {}, 1.0, layer, kind, We[i], be[i])
            inference.forward(tk, patcher=inst, grad_enabled=True, return_activations=False, tokenize_final=False)
            pre = inst.seed_pre_act
            if pre is None:
                raise RuntimeError("target pre-activation was not captured")
            rr = torch.arange(B, device=pre.device)
            metric = pre[rr, an.to(pre.device).long().clamp(0, T - 1)].float().sum()     # per-context attribution
            order = sorted(inst.anchors)
            grads = torch.autograd.grad(metric, [inst.anchors[s] for s in order], allow_unused=True)
            keep = (torch.arange(T)[None, :] <= an.cpu().long()[:, None]).float()        # [B, T] causal prefix
            for s, g in zip(order, grads):
                if g is None:
                    continue
                attr = (g.detach().float() * inst.anchors[s].detach()).cpu() * keep[..., None]
                per = torch.zeros(B, D, dtype=torch.float64).scatter_add(
                    1, inst.deltas[s].reshape(B, -1).long().cpu(), attr.reshape(B, -1).double())
                acc_abs[s] += per.abs().sum(0)
                acc_sum[s] += per.sum(0)
            n += B
            inst.release()
            del inst, grads, codes, residuals
    finally:
        inference.enable_compile()
    return {s: ((acc_abs[s] / max(n, 1)).float(), (acc_sum[s] / max(n, 1)).float()) for s in sites}


def anchor_lags(G, tokens, anchors, sites):
    """{site: [d_sae] min anchor lag} over the contexts (nulls_core.site_min_lag), forward passes only."""
    bank, D = G["bank"], int(G["D"])
    k2i = {k: n for n, k in enumerate(G["KINDS"])}
    out = {s: torch.full((D,), NC.LAG_NONE, dtype=torch.long) for s in sites}
    cur = {}

    def hook(layer_idx, activations):
        for kd in G["KINDS"]:
            s = (layer_idx, kd)
            if s in out:
                ta, ti = bank.encode(activations[k2i[kd]], kd, layer_idx)
                out[s] = torch.minimum(out[s], NC.site_min_lag(ta, ti, cur["an"], D))
    G["inference"].disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), 16):
                cur["an"] = anchors[s0:s0 + 16]
                G["inference"].forward(tokens[s0:s0 + 16].to(G["device"]), activations_callback=hook,
                                       return_activations=False, tokenize_final=False)
    finally:
        G["inference"].enable_compile()
    return out


def build_profile(R, key, tokens, anchors, ref_sites, extra_sets=()):
    """The target's profile (nulls_core's dict form): lag / fdec / count per site and contrib / cdec per (site,
    latent), over every latent active in a causal prefix plus every member of ref_sites and extra_sets."""
    from eval.ablation_faithfulness import upstream_sites
    from pipeline.component_index import component_idx
    from store.latent_stats import latent_stats
    G = R.G
    layer, kind, _ = NC.parse(key)
    sites = sorted((int(l), str(k)) for l, k in upstream_sites(G["bank"], layer, kind))
    t0 = time.time()
    lags = anchor_lags(G, tokens, anchors, sites)
    t_lag = time.time() - t0
    t0 = time.time()
    con = contributions(G, key, tokens.to(G["device"]), anchors, sites)
    t_con = time.time() - t0
    prof = dict(lag={}, fdec={}, count={}, contrib={}, signed={}, cdec={}, sites=sites)
    for s in sites:
        keep = set((lags[s] < NC.LAG_NONE).nonzero(as_tuple=True)[0].tolist())
        for st in (ref_sites,) + tuple(extra_sets):
            keep.update(int(m) for m in st.get(s, ()))
        keep = sorted(keep)
        cnt = latent_stats.active_count[component_idx(s[0], G["KINDS"].index(s[1]), G["NK"])].cpu()
        fd = NC.freq_deciles(cnt.numpy())
        lg = lags[s].tolist() if keep else []
        ca, cs = con[s][0].tolist(), con[s][1].tolist()
        prof["lag"][s] = {l: int(lg[l]) for l in keep}
        prof["fdec"][s] = {l: int(fd[l]) for l in keep}
        prof["count"][s] = {l: int(cnt[l]) for l in keep}
        for l in keep:
            prof["contrib"][(s, l)] = float(ca[l]); prof["signed"][(s, l)] = float(cs[l])
    prof["cdec"] = contrib_deciles(prof, ref_sites, WIN)
    prof["cbin"] = NC.member_bins([prof["contrib"].get((s, int(m)), 0.0) for s, v in ref_sites.items() for m in v],
                                  prof["contrib"])
    prof["t_lag"], prof["t_contrib"] = round(t_lag, 1), round(t_con, 1)
    return prof


def contrib_deciles(prof, ref_sites, win):
    """Contribution deciles over the union U of the anchor-active pool (every site) and the production members (rank
    deciles, the bins the contrib null matches on); every other stored latent (e.g. a DAN-17 member) gets the decile
    its value would take in U (nulls_core.place_deciles), for the reported distributions only."""
    u = {(s, l) for s, v in NC.anchor_pool(prof, win).items() for l in v}
    u.update((s, int(m)) for s, v in ref_sites.items() for m in v)
    dec = NC.rank_deciles({x: prof["contrib"].get(x, 0.0) for x in u})
    rest = {x: v for x, v in prof["contrib"].items() if x not in dec}
    dec.update(NC.place_deciles(rest, [prof["contrib"].get(x, 0.0) for x in u]))
    return dec


# ---------------------------------------------------------------------------------------------- scoring
def scalars(r):
    return {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None}


def train_rec(rec, arm):
    """The record with its held-out strongest set replaced by 16 of the fit's strongest TRAINING contexts (arm B:
    D_train; arm A: strongest-train), so Runner.score gives the training-split row with the same scorer and the same
    48-context reference set."""
    idx = rec["D_train"] if arm == "B" else rec["strong"]["train"]
    out = dict(rec)
    out["strong"] = dict(rec["strong"], held=NC.train_subset(idx))
    return out


def score_job(R, c, rec, rec_tr, split, amode):
    """One production-scorer row: held-out strongest (split "held") or the training-split record ("train"); amode
    "one" scores every member at alpha = 1 (rescore_alpha1.ones: the same nodes, nothing fitted)."""
    if split not in ("held", "train") or amode not in (FIT, ONE):
        raise ValueError("split %r / alpha mode %r" % (split, amode))
    return R.score(c, rec_tr if split == "train" else rec, "strong", alphas=RA.ones(c) if amode == ONE else None)


def circuit_alphas(c):
    return [float(n.metadata.get("amplitude", 1.0)) for n in c.nodes.values() if n.metadata.get("role") != "seed"]


class TrainSink:
    """driver.Runner.fit's train_fh: keeps the fit stats, drops the per-step curve, adds the null labels."""

    def __init__(self, fh):
        self.fh, self.label = fh, {}

    def write(self, s):
        r = json.loads(s)
        r.pop("curve", None); r.pop("curve_steps", None)
        r.update(self.label)
        self.fh.write(json.dumps(r) + "\n")

    def flush(self):
        self.fh.flush()


def main():
    OUT84.mkdir(parents=True, exist_ok=True)
    rows_path = driver.shard_jsonl(OUT84, "nulls")
    prof_path = driver.shard_jsonl(OUT84, "profile")
    done = {(r.get("seed"), r.get("arm"), r.get("draw"), r.get("split")) for r in driver.read_jsonl_all(OUT84, "nulls")
            if "error" not in r}
    prof_done = {r.get("seed") for r in driver.read_jsonl_all(OUT84, "profile") if "error" not in r}
    RN.install_support_wrapper()
    R = driver.Runner()
    G = R.G
    n_layer = int(G["bank"].n_layer)
    order = target_order(n_layer)
    pos_of = {k: j for j, k in enumerate(order)}
    keys = order[driver.SHARD_I::driver.SHARD_K]
    if LIMIT:
        keys = keys[:LIMIT]
    M = R.method(driver.GAMMA, driver.LAM, True)
    from config import config
    lm = config.discovery.learned_mask
    inert = NC.alpha_inert(lm.steps, lm.lr, lm.weight_decay, lm.lr_schedule) if lm.optimizer == "adamw" else None
    # the references (production / DAN-17 circuits) were fitted at the protocol's budget whatever STEPS says here
    inert_ref = NC.alpha_inert(REF_STEPS, lm.lr, lm.weight_decay, lm.lr_schedule) if lm.optimizer == "adamw" else None
    log("084 harder nulls | family %s | %d targets (of %d; shard %d/%d) | nulls %s | WIN %d | members %s | draws %d "
        "(first %d: %d) | A1 %s | splits %s | refs %s | rn_dirs %s refit %s | steps %d lam %g | alpha_inert %s | %d rows "
        "done%s" % (
            driver.FAMILY, len(keys), len(order), driver.SHARD_I, driver.SHARD_K, NULLS, WIN,
            "excluded from %s" % ",".join(sorted(EXCLUDE)) if EXCLUDE else "included", N_DRAWS, SUB_N, max(N_DRAWS, SUB_DRAWS), A1, SPLITS, REFS,
            [str(d) for d in RN_DIRS], DAN17_REFIT, driver.STEPS, driver.LAM, inert, len(done),
            " | PROFILE_ONLY" if PROFILE_ONLY else ""))
    fh, fp = open(rows_path, "a"), open(prof_path, "a")
    sink = TrainSink(open(driver.shard_jsonl(OUT84, "train"), "a"))
    t_start = time.time()
    for n_, key in enumerate(keys):
        layer, kind, _ = NC.parse(key)
        base = dict(seed=key, layer=layer, kind=kind, band=NC.band(layer, n_layer), family=driver.FAMILY, win=WIN,
                    order_pos=pos_of[key], shard=driver.TAG)
        nd = NC.n_draws_for(pos_of[key], N_DRAWS, SUB_N, SUB_DRAWS)
        main_c = driver.OUT / "main" / "circuits" / ("%s.pt" % key)
        rn_c, src17 = dan17_path(key) if "dan17" in REFS else (None, None)
        has17 = rn_c is not None                                       # stored, or refit here (DAN17_REFIT)
        refit17 = src17 == "refit"
        jobs = NC.plan_jobs(REFS, has17, NULLS, nd, SPLITS, A1)        # (arm, null, draw, split, alpha mode)
        todo = [j for j in jobs if (key, j[0], j[2], j[3]) not in done]
        if PROFILE_ONLY:
            todo = [] if key in prof_done else [None]
        if not todo:
            continue
        if not main_c.exists():
            log("%3d/%d %-16s no production circuit; skipped" % (n_ + 1, len(keys), key)); continue
        ref = torch.load(main_c, weights_only=False)
        if ref is None:
            log("%3d/%d %-16s production circuit rejected; skipped" % (n_ + 1, len(keys), key)); continue
        t_target = time.time()
        try:
            ref_sites = {s: list(v) for s, v in RN.members_by_site(ref).items()}   # member order = random_null's
            rec = R.contexts(key)
            arm, thin = R.train_arm(rec)
            c17, s17, d17 = None, {}, {}
            if has17 and not refit17:
                c17 = torch.load(rn_c, weights_only=False)
                s17 = {s: list(v) for s, v in RN.members_by_site(c17).items()} if c17 is not None else {}
                d17 = dict(dan17_source=src17)
            elif has17:                                    # DAN-17's own draw, saved before the fit (resume = same set)
                dpath17 = OUT84 / "draws" / "dan17" / ("%s.d0.json" % key)
                if dpath17.exists():
                    dj = json.loads(dpath17.read_text())
                    s17 = {(int(s.split(".")[0]), s.split(".")[1]): v for s, v in dj["support"].items()}
                    d17 = dj["diag"]
                else:
                    st_ = rec["strong"]
                    tr = st_["pos"][st_["train"]] if "train" in st_ else st_["pos"]
                    live = RN.live_latents(R, tr, set(ref_sites))
                    s17, d17 = NC.dan17_draw(key, ref_sites.items(), live)
                    d17 = dict(d17, dan17_source="refit")
                    dpath17.parent.mkdir(parents=True, exist_ok=True)
                    dpath17.write_text(json.dumps(dict(seed=key, null="dan17", draw=0, diag=d17, support={
                        "%d.%s" % s: v for s, v in s17.items()})))
            from circuit.protocol_contexts import training_set
            tok, _, anc = training_set(rec, arm)
            prof = build_profile(R, key, tok, anc, ref_sites, (s17,))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            fp.write(json.dumps(dict(base, error="%s: %s" % (type(e).__name__, str(e)[:300]))) + "\n"); fp.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        pool = NC.anchor_pool(prof, WIN)
        prow = dict(base, train_arm=arm, thin=thin, n_train_ctx=int(tok.shape[0]), n_sites=len(prof["sites"]),
                    n_ref=sum(len(v) for v in ref_sites.values()), pool_total=sum(len(v) for v in pool.values()),
                    pool_by_win={("w%d" % w): sum(1 for s in prof["lag"] for g in prof["lag"][s].values() if g <= w)
                                 for w in NC.COVER_WINDOWS},
                    cover_wcm=NC.coverage(ref_sites, prof), cover_dan17=NC.coverage(s17, prof) if s17 else None,
                    set_wcm=NC.set_profile(ref_sites, prof, WIN), set_dan17=NC.set_profile(s17, prof, WIN) if s17 else None,
                    dan17_source=src17, exclude_members=sorted(EXCLUDE), t_lag=prof["t_lag"], t_contrib=prof["t_contrib"])
        if key not in prof_done:
            fp.write(json.dumps(prow) + "\n"); fp.flush()
            prof_done.add(key)
        if PROFILE_ONLY:
            log("%3d/%d %-16s profile: pool %d (n_ref %d) | wcm anchor-active %.2f prefix %.2f | lag %.0fs contrib %.0fs" % (
                n_ + 1, len(keys), key, prow["pool_total"], prow["n_ref"], prow["cover_wcm"].get("w%d" % WIN, 0),
                prow["cover_wcm"].get("prefix", 0), prof["t_lag"], prof["t_contrib"]))
            continue
        rec_tr = train_rec(rec, arm)
        circuits = {("wcm", 0): ref}
        if has17 and not refit17:
            circuits[("dan17", 0)] = c17
        sets = {("wcm", 0): ref_sites, ("dan17", 0): s17}
        diags = {("dan17", 0): d17} if has17 and not refit17 else {}
        for arm_, nl, d, split, amode in todo:
            row = dict(base, arm=arm_, null=nl, draw=d, split=split, alpha=amode, train_arm=arm, thin=thin,
                       n_ref=prow["n_ref"])
            t0 = time.time()
            try:
                if (nl, d) not in circuits:                    # a null draw (or the DAN-17 refit): sample / reload, fit
                    dpath = OUT84 / "draws" / nl / ("%s.d%d.json" % (key, d))
                    cpath = OUT84 / "circuits" / nl / ("%s.d%d.pt" % (key, d))
                    if nl == "dan17":                          # drawn above (refit17), same file layout
                        support, diag = s17, d17
                    elif dpath.exists():
                        dj = json.loads(dpath.read_text())
                        support = {(int(s.split(".")[0]), s.split(".")[1]): v for s, v in dj["support"].items()}
                        diag = dj["diag"]
                    else:
                        support, diag = NC.draw_null(nl, ref_sites, prof, WIN, NC.rng_for(key, nl, d),
                                                     exclude_members=nl in EXCLUDE)
                        dpath.parent.mkdir(parents=True, exist_ok=True)
                        dpath.write_text(json.dumps(dict(seed=key, null=nl, draw=d, win=WIN, diag=diag, support={
                            "%d.%s" % s: v for s, v in support.items()})))
                    sink.label = dict(null=nl, draw=d)
                    if not diag.get("n_drawn"):                # empty pool at every site: nothing to fit
                        c, t_fit = None, 0.0
                        diag = dict(diag, empty_support=True)
                    else:
                        RN._SUPPORT.clear(); RN._SUPPORT.update(support)
                        try:
                            c, t_fit = R.fit(M, rec, key, arm, cpath, train_fh=sink)
                        finally:
                            RN._SUPPORT.clear()
                    circuits[(nl, d)], sets[(nl, d)], diags[(nl, d)] = c, support, dict(diag, t_fit=round(t_fit, 1))
                c = circuits[(nl, d)]
                row.update(diags.get((nl, d), {}))
                if c is None:
                    row["skip"] = "empty_support" if row.get("empty_support") else "rejected"
                else:
                    t1 = time.time()
                    row = dict(scalars(score_job(R, c, rec, rec_tr, split, amode)), **row)
                    row["t_score"] = round(time.time() - t1, 1)
                    row["pass"] = NC.passes(row)
                    if amode == FIT:                           # stored references were fitted at REF_STEPS
                        stored = nl == "wcm" or (nl == "dan17" and not refit17)
                        row.update(NC.alpha_stats(circuit_alphas(c), inert_ref if stored else inert, MOVE_TOL))
                    row["set"] = NC.set_profile(sets[(nl, d)], prof, WIN)
                    if nl != "wcm":                            # drawn production members (members are in the pool)
                        ov = sum(len(set(v) & set(ref_sites.get(s, ()))) for s, v in sets[(nl, d)].items())
                        nd_ = sum(len(v) for v in sets[(nl, d)].values())
                        row.update(overlap_wcm=ov, overlap_frac=ov / nd_ if nd_ else None)
            except Exception as e:  # noqa: BLE001
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row["secs"] = round(time.time() - t0, 1)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            f = lambda x: "%.2f" % x if isinstance(x, float) else "-"
            log("%3d/%d %-16s %-10s d%d %-5s n %s/%s  Z %s A %s C %s nec %s  moved %s  %s  %.0fs" % (
                n_ + 1, len(keys), key, arm_, d, split, row.get("n"), row["n_ref"], f(row.get("free0_tk")),
                f(row.get("freeM_topk_tk")), f(row.get("freeN_topk_tk")), f(row.get(NC.NEC)), f(row.get("alpha_moved")),
                row.get("error") or row.get("skip") or ("PASS" if row.get("pass") else "fail"), row["secs"]))
        log("%3d/%d %-16s target done in %.0fs (%.0fs since start)" % (
            n_ + 1, len(keys), key, time.time() - t_target, time.time() - t_start))
    log("DONE")


if __name__ == "__main__":
    main()
