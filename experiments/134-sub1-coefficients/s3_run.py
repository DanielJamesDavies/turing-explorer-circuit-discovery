"""134 step 3 (ours): what do WCM members with fitted coefficient alpha < 0.8 do? GPT-2 small, the 2026-11-07 production
circuits (WCM v4 main), rescored without refit on the local GPU. One process, one GPU; resumable per (target, phase).

Reused, not re-implemented: 062 driver.Runner (FAMILY gpt2-small, WCM v4, the production context records from the pod
archive, set up exactly as exp 131 run.py setup_env does) and its score() = src eval.wcm_eval.score_circuit (held-out
strongest; pass = wcm_eval.passes: Z, A, C in [0.75, 1.5] and nec_b >= 0.9; vacuous = wcm_eval.is_vacuous); the donor
test src eval.donor_test.donor_target (pools A / C / R as production, R = the production random pool from the archive,
8 draws, variant "fit"); the circuit-only patcher (wcm_eval.TapCircuitOnly = the engine's CircuitOnlyPatcher) and the
scorer's fills (eval.floors collect_site_anchors / collect_site_means, exactly as score_circuit builds them).

Groups: low = members with alpha < 0.8; control (ctl) = members with alpha in [0.95, 1.05], same circuits.

PHASES (env PHASES, run in this order over the whole target list, each phase finished for every target before the next):
  reads   per member, on the 16 held-out strongest rows (at their anchors) and the 48 contrast-train rows (at each row's
          would-be anchor = argmax of the target's natural pre-activation, as the scorer picks held-out contrast anchors):
          natural run (unmodified model) and the circuit-only runs Z / A / C (members at alpha x live value, every other
          latent at the fill, SAE errors live), each with a backward from the target's pre-activation at the anchor:
            val_at   member value at the anchor (natural code, or the written alpha x re-encoded live value)
            raw_at   circuit runs only: the re-encoded live value before alpha
            val_ms / raw_ms  the same summed over positions <= anchor
            ctb      first-order contribution to the target pre-activation at the anchor: sum over positions <= anchor
                     of value x (d pre / d site stream . decoder column) (gradient x activation)
          plus the natural share of positions where the member is > 0 (contrast rows: all positions; activating rows:
          positions <= anchor), and the maximum decoder cosine to any other member at the same site.
  arms    rescore (no refit) through Runner.score: (a) production alpha; (b) every low member at alpha = 1; (c) every low
          member removed; (d) the same number of random control members removed (np.random.default_rng([SEED,
          crc32(key)]); all control members if there are fewer, flagged)
  sweep   dose-response (added on request): low members' alpha x c for c = 0.5 and 2 (c = 0 / 1 / 1/alpha are arms
          c / a / b); the same control members as (d) x 2 (c = 0 is arm d)
  donor   donor_target over every arm of arms + sweep at once (shared bases / pools / empty draws): recovery A / C / R
          (variant fit), silent firing SF per pool (share of the 16 x 64 silent positions with the post-Top-K target > 0,
          per draw), resample_pass

Outputs (OUT134 [experiments/134-sub1-coefficients/out/s3]): rows.jsonl (one row per (seed, phase); error rows retried
once), members/<key>.npz (reads), current.json (for the job's watchdog), tripped.txt ((target, phase) pairs the
watchdog killed: that phase is skipped for that target; its other phases still run). STOP_EPOCH: no (target, phase) is started when its estimate would end after it (exit 7).

  wsl, repo root: PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -X utf8 -u \
      experiments/134-sub1-coefficients/s3_run.py
  env: PHASES [reads,arms,sweep,donor]  TARGETS [s3_targets.txt]  LIMIT  KEYS  DONOR_DRAWS [8]  STOP_EPOCH
       OUT134  GRAD_BS [2]  LIMIT_HO / LIMIT_CTR (CPU test: rows read in phase reads)  FD_CHECK=1 (CPU test: finite-
       difference check of the natural contributions on the first held-out batch)  FILLS [Z,A,C]
"""
import json
import os
import sys
import time
import traceback
import zlib
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# MODEL134 [gpt2-small] | pythia-70m (step 3b, 2026-11-20): only the archive, the driver FAMILY and the default
# target list / output directory change; every read, arm and score is the same code.
MODEL = os.environ.get("MODEL134", "gpt2-small")
if MODEL not in ("gpt2-small", "pythia-70m"):
    raise SystemExit("MODEL134 %r: gpt2-small | pythia-70m" % MODEL)
PREFIX = "s3" if MODEL == "gpt2-small" else "s3py"
ARCH = ROOT / "experiments" / "109-pod-run-analysis" / "out" / "archives"
PROD_OUT = ARCH / ("pub-%s-S10-results" % MODEL) / "experiments" / "082-public-runs" / ("out_%s" % MODEL)
CTX = PROD_OUT / "ctx"
POOL = PROD_OUT / "donor" / "random_pool.pt"
CIRC = PROD_OUT / "main" / "circuits"
OUT134 = Path(os.environ.get("OUT134", str(HERE / "out" / PREFIX)))
DRV = OUT134 / "driver"
PHASES = tuple(p for p in os.environ.get("PHASES", "reads,arms,sweep,donor").split(",") if p)
STOP_EPOCH = float(os.environ["STOP_EPOCH"]) if os.environ.get("STOP_EPOCH") else None
DRAWS = int(os.environ.get("DONOR_DRAWS", "8"))
GRAD_BS = int(os.environ.get("GRAD_BS", "2"))
LIMIT_HO = int(os.environ["LIMIT_HO"]) if os.environ.get("LIMIT_HO") else None
LIMIT_CTR = int(os.environ["LIMIT_CTR"]) if os.environ.get("LIMIT_CTR") else None
FD_CHECK = os.environ.get("FD_CHECK") == "1"
FILLS = tuple(f for f in os.environ.get("FILLS", "Z,A,C").split(",") if f)
SEED = 20261119
LOW, CTL = 0.8, (0.95, 1.05)
ARMS = ("a_prod", "b_low1", "c_low0", "d_ctl0")
SWEEP = ("low_x0.5", "low_x2", "ctl_x2")
# seconds per (target, phase) on the local card before any is measured; then 1.3 x the phase's running mean
EST0 = {"reads": 240, "arms": 150, "sweep": 120, "donor": 300}


def load_jsonl(path):
    out = []
    if Path(path).exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def targets():
    if os.environ.get("KEYS"):
        keys = [k for k in os.environ["KEYS"].split(",") if k]
    else:
        keys = [k for k in Path(os.environ.get("TARGETS", str(HERE / ("%s_targets.txt" % PREFIX)))).read_text().split()
                if k]
    if os.environ.get("LIMIT"):
        keys = keys[:int(os.environ["LIMIT"])]
    return keys


def tripped():
    """{(key, phase)} the watchdog killed (lines "key phase date why"): only that phase is skipped for the target."""
    p = OUT134 / "tripped.txt"
    out = set()
    if p.exists():
        for l in p.read_text().splitlines():
            w = l.split()
            if len(w) >= 2:
                out.add((w[0], w[1]))
    return out


def setup_env():
    """exp 131 run.py setup_env, with this experiment's driver OUT (only the version markers live there)."""
    DRV.mkdir(parents=True, exist_ok=True)
    for m in ("WCM_VERSION", "WCM_R"):
        if not (DRV / m).exists():
            (DRV / m).write_bytes((PROD_OUT / m).read_bytes())
    os.environ.update(MODE="main", WCM_VERSION="v4", OUT=str(DRV), CTX_DIR=str(CTX), FAMILY=MODEL,
                      SKIP_SPEC="1")
    os.environ.setdefault("TAG", "s134_unused")
    for p in (str(ROOT / "src"), str(ROOT / "experiments" / "062-h100-protocol-v1")):
        if p not in sys.path:
            sys.path.insert(0, p)


# ------------------------------------------------------------------------------------------------ circuit variants
def member_table(c):
    """[(layer, kind, index, alpha, m)] in node order (the seed skipped)."""
    out = []
    for n in c.nodes.values():
        md = n.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out.append((int(f.layer), str(f.kind), int(f.index), float(md.get("amplitude", 1.0)),
                    float(md.get("attribution_score", float("nan")))))
    return out


def groups(c, key):
    """(low keys, control keys, the matched control removal set, short flag)."""
    import numpy as np
    mt = member_table(c)
    low = [(l, k, i) for (l, k, i, a, _) in mt if a < LOW]
    ctl = [(l, k, i) for (l, k, i, a, _) in mt if CTL[0] <= a <= CTL[1]]
    rng = np.random.default_rng([SEED, zlib.crc32(key.encode())])
    n = min(len(low), len(ctl))
    pick = sorted(rng.choice(len(ctl), size=n, replace=False).tolist()) if n else []
    return low, ctl, [ctl[j] for j in pick], len(ctl) < len(low)


def variant(c, remove=(), scale=None, set_to=None):
    """A circuit-like object (nodes, metadata) for the scorer and the donor test: members in `remove` dropped, members
    in `scale` {key: c} at alpha x c, members in `set_to` {key: alpha} at that alpha; the seed node and every other
    member unchanged (node order kept)."""
    remove, scale, set_to = set(remove), scale or {}, set_to or {}
    nodes = {}
    for nid, n in c.nodes.items():
        md = n.metadata
        if md.get("role") == "seed":
            nodes[nid] = n
            continue
        f = md["feature_id"]
        k = (int(f.layer), str(f.kind), int(f.index))
        if k in remove:
            continue
        if k in scale or k in set_to:
            md = dict(md)
            md["amplitude"] = set_to[k] if k in set_to else float(md.get("amplitude", 1.0)) * scale[k]
            nodes[nid] = SimpleNamespace(metadata=md)
        else:
            nodes[nid] = n
    return SimpleNamespace(nodes=nodes, metadata=getattr(c, "metadata", {}) or {})


def arm_circuits(c, key, which):
    low, ctl, ctl_rm, short = groups(c, key)
    out = {}
    for arm in which:
        if arm == "a_prod":
            out[arm] = c
        elif arm == "b_low1":
            out[arm] = variant(c, set_to={k: 1.0 for k in low})
        elif arm == "c_low0":
            out[arm] = variant(c, remove=low)
        elif arm == "d_ctl0":
            out[arm] = variant(c, remove=ctl_rm)
        elif arm == "low_x0.5":
            out[arm] = variant(c, scale={k: 0.5 for k in low})
        elif arm == "low_x2":
            out[arm] = variant(c, scale={k: 2.0 for k in low})
        elif arm == "ctl_x2":
            out[arm] = variant(c, scale={k: 2.0 for k in ctl_rm})
        else:
            raise ValueError(arm)
    return out, dict(n_low=len(low), n_ctl=len(ctl), n_ctl_removed=len(ctl_rm), ctl_short=short)


# ------------------------------------------------------------------------------------------------ member reads
def make_patchers():
    import torch
    from eval.wcm_eval import NaturalTap, TapCircuitOnly, _tap
    from sae.dense import sparse_topk_to_dense

    def capture(bank, x, kind, layer, idx):
        with torch.no_grad():
            ta, ti = bank.encode(x.detach(), kind, layer)
            return sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=torch.float32)[..., idx].clone()

    class NatGrad(NaturalTap):
        """The unmodified model; at every member site captures the members' natural codes and adds a zero leaf to the
        stream (its gradient = d target pre / d site stream). perturb {(site, j): eps}: the member j's decoded
        contribution scaled by (1 - eps) in the live stream (finite-difference check)."""

        def __init__(self, bank, seed_site, w, b, sl, msets, perturb=None):
            super().__init__(bank, seed_site, w, b, sl)
            self.msets, self.val, self.leaf, self.perturb = msets, {}, {}, perturb or {}

        def transform(self, layer_idx, kind, x):
            s = (layer_idx, kind)
            if s == self.seed_site:
                self.seed_pre = _tap(self, self.bank, x, s)
                return x
            if s in self.msets:
                v = capture(self.bank, x, kind, layer_idx, self.msets[s])
                self.val[s] = v
                for (ps, j), eps in self.perturb.items():
                    if ps == s:
                        col = self.bank.saes[kind][layer_idx].decoder.weight[:, self.msets[s][j]].to(x.dtype)
                        x = x - eps * v[..., j:j + 1].to(x.dtype) * col
                if torch.is_grad_enabled():
                    leaf = torch.zeros_like(x, requires_grad=True)
                    self.leaf[s] = leaf
                    return x + leaf
            return x

    class COGrad(TapCircuitOnly):
        """The scorer's circuit-only patcher; at every member site captures the members' re-encoded live values (before
        alpha) and adds a zero leaf to the edited stream."""

        def __init__(self, *a, msets_=None, **k):
            super().__init__(*a, **k)
            self.msets_, self.raw, self.leaf = msets_, {}, {}

        def transform(self, layer_idx, kind, x):
            out = super().transform(layer_idx, kind, x)
            s = (layer_idx, kind)
            if s in self.msets_:
                self.raw[s] = capture(self.bank, x, kind, layer_idx, self.msets_[s])
                leaf = torch.zeros_like(out, requires_grad=True)
                self.leaf[s] = leaf
                out = out + leaf
            return out

    return NatGrad, COGrad


def member_reads(R, c, rec, key):
    import contextlib
    import io
    import numpy as np
    import torch
    from circuit.protocol_contexts import eval_probe
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_anchors, collect_site_means
    from eval.wcm_eval import NaturalTap, contrast_split, site_encoder
    G = R.G
    inf, bank, dev = G["inference"], G["bank"], G["device"]
    K, D = int(bank.k), int(bank.d_sae)
    NatGrad, COGrad = make_patchers()
    l_, kd_, sl = key.split(".")
    layer, kind, sl = int(l_), kd_, int(sl)
    site = (layer, kind)
    W, b = site_encoder(bank, layer, kind)
    w_seed, b_seed = W[sl], b[sl]
    UPS = set(upstream_sites(bank, layer, kind))
    mt = member_table(c)
    sites = sorted({(l, k) for (l, k, _, _, _) in mt})
    msets = {s: torch.tensor(sorted(i for (l, k, i, _, _) in mt if (l, k) == s), device=dev, dtype=torch.long)
             for s in sites}
    amap = {(l, k, i): a for (l, k, i, a, _) in mt}
    mmap = {(l, k, i): m for (l, k, i, _, m) in mt}
    order = [(s[0], s[1], int(i)) for s in sites for i in msets[s].tolist()]      # flat member order (site, index)
    n_mem = len(order)
    assert n_mem == len(mt)
    scales = {}
    for s in sites:
        sv = torch.ones(D, device=dev, dtype=torch.float32)
        sv[msets[s]] = torch.tensor([amap[(s[0], s[1], int(i))] for i in msets[s].tolist()], device=dev)
        scales[s] = sv
    keep = {s: set(msets[s].tolist()) for s in sites}
    dec = {s: bank.saes[s[1]][s[0]].decoder.weight.detach().float()[:, msets[s]] for s in sites}   # [d_in, n_s]

    probe = eval_probe(rec, "strong", dev)
    pt, pa = probe.pos_tokens[:64], probe.pos_argmax[:64]
    nt = probe.neg_tokens[:64]
    n_tr = int(probe.metadata["n_train"])
    n_ntr = contrast_split(int(nt.shape[0]), probe.metadata)
    pt_tr, pa_tr = pt[:n_tr], pa[:n_tr]
    pt_ho, pa_ho = pt[n_tr:], pa[n_tr:]
    nt_tr = nt[:n_ntr]
    if LIMIT_HO:
        pt_ho, pa_ho = pt_ho[:LIMIT_HO], pa_ho[:LIMIT_HO]
    if LIMIT_CTR:
        nt_tr = nt_tr[:LIMIT_CTR]
    # the scorer's fills (score_circuit: means_tr over the 48 strongest-train rows, means_neg over the contrast-train)
    means_tr, _ = collect_site_anchors(inf, bank, pt_tr, UPS, pa_tr, pin_position_specific=False)
    means_neg = collect_site_means(inf, bank, nt[:n_ntr], UPS)
    fills = {"Z": (None, False), "A": (means_tr, True), "C": (means_neg, True)}
    # contrast would-be anchors: argmax of the natural pre-activation
    pres = []
    inf.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(nt_tr.shape[0]), 16):
                p0 = NaturalTap(bank, site, w_seed, b_seed, sl)
                inf.forward(nt_tr[s0:s0 + 16], patcher=p0, grad_enabled=False, return_activations=False,
                            tokenize_final=False)
                pres.append(p0.seed_pre.detach().float())
    finally:
        inf.enable_compile()
    na_tr = torch.cat(pres).argmax(dim=1)

    def run(tokens, anchors, mode):
        """mode 'nat' or a fill name: per-row [B, n_mem] arrays."""
        acc = {k: [] for k in ("val_at", "val_ms", "raw_at", "raw_ms", "ctb", "fire", "pre_at")}
        inf.disable_compile()
        try:
            for s0 in range(0, int(tokens.shape[0]), GRAD_BS):
                tk = tokens[s0:s0 + GRAD_BS]
                anc = anchors[s0:s0 + GRAD_BS].to(dev)
                B, T = int(tk.shape[0]), int(tk.shape[1])
                if mode == "nat":
                    p = NatGrad(bank, site, w_seed, b_seed, sl, msets)
                else:
                    mm, topk = fills[mode]
                    p = COGrad(bank=bank, keep_indices=keep, in_scope=UPS, seed_layer=layer, seed_kind=kind,
                               seed_latent_idx=sl, pos_argmax=anc, site_means=mm, pin_values=None, respect_topk=topk,
                               topk=K, keep_tensors=msets, keep_scales=scales, w_seed=w_seed, b_seed=b_seed,
                               msets_=msets)
                with torch.enable_grad(), contextlib.redirect_stdout(io.StringIO()):
                    inf.forward(tk, patcher=p, grad_enabled=True, return_activations=False, tokenize_final=False)
                    rr = torch.arange(B, device=dev)
                    pre_at = p.seed_pre[rr, anc.clamp(0, T - 1)]
                    leaves = [p.leaf[s] for s in sites]
                    grads = torch.autograd.grad(pre_at.sum(), leaves, allow_unused=True)
                pm = (torch.arange(T, device=dev)[None, :] <= anc[:, None]).float()        # [B, T]
                parts = {k: [] for k in acc if k != "pre_at"}
                for s, g in zip(sites, grads):
                    if mode == "nat":
                        raw = p.val[s]
                        val = raw
                    else:
                        raw = p.raw[s]
                        val = raw * scales[s][msets[s]][None, None, :]
                    proj = (g.float() @ dec[s]) if g is not None else torch.zeros_like(val)  # [B, T, n_s]
                    parts["ctb"].append((val * proj * pm[..., None]).sum(1))
                    parts["val_at"].append(val[rr, anc])
                    parts["val_ms"].append((val * pm[..., None]).sum(1))
                    parts["raw_at"].append(raw[rr, anc])
                    parts["raw_ms"].append((raw * pm[..., None]).sum(1))
                    parts["fire"].append(((val > 0).float() * pm[..., None]).sum(1) / pm.sum(1, keepdim=True))
                for k_, v in parts.items():
                    acc[k_].append(torch.cat(v, 1).detach().cpu())
                acc["pre_at"].append(pre_at.detach().float().cpu())
                del p, grads, leaves, pre_at
        finally:
            inf.enable_compile()
        return {k: torch.cat(v, 0).numpy().astype(np.float32) for k, v in acc.items()}

    arrs = {}
    t = {}
    full = torch.ones_like(na_tr) * (int(nt_tr.shape[1]) - 1)
    for setname, tokens, anchors in (("ho", pt_ho, pa_ho), ("ctr", nt_tr, na_tr)):
        for mode in ("nat",) + FILLS:
            ts = time.time()
            r = run(tokens, anchors, mode)
            t["%s_%s" % (setname, mode)] = round(time.time() - ts, 1)
            for k_, v in r.items():
                if mode == "nat" and k_ in ("raw_at", "raw_ms"):
                    continue
                arrs["%s_%s_%s" % (setname, mode, k_)] = v
    # natural firing share on the contrast rows at ALL positions (selectivity), not only <= the would-be anchor
    r = run(nt_tr, full, "nat")
    arrs["ctr_nat_fire_all"] = r["fire"]
    arrs["ctr_anchor"] = na_tr.cpu().numpy()
    arrs["ho_anchor"] = pa_ho.cpu().numpy()
    # read 3: decoder cosine to the nearest other member at the same site
    cos = []
    for s in sites:
        Wm = torch.nn.functional.normalize(dec[s], dim=0)
        C_ = Wm.T @ Wm
        C_.fill_diagonal_(-2.0)
        cos.append(C_.max(dim=1).values.cpu() if C_.shape[0] > 1 else torch.full((1,), float("nan")))
    arrs["dec_cos_max"] = torch.cat(cos).numpy().astype(np.float32)
    arrs["alpha"] = np.array([amap[k] for k in order], dtype=np.float32)
    arrs["m"] = np.array([mmap[k] for k in order], dtype=np.float32)
    arrs["m_layer"] = np.array([k[0] for k in order], dtype=np.int16)
    arrs["m_kind"] = np.array([k[1] for k in order])
    arrs["m_index"] = np.array([k[2] for k in order], dtype=np.int32)
    low, ctl, ctl_rm, short = groups(c, key)
    rm = set(ctl_rm)
    arrs["ctl_removed"] = np.array([k in rm for k in order])
    info = dict(n_mem=n_mem, n_sites=len(sites), n_ho=int(pt_ho.shape[0]), n_ctr=int(nt_tr.shape[0]), t=t,
                n_low=len(low), n_ctl=len(ctl))
    if FD_CHECK:
        info["fd_check"] = fd_check(R, rec, key, pt_ho[:GRAD_BS], pa_ho[:GRAD_BS], msets, sites, arrs, w_seed, b_seed,
                                    NatGrad, site)
    return arrs, info


def fd_check(R, rec, key, tok, anc, msets, sites, arrs, w_seed, b_seed, NatGrad, site):
    """Central finite difference of the natural pre-activation at the anchor (summed over the batch) when one
    member's decoded contribution is scaled by 1 -/+ eps, against its gradient x activation (the 4 largest |ctb|)."""
    import contextlib
    import io
    import numpy as np
    import torch
    inf, bank, dev = R.G["inference"], R.G["bank"], R.G["device"]
    sl = int(key.split(".")[2])
    ctb = arrs["ho_nat_ctb"][:int(tok.shape[0])].sum(0)
    off = np.cumsum([0] + [len(msets[s]) for s in sites])
    out = []
    for j in np.argsort(-np.abs(ctb))[:4]:
        si = int(np.searchsorted(off, j, side="right") - 1)
        s, jj = sites[si], int(j - off[si])
        vals = []
        for eps in (0.05, -0.05):
            p = NatGrad(bank, site, w_seed, b_seed, sl, msets, perturb={(s, jj): eps})
            inf.disable_compile()
            try:
                with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                    inf.forward(tok, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
            finally:
                inf.enable_compile()
            rr = torch.arange(int(tok.shape[0]), device=dev)
            vals.append(float(p.seed_pre[rr, anc.to(dev)].sum()))
        fd = -(vals[0] - vals[1]) / 0.1            # d pre / d(scale) at scale 1 = contribution
        out.append(dict(member=int(j), site="%d.%s" % s, ctb=float(ctb[j]), fd=fd))
    return out


# ------------------------------------------------------------------------------------------------ scoring
def score_arms(R, c, rec, key, which):
    from eval.wcm_eval import is_vacuous, passes
    circs, info = arm_circuits(c, key, which)
    out = {}
    for arm, cc in circs.items():
        ts = time.time()
        p = R.score(cc, rec, "strong")
        sc = {k: v for k, v in p.items() if v is None or isinstance(v, (bool, int, float, str))}
        out[arm] = dict(prod=sc, pass_prod=bool(passes(p)), vacuous=bool(is_vacuous(p)), n=sum(
            1 for n in cc.nodes.values() if n.metadata.get("role") != "seed"), t=round(time.time() - ts, 1))
    return out, info


def donor_arms(R, c, rec, key, pool, prod_rows):
    import eval.donor_test as DT
    circs, info = arm_circuits(c, key, ARMS + SWEEP)
    drows, _ = DT.donor_target(R.G, key, rec, circs, pool, variants=("fit",), draws=DRAWS)
    out = {}
    for arm, d in drows.items():
        sf = d["silent_firing"]["fit"]
        out[arm] = dict(recovery=d["recovery"]["fit"], recovery_pre=d["recovery_pre"]["fit"],
                        sf={ds: (sum(v) / len(v) if v else None) for ds, v in sf.items()},
                        sf_draws=sf, sf_empty={ds: (sum(v) / len(v) if v else None)
                                               for ds, v in d["silent_firing"]["empty"].items()},
                        empty_leak=d["empty_leak"], natural_mean=d["natural_mean"], n=d["n"],
                        resample_pass=(bool(DT.resample_pass(prod_rows[arm], d["recovery"]["fit"]))
                                       if arm in prod_rows else None))
    return out, info


# ------------------------------------------------------------------------------------------------ main
def main():
    keys = targets()
    rows = load_jsonl(OUT134 / "rows.jsonl")
    done = {(r["seed"], r["phase"]) for r in rows if "error" not in r}
    errs = {}
    for r in rows:
        if "error" in r:
            errs[(r["seed"], r["phase"])] = errs.get((r["seed"], r["phase"]), 0) + 1
    todo = [(ph, k) for ph in PHASES for k in keys if (k, ph) not in done and errs.get((k, ph), 0) < 2]
    print("134 s3: %d targets, phases %s, %d (target, phase) to do, %d done; draws %d; grad bs %d; stop %s; out %s" % (
        len(keys), PHASES, len(todo), len(done), DRAWS, GRAD_BS,
        time.strftime("%a %H:%M", time.localtime(STOP_EPOCH)) if STOP_EPOCH else "none", OUT134), flush=True)
    if not todo:
        print("134 s3: DONE (nothing to do)")
        return 0
    for p in (CTX, POOL, CIRC, PROD_OUT / "WCM_VERSION"):
        if not p.exists():
            raise SystemExit("missing %s" % p)
    setup_env()
    import numpy as np
    import torch
    import driver  # noqa: E402
    assert driver.FAMILY == MODEL and driver.WCM_VERSION == "v4"
    R = driver.Runner()
    pool = torch.load(POOL, weights_only=False, map_location="cpu")["tokens"]
    OUT134.mkdir(parents=True, exist_ok=True)
    (OUT134 / "members").mkdir(exist_ok=True)
    prod_by = {}                                   # (seed) -> {arm: prod row} from the arms / sweep phases
    for r in rows:
        if "error" not in r and r["phase"] in ("arms", "sweep"):
            for arm, v in r["arms"].items():
                prod_by.setdefault(r["seed"], {})[arm] = v["prod"]
    fh = open(OUT134 / "rows.jsonl", "a")
    cur = OUT134 / "current.json"
    dur = {}
    t0 = time.time()
    rc = 0
    for i, (ph, key) in enumerate(todo):
        if (key, ph) in tripped():
            continue
        est = 1.3 * float(np.mean(dur[ph])) if dur.get(ph) else EST0[ph]
        if STOP_EPOCH and time.time() + est >= STOP_EPOCH:
            print("134 s3: STOP_EPOCH: %s %s (est %.0f s) not started; the rest of the queue is cut" % (ph, key, est),
                  flush=True)
            rc = 7
            break
        cur.write_text(json.dumps(dict(seed=key, phase=ph, pid=os.getpid(), t=time.strftime("%F %T"))) + "\n")
        ts = time.time()
        row = dict(seed=key, phase=ph, layer=int(key.split(".")[0]), kind=key.split(".")[1])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            rec = R.contexts(key)
            c = torch.load(CIRC / ("%s.pt" % key), weights_only=False)
            if ph == "reads":
                arrs, info = member_reads(R, c, rec, key)
                np.savez_compressed(OUT134 / "members" / ("%s.npz" % key), **arrs)
                row.update(info)
            elif ph in ("arms", "sweep"):
                res, info = score_arms(R, c, rec, key, ARMS if ph == "arms" else SWEEP)
                row.update(info)
                row["arms"] = res
                for arm, v in res.items():
                    prod_by.setdefault(key, {})[arm] = v["prod"]
            elif ph == "donor":
                res, info = donor_arms(R, c, rec, key, pool, prod_by.get(key, {}))
                row.update(info)
                row["donor"] = res
            else:
                raise ValueError("phase %r" % ph)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        row["t"] = round(time.time() - ts, 1)
        row["peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None
        if "error" not in row:
            dur.setdefault(ph, []).append(row["t"])
        fh.write(json.dumps(row) + "\n"); fh.flush()
        msg = ""
        if ph in ("arms", "sweep") and "arms" in row:
            g = lambda v: "-" if not isinstance(v, (int, float)) else "%.2f" % v  # noqa: E731
            msg = " | ".join("%s Z %s A %s C %s nec %s %s" % (
                a, g(v["prod"].get("free0_tk")), g(v["prod"].get("freeM_topk_tk")), g(v["prod"].get("freeN_topk_tk")),
                g(v["prod"].get("phi_sup_blind_tk")), "P" if v["pass_prod"] else "F") for a, v in row["arms"].items())
        elif ph == "donor" and "donor" in row:
            msg = " | ".join("%s SF_C %.3f recA %s" % (a, v["sf"]["C"] or 0, v["recovery"].get("A"))
                             for a, v in row["donor"].items())
        elif ph == "reads" and "n_mem" in row:
            msg = "n %d sites %d low %d ctl %d" % (row["n_mem"], row["n_sites"], row["n_low"], row["n_ctl"])
        print("[%5.0fs] %3d/%d %-6s %-18s %5.0fs peak %s MB %s %s" % (time.time() - t0, i + 1, len(todo), ph, key,
                                                                     row["t"], row["peak_mb"], row.get("error", ""),
                                                                     msg), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cur.write_text(json.dumps(dict(idle=True, pid=os.getpid(), t=time.strftime("%F %T"))) + "\n")
    print("134 s3: %s %.0fs" % ("STOPPED" if rc else "DONE", time.time() - t0), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
