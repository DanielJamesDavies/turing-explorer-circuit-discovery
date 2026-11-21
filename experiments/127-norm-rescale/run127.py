"""127 norm rescale (TuringLLM, WCM v4 production circuits): do the fitted amplitudes alpha mostly restore the site
norm that the circuit-only ablation removes?

Everything is scored through exp 118's reduced scorer (run118.Ctx: 049 V.read + the engine CircuitOnlyPatcher with
keep_scales, the same contexts, means and empty baselines; checked 216 / 216 against the full production scorer in
118). Production circuits and context records are read only (T3 / T2 archives, via run118's paths). Nothing in src/
is changed: the rescale is a subclass of 118's TapCO patcher (the engine patcher) defined in this file.

ARMS (env N127_ARMS, comma list; each scored on Z / A / C, held-out "ho" (16 rows) and training "home" (48 rows)):
  alpha1       every member amplitude = 1 (118's alpha = 1 variant).
  fitted       the production alpha (118's red_fitted).
  oracle       alpha = 1, then at every member site and every (row, position) the site's latent reconstruction
               xhat_edit = decode(patched code) (decoder bias included; for A / C that includes the Top-K mean fill) is
               scaled so ||s * xhat_edit|| = ||xhat_nat|| of the same row and position of a natural run. Natural
               information, so an upper bound for "norm repair". The SAE error term is NOT scaled (see below).
  oracle_lat   as oracle, but only the latent part W_dec z is scaled and the decoder bias b is kept:
               ||s W z + b|| = ||xhat_nat|| (positive root; s = max(0, -(Wz.b)/|Wz|^2) when no root exists).
  rescale_site alpha = 1, one scalar per (circuit, member site): the median over the TRAINING rows (anchor) and the
               three fills of ||xhat_nat|| / ||xhat_edit||, applied unchanged to held-out. Calibrated in N127_SITE_ITERS
               [3] passes (each pass measures every site under the previous pass's upstream scalars), so downstream
               scalars see rescaled upstream sites. A cheap norm-matched reference, not optimised (118 job 3 refitted
               per-site / per-circuit scalars on the pass-rule objective itself: 37% / 33% vs 85%).

Hook point and SAE error. The engine's CircuitOnlyPatcher.transform returns decode(patched) + error with
error = x - decode(encode(x)) of the live (already edited) stream, preserved in all three reads (Z, A, C; the default
keep_error_sites = None). The Resc patcher below calls that transform unchanged, recomputes the same error
(error = x - decode(encode(x)), as 118's capture did), takes xhat_edit = out - error, and returns
s * xhat_edit + error (oracle_lat: s * (xhat_edit - b) + b + error). So the error is untouched in every read, as in the
pipeline; only the latent reconstruction is rescaled. alpha1 / fitted return the engine output itself (bit-identical
to 118's reduced scorer); the capture there only reads.

DIAGNOSTIC (ho rows, every arm, every fill): at every RMSNorm input (norm_1 / norm_2 of each block) after the
circuit's earliest member site and before the target's read, plus the target SAE's own input x at the seed site: at the
anchor, cosine of the post-norm vector to the natural one, ||post|| / ||post_nat||, ||pre|| / ||pre_nat||; and per
member site the reconstruction norm ratio ||xhat_out|| / ||xhat_nat|| at the anchor (118 section 4 measured it for
alpha1 / fitted: median 0.53 / 0.91 in the Z run).

Necessity (phi_sup_blind: members SET to 0, alpha-independent) and vacuity come from 118's own full score
(118 out/rows.jsonl, same GPU, same circuits); a target missing there is scored with the full scorer here.
Home rows use the ho necessity (no training-row necessity exists).

  wsl: cd <repo>; PYTHONPATH=src .venv/bin/python -X utf8 experiments/127-norm-rescale/run127.py --targets FILE --out DIR
  env: N127_ARMS (alpha1,fitted,oracle,oracle_lat)  N127_SITE_ITERS (3)  N127_LIMIT  N127_STOP_EPOCH (or --stop-epoch:
       no new target after it)
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E118 = ROOT / "experiments" / "118-coefficient-diagnostics"
sys.path.insert(0, str(E118))
import run118 as X  # noqa: E402  (sets 118's env: MODE=main, WCM_VERSION=v4, CTX_DIR = T2 records; imports driver)

import numpy as np  # noqa: E402
import torch  # noqa: E402

driver = X.driver
ALL_ARMS = ("alpha1", "fitted", "oracle", "oracle_lat", "rescale_site")
FILLS = ("Z", "A", "C")
KORD = {"attn": 0, "mlp": 1, "resid": 2}


def jfloat(v):
    return X.jfloat(v)


def site_ord(layer, kind):
    """Forward order of the hook sites: norm_1(l) 3l - 0.5, attn 3l, norm_2(l) 3l + 0.5, mlp 3l + 1, resid 3l + 2."""
    return 3 * layer + KORD[kind]


def ln_points(C):
    """RMSNorm inputs strictly after the earliest member site and before the target's read."""
    m0 = min(site_ord(*s) for s in C.sites)
    t = site_ord(C.layer, C.kind)
    pts = []
    for l in range(C.layer + 1):
        for which, o in (("n1", 3 * l - 0.5), ("n2", 3 * l + 0.5)):
            if m0 < o < t:
                pts.append((l, which))
    return pts


def own_read(C):
    """The norm whose output the target's site reads (attn: norm_1 of its layer; mlp and resid: norm_2)."""
    return (C.layer, "n1" if C.kind == "attn" else "n2")


@contextlib.contextmanager
def ln_hooks(model, pts, fn):
    hs = []
    try:
        for l, which in pts:
            mod = getattr(model.transformer.h[l], "norm_1" if which == "n1" else "norm_2")
            hs.append(mod.register_forward_pre_hook(lambda m, inp, key=(l, which): fn(m, inp[0], key)))
        yield
    finally:
        for h in hs:
            h.remove()


# ------------------------------------------------------------------------------------------------ natural pass
class NatTap:
    """Natural run (no edit): per member site ||xhat|| = ||decode(encode(x))|| at every (row, position); at the anchor
    the norm inputs / outputs on the LN points and the seed site's input x."""

    def __init__(self, C, anchors, pts):
        self.C, self.pos_argmax, self.pts = C, anchors, pts
        self.nrm, self.ln, self.seed_x = {}, {}, None

    def __call__(self, model):
        from model.hooks import multi_patch

        @contextlib.contextmanager
        def cm():
            with ln_hooks(model, self.pts, self._ln), multi_patch(model, self.transform):
                yield
        return cm()

    def _anc(self, x):
        anc = self.pos_argmax.to(x.device).clamp(0, x.shape[1] - 1)
        return torch.arange(x.shape[0], device=x.device), anc

    def _ln(self, m, x, key):
        rr, anc = self._anc(x)
        v = x[rr, anc]
        self.ln[key] = (v.float().cpu(), m.forward(v).float().cpu())      # .forward: no hook recursion

    def transform(self, layer_idx, kind, x):
        from sae.dense import sparse_topk_to_dense
        C, s = self.C, (layer_idx, kind)
        if s == (C.layer, C.kind):
            rr, anc = self._anc(x)
            self.seed_x = x[rr, anc].float().cpu()
        if s in C.msets:
            ta, ti = C.bank.encode(x, kind, layer_idx)
            dense = sparse_topk_to_dense(ta, ti, C.D, dtype=x.dtype)
            self.nrm[s] = C.bank.decode(dense, kind, layer_idx).float().norm(dim=-1).cpu()
        return x


def natural_pass(C, pts):
    """All 64 activating rows (training then held-out, the Ctx order)."""
    tok, anc_all = C.pt, C.pa
    bs = C.V.EVAL_BS
    nrm, ln, seedx = defaultdict(list), defaultdict(lambda: ([], [])), []
    C.inf.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(tok.shape[0]), bs):
                p = NatTap(C, anc_all[s0:s0 + bs], pts)
                C.inf.forward(tok[s0:s0 + bs], patcher=p, grad_enabled=False, return_activations=False,
                              tokenize_final=False)
                for s, v in p.nrm.items():
                    nrm[s].append(v)
                for k, (a, b) in p.ln.items():
                    ln[k][0].append(a)
                    ln[k][1].append(b)
                seedx.append(p.seed_x)
    finally:
        C.inf.enable_compile()
    return dict(nrm={s: torch.cat(v, 0) for s, v in nrm.items()},
                ln={k: (torch.cat(a, 0), torch.cat(b, 0)) for k, (a, b) in ln.items()},
                seed_x=torch.cat(seedx, 0) if seedx and seedx[0] is not None else None)


# ------------------------------------------------------------------------------------------------ edited reads
def resc_cls(C):
    """118's TapCO (the engine CircuitOnlyPatcher + the seed tap), with an optional rescale of each member site's
    latent reconstruction and read-only captures. Attributes set after construction (cfg, rows)."""
    if getattr(C, "_resc", None) is not None:
        return C._resc
    from sae.dense import sparse_topk_to_dense
    bank, base = C.bank, C.TapCO

    class Resc(base):
        cfg = None
        rows = None            # global row indices (into C.pt) of this chunk

        def __call__(self, model):
            sup = super().__call__
            cfg = self.cfg

            @contextlib.contextmanager
            def cm():
                with ln_hooks(model, cfg["pts"] if cfg.get("capture") else [], self._ln), sup(model):
                    yield
            return cm()

        def _anc(self, x):
            anc = self.pos_argmax.to(x.device).clamp(0, x.shape[1] - 1)
            return torch.arange(x.shape[0], device=x.device), anc

        def _ln(self, m, x, key):
            rr, anc = self._anc(x)
            v = x[rr, anc]
            self.cfg["cap"]["ln"][key].append((v.float().cpu(), m.forward(v).float().cpu()))

        def transform(self, layer_idx, kind, x):
            out = super().transform(layer_idx, kind, x)
            cfg, s = self.cfg, (layer_idx, kind)
            mode, cap = cfg["mode"], cfg.get("capture")
            if s == (C.layer, C.kind) and cap:
                rr, anc = self._anc(x)
                cfg["cap"]["seed_x"].append(x[rr, anc].float().cpu())
            if s not in C.msets or (mode is None and not cap and not cfg.get("ratio")):
                return out
            with torch.no_grad():
                ta, ti = bank.encode(x, kind, layer_idx)
                dense = sparse_topk_to_dense(ta, ti, C.D, dtype=x.dtype)
                err = x - bank.decode(dense, kind, layer_idx)      # = the engine's preserved error, recomputed
            recon = out - err                                       # decode(patched): the latent reconstruction
            nrm = recon.float().norm(dim=-1)                        # [B, T]
            rows = self.rows
            nat = cfg["nat"][s][rows].to(x.device)                  # [B, T] natural ||xhat||
            rr, anc = self._anc(x)
            if mode is None:
                new = out
                post = nrm
            elif mode == "oracle":
                sc = nat / nrm.clamp(min=1e-6)
                new = recon * sc[..., None].to(recon.dtype) + err
                post = nat
                cfg["cap"]["scale"][s].append(sc[rr, anc].cpu())
            elif mode == "oracle_lat":
                b = cfg["bdec"][s].to(device=x.device, dtype=torch.float32)
                w = recon.float() - b
                ww = (w * w).sum(-1).clamp(min=1e-12)
                wb = (w * b).sum(-1)
                disc = wb * wb - ww * ((b * b).sum() - nat * nat)
                sc = torch.where(disc >= 0, (-wb + disc.clamp(min=0).sqrt()) / ww, (-wb / ww).clamp(min=0))
                lat = w * sc[..., None]
                new = (lat + b).to(recon.dtype) + err
                post = (lat + b).norm(dim=-1)
                cfg["cap"]["scale"][s].append(sc[rr, anc].cpu())
            elif mode == "site":
                sc = float(cfg["site_scale"][s])
                new = recon * sc + err
                post = nrm * sc
            else:
                raise ValueError(mode)
            if cfg.get("ratio"):
                cfg["cap"]["pre_ratio"][s].append((nat[rr, anc] / nrm[rr, anc].clamp(min=1e-6)).cpu())
                cfg["cap"]["post_ratio"][s].append((post[rr, anc] / nat[rr, anc].clamp(min=1e-6)).cpu())
            return new

    C._resc = Resc
    return Resc


def new_cap():
    return dict(ln=defaultdict(list), seed_x=[], scale=defaultdict(list), pre_ratio=defaultdict(list),
                post_ratio=defaultdict(list))


def co(C, cfg, scales, fill, which):
    """One circuit-only read (fill Z / A / C) on the ho or training rows under cfg; returns {"pre", "tk"}."""
    mm, tk = C.fills[fill]
    n_tr = int(C.pt_tr.shape[0])
    tokens, anchors, row0 = (C.pt_ho, C.pa_ho, n_tr) if which == "ho" else (C.pt_tr, C.pa_tr, 0)
    cls = resc_cls(C)

    def mk(a_, b_):
        p = C.patcher(C.keep, anchors[a_:b_], mm, tk, scales, cls=cls)
        p.cfg = cfg
        p.rows = torch.arange(row0 + a_, row0 + b_)
        return p
    return C.V.read(mk, tokens, anchors)


def score_arm(C, cfg_base, scales, ho_capture=True):
    """Z / A / C on ho and home for one arm; ho runs carry the diagnostics. Returns (scores, diag)."""
    res, diag = {"ho": {}, "home": {}}, {}
    for which in ("ho", "home"):
        E = C.E_ho if which == "ho" else C.E_tr
        a = C.a_ho if which == "ho" else C.a_tr
        for f in FILLS:
            cfg = dict(cfg_base, capture=(which == "ho" and ho_capture), ratio=(which == "ho" and ho_capture),
                       cap=new_cap())
            r = co(C, cfg, scales, f, which)
            res[which][f] = C.frac(r["tk"], E[f]["tk"], a["tk"])
            if which == "ho" and ho_capture:
                diag[f] = cfg["cap"]
    return res, diag


def calibrate_site(C, nat, bdec, iters):
    """rescale_site scalars: per member site, the median over training rows (anchor) and the three fills of
    ||xhat_nat|| / ||xhat_edit||, measured under the previous pass's scalars at the upstream sites."""
    ones = C.scales_of(np.ones(C.n_mem))
    sc = {s: 1.0 for s in C.sites}
    hist = []
    for it in range(iters):
        pooled = defaultdict(list)
        for f in FILLS:
            cfg = dict(mode="site", site_scale=dict(sc), nat=nat, bdec=bdec, pts=[], capture=False, ratio=True,
                       cap=new_cap())
            co(C, cfg, ones, f, "tr")
            for s, v in cfg["cap"]["pre_ratio"].items():
                pooled[s].append(torch.cat(v))
        new = {s: float(torch.cat(pooled[s]).median()) for s in C.sites}
        hist.append(float(np.median([abs(math.log(max(new[s], 1e-9) / max(sc[s], 1e-9))) for s in C.sites])))
        sc = new
    return sc, hist


# ------------------------------------------------------------------------------------------------ summaries
def q(x, p):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if len(x) else None


def diag_summary(C, natd, diag, pts, n_tr):
    """Per fill: LN points (cos post, post norm ratio, pre norm ratio; medians over ho rows), the target read, and
    the member-site recon ratio at the anchor (pre / post the rescale), the oracle scale distribution."""
    out = {}
    own = own_read(C)
    for f, cap in diag.items():
        d = {"ln": {}}
        for key in pts:
            if key not in cap["ln"] or key not in natd["ln"]:
                continue
            pre = torch.cat([a for a, _ in cap["ln"][key]])
            post = torch.cat([b for _, b in cap["ln"][key]])
            npre, npost = natd["ln"][key][0][n_tr:], natd["ln"][key][1][n_tr:]
            cos = torch.nn.functional.cosine_similarity(post, npost, dim=-1)
            rp = post.norm(dim=-1) / npost.norm(dim=-1).clamp(min=1e-9)
            rq = pre.norm(dim=-1) / npre.norm(dim=-1).clamp(min=1e-9)
            d["ln"]["%d.%s" % key] = dict(cos=float(cos.median()), cos_p10=q(cos.numpy(), 10),
                                          post_ratio=float(rp.median()), pre_ratio=float(rq.median()))
        k_own = "%d.%s" % own
        d["own"] = d["ln"].get(k_own)
        d["own_key"] = k_own if k_own in d["ln"] else None
        if cap["seed_x"] and natd["seed_x"] is not None:
            sx = torch.cat(cap["seed_x"])
            nx = natd["seed_x"][n_tr:]
            cs = torch.nn.functional.cosine_similarity(sx, nx, dim=-1)
            d["seed_x"] = dict(cos=float(cs.median()), cos_p10=q(cs.numpy(), 10),
                               ratio=float((sx.norm(dim=-1) / nx.norm(dim=-1).clamp(min=1e-9)).median()))
        pre_r = np.concatenate([torch.cat(v).numpy() for v in cap["pre_ratio"].values()]) if cap["pre_ratio"] else []
        post_r = np.concatenate([torch.cat(v).numpy() for v in cap["post_ratio"].values()]) if cap["post_ratio"] else []
        # 118's measure: edited / natural recon norm = 1 / pre_ratio before any rescale; post_ratio after it
        d["recon_ratio_med"] = q(post_r, 50)
        d["recon_ratio_p90"] = q(post_r, 90)
        d["recon_ratio_unscaled_med"] = q(1.0 / np.maximum(np.asarray(pre_r, dtype=float), 1e-9), 50) if len(pre_r) else None
        if cap["scale"]:
            sc = np.concatenate([torch.cat(v).numpy() for v in cap["scale"].values()])
            d["scale"] = dict(med=q(sc, 50), p10=q(sc, 10), p90=q(sc, 90), p99=q(sc, 99), max=float(np.max(sc)))
        out[f] = d
    return out


# ------------------------------------------------------------------------------------------------ one target
def load_ref():
    ref = {}
    p = E118 / "out" / "rows.jsonl"
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            if "error" not in r:
                ref[r["seed"]] = r
    return ref


def one(R, key, ref, arms, site_iters):
    t0 = time.time()
    c = torch.load(X.OUT_V4 / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
    rec = R.contexts(key)
    row = dict(seed=key, n=driver.n_nodes(c))
    r118 = ref.get(key)
    if r118 is None:
        full = R.score(c, rec, "strong")
        row["ref_source"] = "full scorer (this run)"
        get = full.get
    else:
        row["ref_source"] = "118 out/rows.jsonl"
        get = lambda k: r118.get("full_" + k)  # noqa: E731
    for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "a_pos_tk", "e0_tk", "eM_topk_tk",
              "eN_topk_tk"):
        row["full_" + k] = get(k)
    C = X.Ctx(R, c, rec)
    n_tr = int(C.pt_tr.shape[0])
    row.update(n_sites=len(C.sites), n_tr=n_tr, n_ho=int(C.pt_ho.shape[0]), t_ctx=round(time.time() - t0, 1))
    pts = ln_points(C)
    natd = natural_pass(C, pts)
    nat = natd["nrm"]
    bdec = {}
    for s in C.sites:
        with torch.no_grad():
            bdec[s] = C.bank.decode(torch.zeros(1, C.D, device=C.dev), s[1], s[0]).float().reshape(-1).cpu()
    nat_anchor = {s: nat[s][torch.arange(len(C.pa)), C.pa.cpu().clamp(0, nat[s].shape[1] - 1)] for s in C.sites}
    row["bdec_share_med"] = float(np.median([float(bdec[s].norm()) / max(float(nat_anchor[s].median()), 1e-9)
                                             for s in C.sites]))
    row["ln_points"] = ["%d.%s" % p for p in pts]
    row["own_read"] = "%d.%s" % own_read(C)
    row["t_nat"] = round(time.time() - t0, 1)
    base = dict(nat=nat, bdec=bdec, pts=pts)
    ones = C.scales_of(np.ones(C.n_mem))
    fitted = C.scales_of(C.alpha_fit)
    row["arms"], row["diag"], row["t_arm"] = {}, {}, {}
    for arm in arms:
        ta = time.time()
        if arm == "alpha1":
            res, dg = score_arm(C, dict(base, mode=None), ones)
        elif arm == "fitted":
            res, dg = score_arm(C, dict(base, mode=None), fitted)
        elif arm == "oracle":
            res, dg = score_arm(C, dict(base, mode="oracle"), ones)
        elif arm == "oracle_lat":
            res, dg = score_arm(C, dict(base, mode="oracle_lat"), ones)
        elif arm == "rescale_site":
            sc, hist = calibrate_site(C, nat, bdec, site_iters)
            res, dg = score_arm(C, dict(base, mode="site", site_scale=sc), ones)
            v = np.array(list(sc.values()))
            row["site_scale"] = {"%d.%s" % s: x for s, x in sc.items()}
            row["site_scale_stats"] = dict(med=float(np.median(v)), p10=q(v, 10), p90=q(v, 90), min=float(v.min()),
                                           max=float(v.max()), iter_change=hist)
        else:
            raise ValueError(arm)
        row["arms"][arm] = res
        row["diag"][arm] = diag_summary(C, natd, dg, pts, n_tr)
        row["t_arm"][arm] = round(time.time() - ta, 1)
    if r118 is not None:
        chk = {}
        if "alpha1" in row["arms"]:
            chk["alpha1"] = max(abs(row["arms"]["alpha1"]["ho"][f] - r118["var"]["alpha1"][f]) for f in FILLS
                                if row["arms"]["alpha1"]["ho"][f] is not None and r118["var"]["alpha1"][f] is not None)
        if "fitted" in row["arms"]:
            chk["fitted"] = max(abs(row["arms"]["fitted"]["ho"][f] - r118["red_fitted"][f]) for f in FILLS
                                if row["arms"]["fitted"]["ho"][f] is not None and r118["red_fitted"][f] is not None)
        row["check_vs_118"] = chk
    row["device"] = str(C.dev)
    row["t_total"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=os.environ.get("N127_ARMS", "alpha1,fitted,oracle,oracle_lat"))
    ap.add_argument("--stop-epoch", type=float, default=float(os.environ.get("N127_STOP_EPOCH") or 0) or None)
    ap.add_argument("--limit", type=int, default=int(os.environ["N127_LIMIT"]) if os.environ.get("N127_LIMIT") else None)
    ap.add_argument("--site-iters", type=int, default=int(os.environ.get("N127_SITE_ITERS", "3")))
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    bad = [x for x in arms if x not in ALL_ARMS]
    if bad:
        sys.exit("unknown arms %s" % bad)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_p = out / "rows.jsonl"
    targets = [t for t in Path(a.targets).read_text().split() if t]
    done = set()
    if rows_p.exists():
        for line in open(rows_p):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add(r["seed"])
            except Exception:  # noqa: BLE001
                pass
    todo = [t for t in targets if t not in done]
    if a.limit:
        todo = todo[:a.limit]
    print("127: %d targets to do (%d done) | arms %s | out %s | stop epoch %s" % (
        len(todo), len(done), arms, out, a.stop_epoch), flush=True)
    if not todo:
        print("DONE", flush=True)
        return
    R = driver.Runner()
    ref = load_ref()
    with open(rows_p, "a") as fh:
        for n_, key in enumerate(todo):
            if a.stop_epoch and time.time() >= a.stop_epoch:
                print("STOP_EPOCH reached: no new target (%d left)" % (len(todo) - n_), flush=True)
                sys.exit(7)
            try:
                row = one(R, key, ref, arms, a.site_iters)
            except Exception as e:  # noqa: BLE001
                row = dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))
                traceback.print_exc()
            fh.write(json.dumps(row, default=jfloat) + "\n")
            fh.flush()
            ar = row.get("arms", {})
            print("%3d/%d %-16s %s n=%s sites=%s chk=%s %s  %ss" % (
                n_ + 1, len(todo), key, row.get("error", "ok"), row.get("n"), row.get("n_sites"),
                row.get("check_vs_118"),
                " ".join("%s=%s" % (k, "/".join("-" if v["ho"][f] is None else "%.2f" % v["ho"][f] for f in FILLS))
                         for k, v in ar.items()), row.get("t_total")), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
