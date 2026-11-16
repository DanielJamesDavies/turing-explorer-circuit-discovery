"""118 coefficient diagnostics (DAN-44), TuringLLM, local GPU. Production circuits are loaded READ-ONLY from the T3
archive; nothing there is written. One jsonl row per target (out/rows.jsonl) and one npz of member-level arrays per
target (out/members/<key>.npz). Resumable; stops cleanly after MAX_MIN minutes (default 85).

Per target:
  1. full    the production scorer (062 driver Runner.score, MODE=main, CTX_DIR = the T2 production records, explicit
             split) at the fitted alpha, on this GPU: necessity, vacuity and the paired reference Z / A / C.
  2. reduced Z / A / C only (free0_tk, freeM_topk_tk, freeN_topk_tk), the same code path as 049 score_circuit
             (V.read + the engine CircuitOnlyPatcher with keep_scales, the same contexts, means and baselines), so any
             coefficient vector can be scored for 3 circuit-only runs. Checked against `full` at the fitted alpha
             (row field red_check = max |diff|). Necessity (phi_sup_blind) does not depend on alpha (members are SET to 0)
             and is taken from `full`.
  3. variants without refit: alpha = 1; fitted alpha clipped to [1/c, c] (c = 1.25, 1.5, 2, 3); fitted alpha capped
             above at 2; each member replaced by its site's / kind's / circuit's median fitted alpha.
  4. refits (members fixed, only the coefficients are fitted, from alpha = 1): one scalar per circuit (global), per
             site kind, per site, and per member (control: the same fitter with the production parameterisation), and
             per member constrained to [0.5, 2] (alpha = 2^tanh(psi)). Fitter: Adam on softplus(psi), loss = sum over the
             three fills (zero, A sparsity-preserving, C sparsity-preserving) of (r - 1)^2, r = the faithfulness ratio of
             the target's relu pre-activation on the 48 strongest-TRAINING contexts (the scorer's reference rows), with
             the scorer's own training means and training empty baselines. Held-out contexts are never seen by a fit.
  5. diagnostics (natural run on all 64 activating rows, plus the fitted-alpha Z and A runs on the 16 held-out rows):
             member codes at the anchor and their max over positions <= anchor; SAE relative reconstruction error at
             the anchor per site; the edited site reconstruction norm vs the natural one; latent natural maxima from
             top_ctx (out/natmax.npy).

  wsl: cd <repo>; PYTHONPATH=src .venv/bin/python -X utf8 experiments/118-coefficient-diagnostics/run118.py
  env: TARGETS (default targets_tl.txt)  MAX_MIN (85)  STEPS (60)  LR (0.05)  FIT_BS (8)  LIMIT  ARMS (comma list of
       refit arms; default all)
"""
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
ARCH = ROOT / "experiments" / "109-pod-run-analysis" / "out" / "archives"
OUT_V4 = ARCH / "pub-turingllm-T3-main" / "root/turing-tl/experiments/094-tl-v2-records/out_v2"
CTX = ARCH / "pub-turingllm-T2-records" / "root/turing-tl/experiments/082-public-runs/out_turingllm/ctx_fresh"
os.environ.update(MODE="main", WCM_VERSION="v4", SKIP_SPEC="1", ORDER="list", TAG="dan44",
                  CTX_DIR=str(CTX), OUT=str(HERE / "out" / "rs"))
sys.path.insert(0, str(ROOT / "experiments" / "062-h100-protocol-v1"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import driver  # noqa: E402

TARGETS = Path(os.environ.get("TARGETS", str(HERE / "targets_tl.txt")))
MAX_MIN = float(os.environ.get("MAX_MIN", "85"))
STEPS = int(os.environ.get("STEPS", "60"))
LR = float(os.environ.get("LR", "0.05"))
FIT_BS = int(os.environ.get("FIT_BS", "8"))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
ALL_ARMS = ("fit_global", "fit_kind", "fit_site", "fit_member", "fit_member_c2", "fit_member_warm")
SKIP_DIAG = os.environ.get("SKIP_DIAG") == "1"
OBJ = os.environ.get("OBJ", "pre")                    # pre (job 2) | tk (job 3: the pass rule's read + censor hinge)
ARMS = tuple(a for a in os.environ.get("ARMS", ",".join(ALL_ARMS)).split(",") if a)
RUN_OUT = Path(os.environ.get("RUN_OUT", str(HERE / "out")))       # smoke tests: a scratch dir
ROWS = RUN_OUT / "rows.jsonl"
MEM = RUN_OUT / "members"
ISP1 = math.log(math.e - 1.0)          # softplus^-1(1)


def jfloat(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


# ------------------------------------------------------------------------------------------------ context
class Ctx:
    """Everything the reduced scorer and the fitter need for one target, built exactly as 049 score_circuit builds it
    (same probe, same slicing, same means, same baselines)."""

    def __init__(self, R, c, rec):
        from eval.ablation_faithfulness import upstream_sites
        from eval.floors import collect_site_anchors, collect_site_means
        V, G = R.V, R.G
        self.V, self.G = V, G
        self.inf, self.bank, self.KINDS, self.D, self.K = G["inference"], G["bank"], G["KINDS"], G["D"], G["K"]
        self.TapCO = V._engine_classes()[0]
        seed = V.seed_of(c)
        sf = seed.metadata["feature_id"]
        self.layer, self.kind, self.sl = sf.layer, sf.kind, sf.index
        pd_ = R.H.patch_eval_contexts(G, rec, "strong")
        M0 = G["M0"]
        pt, pa, nt = pd_.pos_tokens[:V.N_SEQ], pd_.pos_argmax[:V.N_SEQ], pd_.neg_tokens[:V.N_SEQ]
        if V.CTR_SOURCE != "store":
            from pipeline.component_index import component_idx
            comp = component_idx(self.layer, self.KINDS.index(self.kind), len(self.KINDS))
            sel = M0._neg_context_selector().select(comp, self.sl, V.CTR_SOURCE, max_sequences=V.N_SEQ,
                                                    batch_size=V.EVAL_BS, exact=False, non_activation_threshold=0.0,
                                                    filter_batch_size=32, load_window_size=256)
            nt = sel.tokens[:V.N_SEQ]
        n_pos, n_neg = int(pt.shape[0]), int(nt.shape[0])
        n_tr, n_ntr = V.split_n(n_pos), V.split_n(n_neg)
        if (pd_.metadata or {}).get("n_neg_train") is not None:
            n_ntr = int(pd_.metadata["n_neg_train"])
        assert V.EVAL_SPLIT == "explicit"
        n_tr = int(pd_.metadata["n_train"])
        self.pt, self.pa = pt, pa
        self.pt_tr, self.pa_tr, self.pt_ho, self.pa_ho = pt[:n_tr], pa[:n_tr], pt[n_tr:], pa[n_tr:]
        self.nt_tr = nt[:n_ntr]
        sae = self.bank.saes[self.kind][self.layer]
        self.SK = dict(w_seed=sae.encoder.weight[self.sl].detach(), b_seed=sae._get_bias_eff()[self.sl].detach())
        self.UPS = set(sorted(upstream_sites(self.bank, self.layer, self.kind)))
        self.dev = pt.device
        a = defaultdict(dict)
        for n in c.nodes.values():
            if n is seed:
                continue
            f = n.metadata["feature_id"]
            a[(f.layer, f.kind)][int(f.index)] = float(n.metadata.get("amplitude", 1.0))
        self.alphas = dict(a)
        self.sites = sorted(s for s, d in self.alphas.items() if d)
        self.msets = {s: torch.tensor(sorted(self.alphas[s]), device=self.dev, dtype=torch.long) for s in self.sites}
        self.keep = {s: set(self.alphas[s]) for s in self.sites}
        # member order: by site, then latent index (the flat vector every arm uses)
        self.mem_site = [s for s in self.sites for _ in range(len(self.msets[s]))]
        self.mem_idx = [int(i) for s in self.sites for i in self.msets[s].tolist()]
        self.alpha_fit = np.array([self.alphas[s][i] for s, i in zip(self.mem_site, self.mem_idx)], dtype=np.float64)
        self.n_mem = len(self.mem_idx)
        self.means_tr, self.pins_tr = collect_site_anchors(self.inf, self.bank, self.pt_tr, self.UPS, self.pa_tr,
                                                           pin_position_specific=False)
        self.means_neg = collect_site_means(self.inf, self.bank, self.nt_tr, self.UPS) if self.UPS else {}
        self.fills = {"Z": (None, False), "A": (self.means_tr, True), "C": (self.means_neg, True)}
        nat_pre, nat_tk = V.read(lambda a_, b_: V.AmpInjectPatcher({}, (self.layer, self.kind), self.SK["w_seed"],
                                                                    self.SK["b_seed"], self.sl), pt, pa, per_seq=True)
        self.a_ho = {"pre": float(nat_pre[n_tr:].mean()), "tk": float(nat_tk[n_tr:].mean())}
        self.a_tr = {"pre": float(nat_pre[:n_tr].mean()), "tk": float(nat_tk[:n_tr].mean())}
        self.E_ho = {f: self.co({}, self.pt_ho, self.pa_ho, *mt) for f, mt in self.fills.items()}
        self.E_tr = {f: self.co({}, self.pt_tr, self.pa_tr, *mt) for f, mt in self.fills.items()}

    def scales_of(self, alpha_vec):
        """{site: [D] scale vector} from a flat member coefficient vector (numpy or torch; torch keeps the graph)."""
        out, off = {}, 0
        for s in self.sites:
            n = len(self.msets[s])
            v = alpha_vec[off:off + n]
            if not torch.is_tensor(v):
                v = torch.tensor(np.asarray(v), dtype=torch.float32, device=self.dev)
            out[s] = torch.ones(self.D, device=self.dev, dtype=v.dtype).index_put((self.msets[s],), v)
            off += n
        return out

    def patcher(self, keep_, anchors, means, topk, scales, cls=None, **extra):
        kt = {s: self.msets[s] for s in keep_}
        return (cls or self.TapCO)(bank=self.bank, keep_indices=keep_, in_scope=self.UPS, seed_layer=self.layer,
                                   seed_kind=self.kind, seed_latent_idx=self.sl, pos_argmax=anchors, site_means=means,
                                   pin_values=None, respect_topk=topk, topk=self.K, keep_tensors=kt,
                                   keep_scales=scales if keep_ else None, **self.SK, **extra)

    def co(self, keep_, tokens, anchors, means, topk, scales=None):
        return self.V.read(lambda a_, b_: self.patcher(keep_, anchors[a_:b_], means, topk, scales), tokens, anchors)

    @staticmethod
    def frac(x, e, a):
        return (x - e) / (a - e) if e < a and abs(a - e) > 1e-9 else None

    def reduced(self, alpha_vec):
        sc = self.scales_of(alpha_vec)
        out = {}
        for f, (mm, tk) in self.fills.items():
            r = self.co(self.keep, self.pt_ho, self.pa_ho, mm, tk, sc)
            out[f] = self.frac(r["tk"], self.E_ho[f]["tk"], self.a_ho["tk"])
        return out

    # -------------------------------------------------------------------------------------------- fitter
    def fit(self, groups, n_groups, mode="softplus", steps=STEPS, lr=LR, bs=FIT_BS, seed=0, init=None):
        """Fit n_groups coefficients (member j uses group groups[j]) on the training reference rows. Returns the
        flat member vector, the final training loss and the loss trace.
        OBJ=pre (job 2): (ratio - 1)^2 on the relu pre-activation. OBJ=tk (job 3+): the ratio of the post-Top-K read
        the pass rule uses, with a straight-through surrogate (pre-activation where the target is inside the site's
        Top-K at the anchor, 0 where it is censored) plus a hinge (k-th largest site activation - pre) / (a - e)
        that pushes a censored target back inside (the role of v4's censored-activation term). init = a flat member
        vector to start from (default alpha = 1)."""
        g = torch.tensor(groups, device=self.dev, dtype=torch.long)
        if init is not None:
            v = np.asarray(init, dtype=np.float64)
            p0 = np.where(v > 20, v, np.log(np.expm1(np.maximum(v, 1e-6))))
            psi = torch.tensor(p0, dtype=torch.float32, device=self.dev).requires_grad_(True)
        else:
            psi = torch.full((n_groups,), ISP1 if mode == "softplus" else 0.0, device=self.dev, requires_grad=True)
        opt = torch.optim.Adam([psi], lr=lr)
        gen = torch.Generator().manual_seed(seed)
        n_tr = int(self.pt_tr.shape[0])
        fills = [f for f in ("Z", "A", "C") if self.E_tr[f]["pre"] < self.a_tr["pre"] - 1e-9]
        trace = []

        def amp(p):
            return torch.nn.functional.softplus(p) if mode == "softplus" else torch.pow(2.0, torch.tanh(p))

        self.inf.disable_compile()
        try:
            for step in range(steps):
                for pg in opt.param_groups:
                    pg["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
                idx = torch.randperm(n_tr, generator=gen)[:bs].to(self.dev)
                tok, anc = self.pt_tr[idx], self.pa_tr[idx]
                T = int(anc.max()) + 1
                tok = tok[:, :T]
                opt.zero_grad(set_to_none=True)
                tot = 0.0
                for f in fills:
                    mm, tk = self.fills[f]
                    sc = self.scales_of(amp(psi)[g])
                    p = self.patcher(self.keep, anc, mm, tk, sc, cls=self._kth_cls())
                    import contextlib, io
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.inf.forward(tok, patcher=p, grad_enabled=True, return_activations=False,
                                         tokenize_final=False)
                    pre = p.seed_pre
                    rr_ = torch.arange(len(anc), device=pre.device)
                    pa_ = pre[rr_, anc.to(pre.device)].float()
                    if OBJ == "tk" and self.E_tr[f]["tk"] < self.a_tr["tk"] - 1e-9:
                        kth = p.kth[rr_, anc.to(pre.device)].float()
                        inside = (pa_ >= kth).float().detach()
                        e, a = self.E_tr[f]["tk"], self.a_tr["tk"]
                        x = (torch.relu(pa_) * inside).mean()
                        loss = ((x - e) / (a - e) - 1.0) ** 2 + (torch.relu(kth - pa_).mean() / (a - e))
                    else:
                        x = torch.relu(pa_).mean()
                        e, a = self.E_tr[f]["pre"], self.a_tr["pre"]
                        loss = ((x - e) / (a - e) - 1.0) ** 2
                    loss.backward()
                    tot += float(loss)
                    del p, pre, x, loss
                opt.step()
                trace.append(round(tot, 5))
        finally:
            self.inf.enable_compile()
        with torch.no_grad():
            vec = amp(psi)[g].detach().cpu().numpy().astype(np.float64)
        return vec, trace

    def _kth_cls(self):
        """TapCO that also records the k-th largest Top-K activation at the seed site (detached): the target's
        censoring threshold."""
        if getattr(self, "_kcls", None) is None:
            bank, base = self.bank, self.TapCO

            class TapK(base):
                def transform(self_, layer_idx, kind, x):
                    if layer_idx == self_.seed_layer and kind == self_.seed_kind:
                        with torch.no_grad():
                            ta, _ = bank.encode(x.detach(), kind, layer_idx)
                            self_.kth = ta.min(dim=-1).values
                    return super().transform(layer_idx, kind, x)
            self._kcls = TapK
        return self._kcls

    def train_loss_tk(self, alpha_vec):
        """Sum over Z / A / C of (ratio - 1)^2 on the post-Top-K read of all training reference rows (no-grad)."""
        sc = self.scales_of(alpha_vec)
        tot = 0.0
        for f in ("Z", "A", "C"):
            if not self.E_tr[f]["tk"] < self.a_tr["tk"] - 1e-9:
                continue
            mm, tk = self.fills[f]
            r = self.co(self.keep, self.pt_tr, self.pa_tr, mm, tk, sc)
            tot += (self.frac(r["tk"], self.E_tr[f]["tk"], self.a_tr["tk"]) - 1.0) ** 2
        return tot

    def train_loss(self, alpha_vec):
        """The fitter's loss on ALL training reference rows (no-grad), for comparing arms on the fitted objective."""
        sc = self.scales_of(alpha_vec)
        tot = 0.0
        for f in ("Z", "A", "C"):
            if not self.E_tr[f]["pre"] < self.a_tr["pre"] - 1e-9:
                continue
            mm, tk = self.fills[f]
            r = self.co(self.keep, self.pt_tr, self.pa_tr, mm, tk, sc)
            tot += (self.frac(r["pre"], self.E_tr[f]["pre"], self.a_tr["pre"]) - 1.0) ** 2
        return tot


# ------------------------------------------------------------------------------------------------ diagnostics
def natural_capture(C):
    """Natural run on all activating rows (train then held-out): member codes at the anchor and max over positions
    <= anchor; per member site, ||x - xhat|| / ||x|| and ||xhat|| at the anchor; the site's natural ||xhat|| p99 over
    all positions <= anchor."""
    from sae.dense import sparse_topk_to_dense
    bank, D = C.bank, C.D
    k2i = {k: i for i, k in enumerate(C.KINDS)}
    tok, anc_all = C.pt, C.pa.to(C.dev)
    B, T = int(tok.shape[0]), int(tok.shape[1])
    out = {"a_anchor": defaultdict(list), "a_max": defaultdict(list), "relerr": defaultdict(list),
           "recon": defaultdict(list), "recon_p99": defaultdict(list), "xnorm": defaultdict(list)}
    cur = {}

    def hook(layer_idx, acts):
        rr, anc, posmask = cur["rr"], cur["anc"], cur["pm"]
        for kd in C.KINDS:
            s = (layer_idx, kd)
            if s not in C.msets:
                continue
            x = acts[k2i[kd]].float()
            ta, ti = bank.encode(x, kd, layer_idx)
            dense = sparse_topk_to_dense(ta, ti, D, dtype=torch.float32)
            xh = bank.decode(dense, kd, layer_idx).float()
            m = dense[..., C.msets[s]]                                               # [B, T, n_s]
            out["a_anchor"][s].append(m[rr, anc].cpu())
            out["a_max"][s].append(torch.where(posmask[..., None], m, torch.zeros_like(m)).amax(1).cpu())
            err = (x - xh).norm(dim=-1)
            xn = x.norm(dim=-1).clamp(min=1e-9)
            out["relerr"][s].append((err / xn)[rr, anc].cpu())
            out["recon"][s].append(xh.norm(dim=-1)[rr, anc].cpu())
            out["xnorm"][s].append(xn[rr, anc].cpu())
            out["recon_p99"][s].append(float(torch.quantile(xh.norm(dim=-1)[posmask].float(), 0.99)))
            del dense, xh, m

    import contextlib, io
    C.inf.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, B, 16):
                anc = anc_all[s0:s0 + 16]
                cur.update(rr=torch.arange(int(anc.shape[0]), device=C.dev), anc=anc,
                           pm=torch.arange(T, device=C.dev)[None, :] <= anc[:, None])
                C.inf.forward(tok[s0:s0 + 16], activations_callback=hook, return_activations=False,
                              tokenize_final=False)
    finally:
        C.inf.enable_compile()
    res = {}
    for k, d in out.items():
        if k == "recon_p99":
            res[k] = {s: float(np.max(v)) for s, v in d.items()}        # max over chunks of the per-chunk p99
        else:
            res[k] = {s: torch.cat(v, 0).numpy() for s, v in d.items()}
    return res


def edited_capture(C, alpha_vec, fill):
    """The fitted-alpha circuit-only run (fill Z or A) on the held-out rows: per member site, the members' edited
    values (alpha x live) at the anchor and max over positions <= anchor, and the edited site reconstruction norm at
    the anchor (decode of the patched code)."""
    from sae.dense import sparse_topk_to_dense
    bank, D = C.bank, C.D
    TapCO = C.TapCO
    rec = {"v_anchor": defaultdict(list), "v_max": defaultdict(list), "recon": defaultdict(list)}

    class Cap(TapCO):
        def transform(self, layer_idx, kind, x):
            out = super().transform(layer_idx, kind, x)
            s = (layer_idx, kind)
            if s in C.msets:
                ta, ti = bank.encode(x, kind, layer_idx)
                dense = sparse_topk_to_dense(ta, ti, D, dtype=torch.float32)
                err = x.float() - bank.decode(dense, kind, layer_idx).float()
                recon = (out.float() - err).norm(dim=-1)
                anc = self.pos_argmax.to(x.device).clamp(0, x.shape[1] - 1)
                rr = torch.arange(x.shape[0], device=x.device)
                v = dense[..., C.msets[s]] * self.keep_scales[s][C.msets[s]].float()
                pm = torch.arange(x.shape[1], device=x.device)[None, :] <= anc[:, None]
                rec["v_anchor"][s].append(v[rr, anc].cpu())
                rec["v_max"][s].append(torch.where(pm[..., None], v, torch.zeros_like(v)).amax(1).cpu())
                rec["recon"][s].append(recon[rr, anc].cpu())
            return out

    mm, tk = C.fills[fill]
    sc = C.scales_of(alpha_vec)
    C.V.read(lambda a_, b_: C.patcher(C.keep, C.pa_ho[a_:b_], mm, tk, sc, cls=Cap), C.pt_ho, C.pa_ho)
    return {k: {s: torch.cat(v, 0).numpy() for s, v in d.items()} for k, d in rec.items()}


# ------------------------------------------------------------------------------------------------ one target
def one_a1(R, key):
    """DIAG_A1=1: the off-manifold control. The same edited capture as the fitted-alpha diagnostics, with every
    member at alpha = 1, so the circuit-only ablation's own effect on member values is separated from alpha's."""
    t0 = time.time()
    c = torch.load(OUT_V4 / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
    C = Ctx(R, c, R.contexts(key))
    one_ = np.ones_like(C.alpha_fit)
    arrs = {}
    for nm in ("Z", "A"):
        ed = edited_capture(C, one_, nm)
        arrs["v%s_max_a1" % nm] = np.concatenate([ed["v_max"][s] for s in C.sites], axis=1).astype(np.float32)
        arrs["recon_%s_a1" % nm] = np.stack([ed["recon"][s] for s in C.sites], 1).astype(np.float32)
    MEM.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MEM / ("%s.npz" % key), **arrs)
    return dict(seed=key, t_total=round(time.time() - t0, 1))


def one(R, key, natmax, dec_norm):
    if os.environ.get("DIAG_A1") == "1":
        return one_a1(R, key)
    t0 = time.time()
    c = torch.load(OUT_V4 / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
    rec = R.contexts(key)
    full = R.score(c, rec, "strong")
    row = dict(seed=key, n=driver.n_nodes(c), t_full=round(time.time() - t0, 1))
    for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "a_pos_tk", "e0_tk", "eM_topk_tk",
              "eN_topk_tk", "vacuous_tk", "vacuous_v2", "free0_a1_tk"):
        row["full_" + k] = full.get(k)
    C = Ctx(R, c, rec)
    a = C.alpha_fit
    red = C.reduced(a)
    row["red_fitted"] = red
    diffs = [abs(red[f] - full[h]) for f, h in zip("ZAC", ("free0_tk", "freeM_topk_tk", "freeN_topk_tk"))
             if red[f] is not None and full[h] is not None]
    row["red_check"] = max(diffs) if diffs else None

    var = {"alpha1": np.ones_like(a), "cap_hi2": np.minimum(a, 2.0)}
    for cc in (1.25, 1.5, 2.0, 3.0):
        var["clip_%g" % cc] = np.clip(a, 1.0 / cc, cc)
    site_of = np.array(["%d.%s" % s for s in C.mem_site])
    kind_of = np.array([s[1] for s in C.mem_site])
    for name, lab in (("med_site", site_of), ("med_kind", kind_of), ("med_global", np.zeros(len(a)))):
        v = np.empty_like(a)
        for u in np.unique(lab):
            v[lab == u] = np.median(a[lab == u])
        var[name] = v
    row["var"] = {k: C.reduced(v) for k, v in var.items()}
    row["t_var"] = round(time.time() - t0, 1)

    fits, losses, traces = {}, {}, {}
    def groups_of(lab):
        u = {x: i for i, x in enumerate(sorted(set(lab.tolist())))}
        return [u[x] for x in lab.tolist()], len(u)
    spec = {"fit_global": (groups_of(np.zeros(len(a), dtype=int)), "softplus"),
            "fit_kind": (groups_of(kind_of), "softplus"), "fit_site": (groups_of(site_of), "softplus"),
            "fit_member": ((list(range(len(a))), len(a)), "softplus"),
            "fit_member_c2": ((list(range(len(a))), len(a)), "c2"),
            "fit_member_warm": ((list(range(len(a))), len(a)), "softplus")}
    for arm in ARMS:
        (gr, ng), mode = spec[arm]
        ts = time.time()
        vec, tr = C.fit(gr, ng, mode=mode, init=a if arm == "fit_member_warm" else None)
        fits[arm] = vec
        traces[arm] = tr
        row.setdefault("fit", {})[arm] = dict(red=C.reduced(vec), n_par=ng, loss_tr=C.train_loss(vec),
                                              loss_tr_tk=C.train_loss_tk(vec),
                                              t=round(time.time() - ts, 1), loss_first=tr[0], loss_last=tr[-1],
                                              alpha_med=float(np.median(vec)), alpha_min=float(vec.min()),
                                              alpha_max=float(vec.max()))
    row["loss_tr_fitted"] = C.train_loss(a)
    row["loss_tr_alpha1"] = C.train_loss(np.ones_like(a))
    row["loss_tr_tk_fitted"] = C.train_loss_tk(a)
    row["obj"] = OBJ
    row["t_fit"] = round(time.time() - t0, 1)
    row["traces"] = traces
    if SKIP_DIAG:
        MEM.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(MEM / ("%s.npz" % key), alpha=a, **{"alpha_" + k: v.astype(np.float32)
                                                               for k, v in fits.items()})
        row["t_total"] = round(time.time() - t0, 1)
        return row

    # diagnostics
    nat = natural_capture(C)
    edZ = edited_capture(C, a, "Z")
    edA = edited_capture(C, a, "A")
    n_tr = int(C.pt_tr.shape[0])
    comp = {s: s[0] * len(C.KINDS) + C.KINDS.index(s[1]) for s in C.sites}
    arrs = dict(alpha=a, site_layer=np.array([s[0] for s in C.mem_site]), site_kind=kind_of,
                latent=np.array(C.mem_idx), n_train=np.array(n_tr))
    def stack(d):
        return np.concatenate([d[s] for s in C.sites], axis=1)            # [B, n_mem]
    arrs["a_anchor"] = stack(nat["a_anchor"]).astype(np.float32)        # natural, all 64 rows
    arrs["a_max"] = stack(nat["a_max"]).astype(np.float32)
    arrs["natmax"] = np.array([natmax[0, comp[s], i] for s, i in zip(C.mem_site, C.mem_idx)], dtype=np.float32)
    arrs["nat64"] = np.array([natmax[1, comp[s], i] for s, i in zip(C.mem_site, C.mem_idx)], dtype=np.float32)
    arrs["dec_norm"] = np.array([dec_norm[s][i] for s, i in zip(C.mem_site, C.mem_idx)], dtype=np.float32)
    for nm, ed in (("Z", edZ), ("A", edA)):
        arrs["v%s_anchor" % nm] = stack(ed["v_anchor"]).astype(np.float32)
        arrs["v%s_max" % nm] = stack(ed["v_max"]).astype(np.float32)
    for k, v in fits.items():
        arrs["alpha_" + k] = v.astype(np.float32)
    arrs["sites"] = np.array(["%d.%s" % s for s in C.sites])
    arrs["relerr"] = np.stack([nat["relerr"][s] for s in C.sites], 1).astype(np.float32)   # [64, n_sites]
    arrs["recon_nat"] = np.stack([nat["recon"][s] for s in C.sites], 1).astype(np.float32)
    arrs["xnorm"] = np.stack([nat["xnorm"][s] for s in C.sites], 1).astype(np.float32)
    arrs["recon_p99"] = np.array([nat["recon_p99"][s] for s in C.sites], dtype=np.float32)
    for nm, ed in (("Z", edZ), ("A", edA)):
        arrs["recon_%s" % nm] = np.stack([ed["recon"][s] for s in C.sites], 1).astype(np.float32)   # [16, n_sites]
    MEM.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MEM / ("%s.npz" % key), **arrs)
    row["traces"] = traces
    row["t_total"] = round(time.time() - t0, 1)
    return row


def main():
    targets = [t for t in TARGETS.read_text().split() if t]
    done = set()
    if ROWS.exists():
        for line in open(ROWS):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add(r["seed"])
            except Exception:  # noqa: BLE001
                pass
    todo = [t for t in targets if t not in done]
    if LIMIT:
        todo = todo[:LIMIT]
    print("118: %d targets to do (%d done); arms %s; steps %d lr %g bs %d" % (len(todo), len(done), ARMS, STEPS, LR,
                                                                            FIT_BS), flush=True)
    R = driver.Runner()
    natmax = np.load(HERE / "out" / "natmax.npy", mmap_mode="r")
    bank = R.G["bank"]
    dec_norm = {}
    for kd in R.G["KINDS"]:
        for L_ in range(bank.n_layer):
            W = bank.saes[kd][L_].decoder.weight.detach().float()          # [d_in, d_sae]
            dec_norm[(L_, kd)] = W.norm(dim=0).cpu().numpy()
    t_start = time.time()
    with open(ROWS, "a") as fh:
        for n_, key in enumerate(todo):
            if (time.time() - t_start) / 60 > MAX_MIN:
                print("MAX_MIN reached; stopping", flush=True)
                break
            try:
                row = one(R, key, natmax, dec_norm)
            except Exception as e:  # noqa: BLE001
                row = dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))
                traceback.print_exc()
                torch.cuda.empty_cache()
            fh.write(json.dumps(row, default=jfloat) + "\n"); fh.flush()
            f = row.get("fit", {})
            print("%3d/%d %-16s %s n=%s chk=%s Z/A/C fit=%s a1=%s site=%s member=%s  %ss" % (
                n_ + 1, len(todo), key, row.get("error", "ok"), row.get("n"), row.get("red_check"),
                row.get("red_fitted"), row.get("var", {}).get("alpha1"), f.get("fit_site", {}).get("red"),
                f.get("fit_member", {}).get("red"), row.get("t_total")), flush=True)
            torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
