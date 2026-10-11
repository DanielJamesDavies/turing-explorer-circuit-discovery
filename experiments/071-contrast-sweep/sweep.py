"""DAN-64 CONTRAST-CONTEXT SWEEP: does the contrast source WCM trains its C term on matter? Four sources, all post-Top-K
silent: close (the production set), random, distant, and fused (a mix of the three; a training arm only). Everything
else is protocol v1 (paper/protocol-v1.md).

Per target, in one process per shard (SHARD=i/k), resumable at every step:
  1. contexts  the 15k run's cached record, copied read-only from SRC_OUT/ctx/<key>.pt into OUT/ctx (as 062's
               run_lowlam.sh does). Activating contexts, split and anchors are the production ones.
  2. sets      the four contrast sets, built in logic.BUILD_ORDER and cached as OUT/sets/<source>/<key>.pt:
                 close_silent    rec["neg"], the production set as is (its re-selection is checked against it,
                                 v1_match, and continued past 64 for the reserve)
                 random_silent   excluding every close context (train, held-out and reserve)
                 distant_silent  excluding every close and random context
                 fused_silent    excluding every close, random and distant context: sub-mode thirds (close 22,
                                 distant 21, random 21 of 64; 12 / 10 / 10 of the reserve), each sub-mode selected
                                 with the cumulative exclusions plus fused's earlier sub-modes; each part split by
                                 the stratified rule separately (train 16 / 16 / 16, held-out 6 / 5 / 5), training
                                 slices merged round robin (close, distant, random, ...) first, held-out slices last
               Every source also excludes the target's activating contexts and its own repeats (by corpus id and by
               token content). 64 contexts in the source's order, then stratified (train first, held-out last, as v1
               orders the close set), plus RESERVE more. The sets are then asserted pairwise disjoint and no held-out
               context of any set may be a training context of any arm (logic.ho_trained_matrix; the one tolerated
               exception is the production close set's own exact duplicates, recorded). One sets.shardN.jsonl row per
               (target, source): contamination (post-Top-K firing, pre-activation relative to the activating
               reference), similarity to the top contexts, cross-set overlaps, held-out-in-training counts, metadata,
               and for fused the per-sub-mode counts, firing and similarity (part_*). Set files carry
               logic.set_version(source): the three eval sets keep 071-disjoint-v1, so an OUT built before fused was
               added resumes on its sets and only builds fused.
  3. fit       WCM[Z+C+A] per (arm, lambda) through 062 driver.Runner.fit with rec["neg"] replaced by the arm's set:
               OUT/circuits/<arm>/<lambda>/<key>.pt. close_silent at lambda 1e-3 is the production circuit, copied
               from SRC_OUT/main/circuits (REUSE_MAIN=1).
  4. battery   every circuit scored by the protocol scorer (amp_eval_pass_v2.score_circuit, EVAL_SPLIT=explicit)
               on held-out strongest, once per eval set (close, random, distant): the eval set's 48 training contexts give
               the C mean, its 16 held-out contexts the sufficiency-to-induce set. The same held-out sets for every
               arm; the training source is the only thing that varies. One eval.shardN.jsonl row per
               (target, arm, lambda, eval set), with n_ho_trained (0) and n_ctr_train_overlap.
  5. random    targets in RANDOM_TARGETS: a random circuit matched site by site to each arm's lambda 1e-3 circuit,
               members drawn from latents live on the 48 strongest-train contexts, coefficients fitted on the arm's
               contrast set (062 random_null's support-restricted engine), scored on the battery:
               random.shardN.jsonl.

  CUDA_VISIBLE_DEVICES=0 SHARD=0/16 PYTHONPATH=src .venv/bin/python -X utf8 experiments/071-contrast-sweep/sweep.py
  env: OUT (experiments/071-contrast-sweep/out)  TARGETS (targets.txt)  SRC_OUT (062 out_full/out)
       ARMS (comma list of logic.ARMS, default all 4; the eval sets are always the 3 of logic.EVAL_SETS; all 4 sets
       are always built)  LAMS (3e-4,1e-3,3e-3)
       RANDOM_TARGETS (targets_random.txt, or a comma list; empty = no random baselines)  REUSE_MAIN (1)
       MODE (run | sets: build, check and log the contrast sets only, no fits)  LIMIT (first N targets of this shard)
       ORDER (interleaved | list)  SLOT (co-tenant slot on the GPU; launch_pod.sh sets it)
       NO_TOKEN_PRELOAD=1 (skip the selector's negctx token cache: for a CPU dry run with little RAM)
"""
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
D062 = EXP / "062-h100-protocol-v1"
OUT = Path(os.environ.get("OUT", str(HERE / "out")))
# Before 062's driver (and through it the scorer) is imported: the driver reads OUT; the scorer reads EVAL_SPLIT and
# CTR_SOURCE at import. "explicit" scores the exact held-out set (protocol-v1.md 3.6); a CTR_SOURCE other than
# "store" makes the scorer read the contrast set through the injected FixedSelector, i.e. the set given here.
os.environ["OUT"] = str(OUT)
os.environ["EVAL_SPLIT"] = "explicit"
os.environ["CTR_SOURCE"] = "close"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(D062))
import torch  # noqa: E402

import driver  # noqa: E402
import logic as L  # noqa: E402

MODE = os.environ.get("MODE", "run")
TARGETS = Path(os.environ.get("TARGETS", str(HERE / "targets.txt")))
SRC_OUT = Path(os.environ.get("SRC_OUT", str(D062 / "out_full" / "out")))
ARMS = [a for a in os.environ.get("ARMS", ",".join(L.ARMS)).split(",") if a]
LAMS = [float(x) for x in os.environ.get("LAMS", ",".join(L.lam_tag(x) for x in L.LAMS)).split(",") if x]
EVALS = list(L.EVAL_SETS)                                    # always all three: the 4 x 3 table
_rt = os.environ.get("RANDOM_TARGETS", str(HERE / "targets_random.txt"))      # a file, or a comma list of keys
RANDOM_TARGETS = (set(Path(_rt).read_text().split()) if _rt and Path(_rt).is_file()
                  else {k for k in _rt.split(",") if k.strip()})
REUSE_MAIN = os.environ.get("REUSE_MAIN", "1") == "1"
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
ORDER = os.environ.get("ORDER", "interleaved")
SLOT = int(os.environ.get("SLOT", "0"))
TAG = driver.TAG
for a in ARMS:
    assert a in L.SOURCES, "unknown arm %r (logic.ARMS)" % a
REF_IN_GRID = any(abs(l - L.REF_LAM) < 1e-12 for l in LAMS)  # random baselines match the lambda 1e-3 circuits
FWD_BATCH = 16
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


class _Stop(Exception):
    """End this target's work early (skip record, or MODE=sets)."""


def log(s):
    print("[%s %s] %s" % (time.strftime("%H:%M:%S"), TAG, s), flush=True)


def jsonable(v):
    if isinstance(v, (str, bool, int)) or v is None:
        return v
    if isinstance(v, float):
        return None if v != v else v
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v] if len(v) <= 16 else "<%d items>" % len(v)
    if torch.is_tensor(v):
        return jsonable(v.item()) if v.numel() == 1 else "<tensor %s>" % list(v.shape)
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


# ============================================================================================= contrast sets
class Sets:
    """Builds, caches, checks and logs one target's four disjoint contrast sets."""

    def __init__(self, R, key, rec, sets_fh, sets_done):
        from pipeline.component_index import component_idx
        self.R, self.G, self.key, self.rec = R, R.G, key, rec
        self.layer, self.kind, self.latent = R.H.parse(key)
        self.comp = component_idx(self.layer, self.G["KINDS"].index(self.kind), self.G["NK"])
        self.fh, self.done = sets_fh, sets_done
        self._sel = None
        self._ref = None
        self.cache = {}
        self.ho = {}
        self.ovl = {}
        s = rec["strong"]
        self.strong_train = s["pos"][s["train"]]
        acts = [s["pos"]] + ([rec["mid"]["pos"]] if rec.get("mid") is not None else [])
        self.activating = {L.content_key(r) for p in acts for r in p}

    @property
    def sel(self):
        """A fresh close-contrast selector per target, as the 15k run built one (protocol_contexts.make_selector)."""
        if self._sel is None:
            from circuit.protocol_contexts import make_selector
            self._sel = make_selector(self.G["inference"], self.G["bank"], self.G["M0"].probe_builder.loader)
        return self._sel

    @property
    def ref(self):
        """The activating reference for the contamination log: the median over the 48 strongest-train contexts of the
        per-context max relu(w.x + b) (NegContextSelector.posctx_reference)."""
        if self._ref is None:
            self._ref = self.sel.posctx_reference(self.strong_train.to(self.G["device"]), self.comp, self.latent,
                                                  batch_size=FWD_BATCH, stat="median")
        return self._ref

    # ------------------------------------------------------------------------------------------- selection
    def _select(self, mode, m):
        """(ids, tokens [<=m, 64] cpu long, metadata) in the selector's order, or None."""
        kw = dict(L.SELECT_KW, selection_seed=L.SELECTION_SEED)
        s = self.sel.select(self.comp, self.latent, mode, max_sequences=m, **kw)
        if s is None:
            return None
        return [int(x) for x in s.sequence_ids], s.tokens.cpu().long(), dict(s.metadata)

    def _select_excluding(self, mode, excl_ids, excl_keys, want, m0):
        """Select m0 contexts in the selector's order and drop the excluded ones (logic.disjoint_keep); while fewer
        than `want` survive and the selector is not exhausted, select twice as many. The selector's order does not
        depend on the count asked for (close / distant: the similarity ranking; random: one fixed permutation of the
        candidates), so a larger selection only extends the smaller one. Returns (ids, tokens, meta, kept, drop) or
        None."""
        m = m0
        for _ in range(L.MAX_GROW + 1):
            got = self._select(mode, m)
            if got is None:
                return None
            ids, toks, meta = got
            kept, drop = L.disjoint_keep(ids, [L.content_key(r) for r in toks], excl_ids, excl_keys)
            if len(kept) >= want or toks.shape[0] < m:
                break
            m *= 2
        meta.update(n_asked=m, n_selected=int(toks.shape[0]), dropped=drop)
        return ids, toks, meta, kept, drop

    def _prior(self, name):
        """Ids and content keys of every context (train, held-out, reserve) of the sets built before `name`."""
        ids, keys = set(), set()
        for p in L.BUILD_ORDER[:L.BUILD_ORDER.index(name)]:
            st = self.get(p)
            if st.get("unusable"):
                continue
            ids |= {int(i) for i in (st.get("ids") or []) + list(st.get("reserve_ids") or [])}
            keys |= set(st["keys"]) | set(st["reserve_keys"])
        return ids, keys

    def _build(self, name):
        mode, filt = L.SOURCES[name]
        t0 = time.time()
        n = L.CONTRAST_COUNT
        extra = {}
        if name == "close_silent":
            # the production set: rec["neg"] (already stratified) and rec["neg_ids"]; its re-selection checks the
            # selector path (v1_match on the first 64) and, continued past them, gives the reserve
            toks = self.rec["neg"].cpu().long()
            ids = [int(i) for i in self.rec["neg_ids"]] if self.rec.get("neg_ids") is not None else None
            prod = [L.content_key(r) for r in toks]
            got = self._select_excluding(mode, ids or (), set(prod) | self.activating, L.RESERVE,
                                         n + L.RESERVE + L.SLACK)
            if got is None:
                res_ids, res_toks, extra["v1_match"], extra["meta"] = [], toks[:0], None, None
            else:
                g_ids, g_toks, meta, kept, _ = got
                first = {L.content_key(r) for r in g_toks[:toks.shape[0]]}
                extra["v1_match"] = len(first & set(prod)) / max(1, len(first | set(prod)))
                res = kept[:L.RESERVE]
                res_ids, res_toks = [g_ids[j] for j in res], g_toks[res]
                extra["meta"] = meta
            extra["n_dup_internal"] = len(prod) - len(set(prod))
            st = dict(ids=ids, tokens=toks, reserve_ids=res_ids, reserve_tokens=res_toks)
        elif name == L.FUSED:
            st = self._build_fused(name, n, extra)
            if st is None:
                return dict(name=name, unusable=True, n=extra["n_main"], version=L.set_version(name),
                            t_select=round(time.time() - t0, 1), meta=extra.get("meta"))
        else:
            excl_ids, excl_keys = self._prior(name)
            got = self._select_excluding(mode, excl_ids, excl_keys | self.activating, n + L.RESERVE,
                                         n + L.RESERVE + L.SLACK)
            main, res = ([], []) if got is None else L.split_main_reserve(got[3], n, L.RESERVE)
            if len(main) < L.MIN_CONTRAST:
                return dict(name=name, unusable=True, n=len(main), version=L.set_version(name),
                            t_select=round(time.time() - t0, 1))
            g_ids, g_toks, meta, _, _ = got
            order = L.stratified_positions(len(main))
            main = [main[j] for j in order]
            st = dict(ids=[g_ids[j] for j in main], tokens=g_toks[main], reserve_ids=[g_ids[j] for j in res],
                      reserve_tokens=g_toks[res])
            extra.update(meta=meta, n_dup_internal=0)
        st.update(name=name, mode=mode, filter=filt, n=int(st["tokens"].shape[0]), version=L.set_version(name),
                  n_train=L.n_train_of(int(st["tokens"].shape[0])), t_select=round(time.time() - t0, 1), **extra)
        if "parts" in st:
            st["part_counts"] = L.part_counts(st["parts"], st["n_train"], st["reserve_parts"])
        if st["n"] < L.MIN_CONTRAST:
            st["unusable"] = True
        return st

    def _build_fused(self, name, n, extra):
        """fused_silent: sub-mode thirds (logic.fused_quotas, as src's _select_fused) of the set and of the reserve.
        Each sub-mode (close, distant, random) is selected excluding every context of the sets built before fused,
        the activating contexts, and the contexts fused's earlier sub-modes took (their set and reserve parts). Each
        part is split separately by v1's stratified rule, with held-out counts apportioned so the totals are exactly
        48 / 16 (logic.fused_split_merge); the training slices are merged round robin (close, distant, random, ...)
        first and the held-out slices last. Of 64: train 16 / 16 / 16, held-out 6 / 5 / 5. The reserve parts are
        merged round robin. A short sub-mode is not topped up by the others (as in src). Returns the set dict (with per-context `parts`), or None
        if fewer than MIN_CONTRAST contexts survive; extra gets the metadata (per sub-mode: the selector's, quotas,
        `dropped`)."""
        excl_ids, excl_keys = self._prior(name)
        excl_keys = excl_keys | self.activating
        qm, qr = L.fused_quotas(n), L.fused_quotas(L.RESERVE)
        mains, ress, subs = {}, {}, {}
        drop = {"id": 0, "content": 0, "dup": 0}
        for sub in L.FUSED_PARTS:
            want = qm[sub] + qr[sub]
            # the sub-mode's ranking starts with its own source's set and reserve (excluded): ask past them
            got = self._select_excluding(sub, excl_ids, excl_keys, want, n + L.RESERVE + want + L.SLACK)
            if got is None:
                mains[sub], ress[sub], subs[sub] = [], [], None
                continue
            g_ids, g_toks, meta, kept, d = got
            m_, r_ = L.split_main_reserve(kept, qm[sub], qr[sub])
            mains[sub] = [(g_ids[j], g_toks[j]) for j in m_]
            ress[sub] = [(g_ids[j], g_toks[j]) for j in r_]
            excl_ids |= {int(i) for i, _ in mains[sub] + ress[sub]}
            excl_keys |= {L.content_key(t) for _, t in mains[sub] + ress[sub]}
            subs[sub] = meta
            drop = {k: drop[k] + d[k] for k in drop}
        n_main = sum(len(v) for v in mains.values())
        extra.update(meta=dict(fused_quotas=qm, reserve_quotas=qr, submodes=subs, dropped=drop), n_dup_internal=0,
                     n_main=n_main)
        if n_main < L.MIN_CONTRAST:
            return None
        main, res = L.fused_split_merge(mains), L.round_robin(ress)
        width = main[0][1][1].shape[-1]
        stack = lambda xs: (torch.stack([t for _, (_, t) in xs]) if xs  # noqa: E731
                            else torch.zeros(0, width, dtype=torch.long))
        return dict(ids=[int(i) for _, (i, _) in main], tokens=stack(main), parts=[p for p, _ in main],
                    reserve_ids=[int(i) for _, (i, _) in res], reserve_tokens=stack(res),
                    reserve_parts=[p for p, _ in res])

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        path = OUT / "sets" / name / ("%s.pt" % self.key)
        if path.exists():
            st = torch.load(path, weights_only=False)
            if st.get("version") != L.set_version(name):
                raise RuntimeError("%s was built by another design (%r, not %r): use a fresh OUT"
                                   % (path, st.get("version"), L.set_version(name)))
        else:
            st = self._build(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(st, path)
        if not st.get("unusable"):
            st["keys"] = [L.content_key(r) for r in st["tokens"]]
            st["reserve_keys"] = [L.content_key(r) for r in st["reserve_tokens"]]
        self.cache[name] = st
        return st

    def build_all(self):
        """Build (or load) the four sets in BUILD_ORDER, assert the disjointness, log one row per set. If any set's row
        is missing, every set's row is (re)written: a set added to a resumed OUT (fused) changes the others' overlap
        and held-out-in-training columns, and the report keeps the last row per (target, set)."""
        for name in L.BUILD_ORDER:
            self.get(name)
        use = {n: st for n, st in self.cache.items() if not st.get("unusable")}
        allk = {n: st["keys"] + st["reserve_keys"] for n, st in use.items()}
        alli = {n: [int(i) for i in (st.get("ids") or []) + list(st.get("reserve_ids") or [])] for n, st in use.items()}
        self.ovl = {"content": L.pairwise_overlap(allk), "id": L.pairwise_overlap(alli)}
        self.ho = L.ho_trained_matrix(use)
        bad = ["%s/%s %s=%d" % (a, b, w, v) for w, d in self.ovl.items() for (a, b), v in d.items() if v]
        bad += ["held-out of %s in training of %s: %d" % (e, a, v) for (a, e), v in self.ho.items()
                if v and not L.ho_trained_allowed(a, e)]
        bad += ["%s overlaps the activating contexts: %d" % (n, len(set(k) & self.activating))
                for n, k in allk.items() if n != L.REFERENCE and set(k) & self.activating]
        if any((self.key, name) not in self.done for name in L.BUILD_ORDER):
            for name in L.BUILD_ORDER:
                self.fh.write(json.dumps(self.row(self.cache[name])) + "\n"); self.fh.flush()
                self.done.add((self.key, name))
        assert not bad, "contrast sets are not disjoint: " + "; ".join(bad)

    # ------------------------------------------------------------------------------------------- logging
    def reads(self, tokens):
        """Per context: the target's max post-Top-K activation and max relu(w.x + b) over the 64 positions."""
        from sae.dense import target_latent_activations
        G, layer, kind, latent = self.G, self.layer, self.kind, self.latent
        bank, ki = G["bank"], G["KINDS"].index(kind)
        sae = bank.saes[kind][layer]
        w, b = sae.encoder.weight[latent].detach(), sae._get_bias_eff()[latent].detach()
        tk, pre = [], []

        def hook(layer_idx, activations):
            if layer_idx == layer:
                act = activations[ki]
                ta, ti = bank.encode(act, kind, layer_idx)
                tk.append(target_latent_activations(ta, ti, latent).float().max(-1).values.cpu())
                p = torch.relu(act @ w.to(act.device, act.dtype) + b.to(act.device, act.dtype))
                pre.append(p.float().max(-1).values.cpu())
        G["inference"].disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(tokens.shape[0]), FWD_BATCH):
                    G["inference"].forward(tokens[s0:s0 + FWD_BATCH].to(G["device"]), activations_callback=hook,
                                           return_activations=False, tokenize_final=False)
        finally:
            G["inference"].enable_compile()
        return torch.cat(tk), torch.cat(pre)

    def sims(self, ids):
        """Max cosine similarity of each context to the target's top-context store (the close ranking's own score);
        NaN for ids absent from seq_repr."""
        sel = self.sel
        ref, _ = sel.topctx_reference_reprs(self.comp, self.latent)
        dev = sel._ranking_device()
        refn = sel._normalized_reference_reps(ref, dev)
        reps = sel._get_repr_for_ids([int(i) for i in ids])
        valid = reps.abs().sum(dim=1) > 0
        out = torch.full((len(ids),), float("nan"))
        if bool(valid.any()):
            out[valid] = sel._positive_set_max_similarity(reps[valid], refn, dev)
        return out

    def row(self, st):
        r = dict(seed=self.key, set=st["name"], mode=L.SOURCES[st["name"]][0], filter=L.SOURCES[st["name"]][1],
                 layer=self.layer, kind=self.kind, band=L.band(self.layer), n=st.get("n"),
                 t_select=st.get("t_select"), shard=TAG, version=L.set_version(st["name"]))
        if st.get("unusable"):
            r["unusable"] = True
            return r
        n_tr = st["n_train"]
        tk, pre = self.reads(st["tokens"])
        ref = float(self.ref) if self.ref else None
        rel = pre / ref if ref else torch.full_like(pre, float("nan"))
        q = lambda v, p: round(float(torch.quantile(v.float(), p)), 4) if v.numel() else None  # noqa: E731
        for part, sl in (("", slice(None)), ("_train", slice(0, n_tr)), ("_held", slice(n_tr, None))):
            r["frac_fire" + part] = round(float((tk[sl] > 0).float().mean()), 4) if tk[sl].numel() else None
            r["pre_rel_med" + part] = q(rel[sl], 0.5)
        r.update(n_train=n_tr, n_held=st["n"] - n_tr, n_reserve=len(st["reserve_keys"]), posctx_ref=ref,
                 pre_rel_p90=q(rel, 0.9), pre_rel_max=q(rel, 1.0), tk_max_mean=round(float(tk.mean()), 4))
        parts = st.get("parts")
        masks = {p: torch.tensor([x == p for x in parts], dtype=torch.bool) for p in L.FUSED_PARTS} if parts else {}
        for p, c in (st.get("part_counts") or {}).items():          # fused: per sub-mode counts, firing, pre_rel
            r.update({"part_%s_%s" % (w, p): v for w, v in c.items()})
            m = masks[p]
            r["part_fire_%s" % p] = round(float((tk[m] > 0).float().mean()), 4) if bool(m.any()) else None
            r["part_pre_rel_med_%s" % p] = q(rel[m], 0.5)
        if st.get("ids") is not None and len(st["ids"]) == st["n"]:
            s = self.sims(st["ids"])
            ok = s[~torch.isnan(s)]
            r.update(sim_med=q(ok, 0.5), sim_min=q(ok, 0.0), sim_max=q(ok, 1.0), sim_valid=int(ok.numel()))
            for p, m in masks.items():
                r["part_sim_med_%s" % p] = q(s[m & ~torch.isnan(s)], 0.5)
        keys = set(st["keys"])
        r["n_activating_overlap"] = len((keys | set(st["reserve_keys"])) & self.activating)
        r["n_unique"] = len(keys)
        r["n_dup_internal"] = st.get("n_dup_internal")
        name = st["name"]
        for w, d in self.ovl.items():                  # cross-set overlaps (train + held-out + reserve): must be 0
            for (a, b), v in d.items():
                if name in (a, b):
                    r["ovl_%s_%s" % (w, b if a == name else a)] = v
        for (a, e), v in self.ho.items():              # this set as the eval set: its held-out in arm a's training
            if e == name:
                r["ho_in_train_%s" % a] = v
        for k in ("v1_match",):
            if k in st:
                r[k] = st[k]
        r["meta"] = jsonable(st.get("meta"))
        return r


# ============================================================================================= battery
def battery_neg(sets, ev, arm):
    """The contrast tensor scored for (arm, eval set): the eval set itself (48 training contexts for the C mean, 16
    held-out for the induce set), the same for every arm. Returns (tokens, info); raises if a held-out context of the
    eval set is a training context of the arm (other than the production close set's own duplicates)."""
    S, T = sets.get(ev), sets.get(arm)
    n_tr, t_tr = S["n_train"], T["keys"][:T["n_train"]]
    n_ho = L.held_in_train(t_tr, S["keys"][n_tr:])
    if n_ho and not L.ho_trained_allowed(arm, ev):
        raise AssertionError("%d held-out contexts of %s are training contexts of %s" % (n_ho, ev, arm))
    return S["tokens"], dict(n_ho_trained=n_ho, n_ctr_train_overlap=len(set(S["keys"][:n_tr]) & set(t_tr)))


def score_battery(R, c, rec, sets, arm, fh, done, label, done_key):
    """Score circuit c once per eval set; one row each (label fields + eval set + the overlap counts)."""
    t0 = time.time()
    for ev in EVALS:
        if done_key(ev) in done:
            continue
        S = sets.get(ev)
        try:
            if S.get("unusable"):
                row = dict(skip="eval_set_unusable")
            else:
                toks, info = battery_neg(sets, ev, arm)
                t = time.time()
                row = R.score(c, dict(rec, neg=toks), "strong")
                row.update(info, t_score=round(time.time() - t, 2))
        except Exception as e:  # noqa: BLE001
            row = dict(error="%s: %s" % (type(e).__name__, str(e)[:300]))
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        row.update(label, eval=ev, held="strong", shard=TAG)
        fh.write(json.dumps(row) + "\n"); fh.flush()
    return round(time.time() - t0, 1)


# ============================================================================================= main
def copy_if_missing(src, dst):
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def run():
    targets = [t for t in TARGETS.read_text().split() if t][driver.SHARD_I::driver.SHARD_K]
    if ORDER == "interleaved":
        targets = driver.interleave(targets, SLOT)
    if LIMIT:
        targets = targets[:LIMIT]
    log("MODE %s | %d targets | arms %s | lambdas %s | evals %s | random %d | out %s | %s" % (
        MODE, len(targets), ",".join(ARMS), ",".join(map(L.lam_tag, LAMS)), ",".join(EVALS),
        len(RANDOM_TARGETS & set(targets)), OUT, DEVICE_NAME))
    OUT.mkdir(parents=True, exist_ok=True)
    p = {k: OUT / ("%s.%s.jsonl" % (k, TAG)) for k in ("sets", "eval", "random", "train", "status")}
    sets_done = driver.jsonl_keys(p["sets"], ("seed", "set"))
    ev_done = driver.jsonl_keys(p["eval"], ("seed", "arm", "lam", "eval"))
    rn_done = driver.jsonl_keys(p["random"], ("seed", "arm", "eval"))
    fh = {k: open(v, "a") for k, v in p.items()}

    if os.environ.get("NO_TOKEN_PRELOAD") == "1":
        from config import config
        config.discovery.neg_context_selection.preload_negctx_tokens = False
    R = driver.Runner()
    import amp_eval_pass_v2 as V
    assert V.EVAL_SPLIT == "explicit" and V.CTR_SOURCE != "store", (V.EVAL_SPLIT, V.CTR_SOURCE)
    methods = {}

    def method_for(lam):
        """One method per lambda; the config is re-applied before every fit (the mask engine reads it at fit time)."""
        if lam not in methods:
            methods[lam] = R.method(driver.GAMMA, lam, True)
        R.H.configure(driver.GAMMA, lam, True, driver.STEPS)
        return methods[lam]

    RN = None
    if MODE == "run" and RANDOM_TARGETS & set(targets):
        import random_null as RN
        RN.install_support_wrapper()

    for n_, key in enumerate(targets):
        want_ev = [(key, a, L.lam_tag(l), e) for a in ARMS for l in LAMS for e in EVALS]
        want_rn = [(key, a, e) for a in ARMS for e in EVALS] if (RN and key in RANDOM_TARGETS and REF_IN_GRID) else []
        want_sets = [(key, s) for s in L.BUILD_ORDER]
        if (MODE == "run" and all(k in ev_done for k in want_ev) and all(k in rn_done for k in want_rn)) or (
                MODE == "sets" and all(k in sets_done for k in want_sets)):
            continue
        st = dict(seed=key, shard=TAG, device=DEVICE_NAME)
        t_all = time.time()
        try:
            copy_if_missing(SRC_OUT / "ctx" / ("%s.pt" % key), OUT / "ctx" / ("%s.pt" % key))
            rec = R.contexts(key)
            if rec.get("strong") is None:
                st["skip"] = "no_contrast" if rec.get("no_contrast") else "too_few_contexts"
                raise _Stop
            tv_arm, thin = R.train_arm(rec)
            st.update(train_arm=tv_arm, thin=thin)
            sets = Sets(R, key, rec, fh["sets"], sets_done)
            t = time.time()
            sets.build_all()
            st["t_sets"] = round(time.time() - t, 1)
            st["unusable"] = [s for s in L.BUILD_ORDER if sets.get(s).get("unusable")]
            st["ho_close_own"] = sets.ho.get((L.REFERENCE, L.REFERENCE))
            if MODE == "sets":
                st["ok"] = True
                raise _Stop
            if REUSE_MAIN and L.REFERENCE in ARMS and REF_IN_GRID:
                st["reused_main"] = copy_if_missing(
                    SRC_OUT / "main" / "circuits" / ("%s.pt" % key),
                    OUT / "circuits" / L.REFERENCE / L.lam_tag(L.REF_LAM) / ("%s.pt" % key))
            n_fits = 0
            for arm in ARMS:
                T = sets.get(arm)
                for lam in LAMS:
                    lt = L.lam_tag(lam)
                    if all((key, arm, lt, e) in ev_done for e in EVALS):
                        continue
                    label = dict(seed=key, arm=arm, lam=lt, lam_value=lam, train_arm=tv_arm, thin=thin)
                    if T.get("unusable"):
                        for e in EVALS:
                            fh["eval"].write(json.dumps(dict(label, eval=e, skip="arm_set_unusable", shard=TAG)) + "\n")
                        fh["eval"].flush()
                        continue
                    path = OUT / "circuits" / arm / lt / ("%s.pt" % key)
                    fresh = not path.exists()
                    try:
                        c, t_fit = R.fit(method_for(lam), dict(rec, neg=T["tokens"]), key, tv_arm, path,
                                         train_fh=fh["train"], label=dict(arm=arm, lam=lt))
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        for ev in EVALS:
                            fh["eval"].write(json.dumps(dict(label, eval=ev, shard=TAG, error="fit %s: %s" % (
                                type(e).__name__, str(e)[:300]))) + "\n")
                        fh["eval"].flush()
                        continue
                    n_fits += int(fresh)
                    if c is None:
                        for ev in EVALS:
                            fh["eval"].write(json.dumps(dict(label, eval=ev, skip="rejected", shard=TAG)) + "\n")
                        fh["eval"].flush()
                        continue
                    label.update(n_nodes=driver.n_nodes(c), t_fit=round(t_fit, 1), fresh_fit=fresh)
                    t_ev = score_battery(R, c, rec, sets, arm, fh["eval"], ev_done, label,
                                         lambda ev, _a=arm, _l=lt: (key, _a, _l, ev))
                    log("%-16s %-14s lam %-6s n=%-5s fit %5.1fs battery %5.1fs" % (
                        key, arm, lt, label["n_nodes"], t_fit, t_ev))
            st["n_fits"] = n_fits
            if key in RANDOM_TARGETS and RN is not None and REF_IN_GRID:
                st["n_random"] = random_baselines(R, RN, key, rec, tv_arm, thin, sets, method_for, fh["random"],
                                                  rn_done)
            st["ok"] = True
        except _Stop:
            pass
        except Exception as e:  # noqa: BLE001
            st["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        st["t_total"] = round(time.time() - t_all, 1)
        fh["status"].write(json.dumps(st) + "\n"); fh["status"].flush()
        log("%4d/%d %-16s %s sets %ss total %ss" % (n_ + 1, len(targets), key,
                                                     st.get("error") or st.get("skip") or "ok", st.get("t_sets"),
                                                     st["t_total"]))
    log("DONE")


def random_baselines(R, RN, key, rec, tv_arm, thin, sets, method_for, fh, done):
    """062 random_null per arm: site-matched to the arm's lambda 1e-3 circuit, members drawn uniformly from the
    latents live on the 48 strongest-train contexts, only the coefficients fitted (support-restricted engine, lambda
    0, gates open), on the arm's contrast set; scored on the battery."""
    from eval.ablation_faithfulness import upstream_sites
    live, n = None, 0
    for arm in ARMS:
        if all((key, arm, e) in done for e in EVALS):
            continue
        ref_path = OUT / "circuits" / arm / L.lam_tag(L.REF_LAM) / ("%s.pt" % key)
        T = sets.get(arm)
        if T.get("unusable") or not ref_path.exists():
            continue
        ref = torch.load(ref_path, weights_only=False)
        if ref is None:
            continue
        ref_sites = RN.members_by_site(ref)
        if live is None:
            layer, kind, _ = R.H.parse(key)
            live = RN.live_latents(R, rec["strong"]["pos"][rec["strong"]["train"]],
                                   set(upstream_sites(R.G["bank"], layer, kind)))
        rng = random.Random("%s-%s-null" % (key, arm))
        support = {s: rng.sample(sorted(live.get(s, ())), min(len(v), len(live.get(s, ())))) for s, v in ref_sites.items()}
        label = dict(seed=key, arm=arm, draw=0, train_arm=tv_arm, thin=thin,
                     n_ref=sum(len(v) for v in ref_sites.values()), n_drawn=sum(len(v) for v in support.values()),
                     short_sites=sum(1 for s, v in ref_sites.items() if len(live.get(s, ())) < len(v)))
        if not any(support.values()):                 # an empty support would run an ordinary (unrestricted) fit
            for e in EVALS:
                fh.write(json.dumps(dict(label, eval=e, skip="empty_support", shard=TAG)) + "\n")
            fh.flush()
            continue
        try:
            RN._SUPPORT.clear(); RN._SUPPORT.update(support)
            c, t_fit = R.fit(method_for(L.REF_LAM), dict(rec, neg=T["tokens"]), key, tv_arm,
                             OUT / "random" / "circuits" / arm / ("%s.d0.pt" % key))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            for ev in EVALS:
                fh.write(json.dumps(dict(label, eval=ev, shard=TAG, error="fit %s: %s" % (
                    type(e).__name__, str(e)[:300]))) + "\n")
            fh.flush()
            continue
        finally:
            RN._SUPPORT.clear()
        if c is None:
            for ev in EVALS:
                fh.write(json.dumps(dict(label, eval=ev, skip="rejected", shard=TAG)) + "\n")
            fh.flush()
            continue
        label.update(n_nodes=driver.n_nodes(c), t_fit=round(t_fit, 1))
        score_battery(R, c, rec, sets, arm, fh, done, label, lambda ev, _a=arm: (key, _a, ev))
        n += 1
    return n


if __name__ == "__main__":
    run()
