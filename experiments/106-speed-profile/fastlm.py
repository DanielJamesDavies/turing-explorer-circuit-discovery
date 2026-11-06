"""106 prototypes: a patched in-memory copy of src/circuit/instrument/learned_mask.py (src is NOT edited).

install(flags) reads learned_mask.py's text, applies exact string replacements (each must match exactly once, so a
changed engine fails loudly instead of silently running the original), executes the result as a new module and points
circuit.instrument.learned_mask.run_learned_mask at it (ablation_gradient imports run_learned_mask at call time, so the
production call path picks it up unchanged). Each replacement is one proposed hunk for adoption.

Flags (all expected bit-identical, i.e. no change of arithmetic; checked by bench.py):
  early_exit   stop the pre-activation forward at the seed's site (the layers after the target's site do work nobody
               reads: the seed tap captures seed_pre and the loss reads only that). An exception raised in the seed
               tap ends inference.forward; every context manager on that path restores state in a finally.
  donor_gpu    DonorPool keeps the donor codes on the GPU and moves the drawn rows once per draw (pinned,
               non-blocking) instead of indexing CPU tensors and copying 2 pageable tensors per site per R forward
               (each pageable H2D copy blocks the host until the stream drains).
  rank_nosync  rank-keep ("keep" mode): topk with k = the max clean rank over all training anchors, computed once,
               instead of int(_r.max()) per term (a device->host sync per term per step); the seed index tensor is
               built once instead of torch.tensor([_sd], device=...) per term (an H2D copy per term per step).
               topk values are sorted, so the r-th value is the same for any k >= r: identical arithmetic.
Optional (may change results; reported separately):
  triton_topk  grad-mode SAE encode: indices from the Triton radix-select (no grad), values gathered from the same
               pre-activations (differentiable). Identical unless the k-th value ties (then a different tied latent
               may be picked).
  batch2 / batch4  one forward + one backward per group of 2 / 4 data terms (Z, C, A, R share the step's token rows;
               the rows are stacked [Z | C | A | R] through one model forward, one SAE encode and one decode per site;
               each term keeps its own per-chunk code, keep mask, donor draw, loss and rank-keep term). Not
               bit-identical: the gemms run at 2-4x the rows (other cuBLAS kernels) and theta's gradient is summed
               inside one backward. Memory: the group's graphs are alive together (~group size x one term's).
"""
import sys
import types
from pathlib import Path

FLAGS_ALL = ("early_exit", "donor_gpu", "rank_nosync", "triton_topk", "batch2", "batch4", "compile_sp")
# compile_sp (may change results at float-noise level): torch.compile (inductor, emulate_precision_casts) of the
#   sparse fill's elementwise chain: the keep mask after the eager argsort (integer / bool ops: exact) and the
#   differentiable code + straight-through tail (forward rounding emulated; the backward's reductions may sum in
#   another order). Fuses ~15 full [B, T, d_sae] passes per C / A site-forward into a few kernels.

_COMPILE_SRC = '''


def _keep_rest(dense, member, a_c, rank_of, K: int):
    D = dense.shape[-1]
    act = ((dense * a_c) != 0) & member.view(1, 1, D)
    budget = (int(K) - act.sum(dim=-1)).clamp(min=0)
    fill = (rank_of.view(1, 1, D) < budget.unsqueeze(-1)) & ~member.view(1, 1, D)
    return fill | member.view(1, 1, D)


def _sp_tail(m, alpha, dense, f, H):
    code = m * alpha * dense + (1.0 - m) * f
    return sparse_code(code, H, "ste") - dense


import torch._inductor.config as _icfg
_icfg.emulate_precision_casts = True
_keep_rest_c = torch.compile(_keep_rest, dynamic=False, fullgraph=True)
_sp_tail_c = torch.compile(_sp_tail, dynamic=False, fullgraph=True)


def _sparse_keep_mask_c(dense, member, alpha, floor_c, K):
    D = int(dense.shape[-1])
    mean_rank = floor_c.clone().to(torch.float32)
    mean_rank[member] = float("-inf")
    ranked = torch.argsort(mean_rank, descending=True)
    rank_of = torch.empty_like(ranked)
    rank_of[ranked] = torch.arange(D, device=ranked.device)
    a_c = alpha.to(device=dense.device, dtype=dense.dtype)
    return _keep_rest_c(dense, member, a_c, rank_of, int(K))
'''

_MULTI_SRC = '''

class _MultiPatcher:
    """106 batch_terms: the patchers of a group of terms over stacked rows [term 0 | term 1 | ...] (b rows each)."""

    def __init__(self, pats, b):
        p0 = pats[0]
        self.pats, self.b, self.bank = list(pats), int(b), p0.bank
        self.tap_seed, self.seed_layer, self.seed_kind = p0.tap_seed, p0.seed_layer, p0.seed_kind
        self.w_seed, self.b_seed, self.code_dtype = p0.w_seed, p0.b_seed, p0.code_dtype
        self.capture_site = any(p.capture_site for p in pats)
        self.seed_pre = None
        self.seed_x = None
        self.donors = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if self.tap_seed and layer_idx == self.seed_layer and kind == self.seed_kind:
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            bb = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = torch.cat([xc @ w + bb for xc in x.split(self.b)], 0)
            if self.capture_site:
                self.seed_x = x
            if getattr(self, "_stop_at_seed", False):
                raise _SeedReached()
            return x
        p0 = self.pats[0]
        if p0.thetas.get((layer_idx, kind)) is None:
            return x
        ta, ti = self.bank.encode(x, kind, layer_idx)
        code_dt = torch.float32 if self.code_dtype == "fp32" else x.dtype
        dense = sparse_topk_to_dense(ta, ti, self.bank.d_sae, dtype=code_dt)
        parts = [p._delta_from_dense(layer_idx, kind, dense[i * self.b:(i + 1) * self.b])
                 for i, p in enumerate(self.pats)]
        out = self.bank.decode(torch.cat(parts, 0), kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


def _lmp_delta_from_dense(self, layer_idx, kind, dense):
    """transform()'s free-amplitude, unpinned branch on a given dense code (106 batch_terms); same expressions."""
    theta = self.thetas.get((layer_idx, kind))
    if self.binarize == "anneal":
        m = torch.sigmoid(theta / max(self.temperature, 1e-4))
    else:
        m = torch.sigmoid(theta)
    m = m.to(device=dense.device, dtype=dense.dtype)
    if self.binarize == "ste":
        hard = (m > self.bin_threshold).to(m.dtype)
        m = hard + m - m.detach()
    floor = self.floors.get((layer_idx, kind))
    if self.donors is not None:
        floor = self.donors.get((layer_idx, kind))
    amp = self.amps.get((layer_idx, kind))
    alpha = _amp_of(amp).to(device=dense.device, dtype=dense.dtype)
    if floor is not None and self.train_fill == "sparse":
        return self._sparse_delta((layer_idx, kind), dense, m, alpha, floor)
    delta_code = dense * (m * alpha - 1.0)
    if floor is not None:
        delta_code = delta_code + (1.0 - m) * self._const("floor", (layer_idx, kind), floor, dense)
    return delta_code


LearnedMaskPatcher._delta_from_dense = _lmp_delta_from_dense
'''

_BATCH_FNS = '''
    def _batchable(terms_):
        return (objective == "pos" and _MARGIN[0] is None and not deltas and _pins is None and bool(amps)
                and all(t[0] is pt_tr and t[4] is not None and not t[4].pins for t in terms_))

    def _batched_parts(terms_g, micro_index):
        """106 batch_terms: data_loss for a group of terms from ONE forward over stacked rows."""
        s = (micro_index * micro_bs) % max(int(pt_tr.shape[0]), 1)
        tk, an = pt_tr[s:s + micro_bs], pa_tr[s:s + micro_bs]
        tg = ptgt_tr[s:s + micro_bs].to(device)
        if tk.shape[0] == 0:
            s = 0
            tk, an, tg = pt_tr[:micro_bs], pa_tr[:micro_bs], ptgt_tr[:micro_bs].to(device)
        b = int(tk.shape[0])
        pats = [t[4] for t in terms_g]
        for p_ in pats:
            if p_.donors is not None:
                p_.donors.draw(b)                       # the draw the term's own forward would make, in term order
        mp = _MultiPatcher(pats, b)
        pre = _forward_preact(inference, mp, tk.repeat(len(pats), 1), grad=True)
        parts = []
        for i, (_tk, _an, _tg, w, pat) in enumerate(terms_g):
            vals = _at(pre[i * b:(i + 1) * b], an)
            tgt = tg[:vals.shape[0]].to(vals.device, vals.dtype)
            loss = ((vals - tgt) ** 2).mean()
            if pat.capture_site:
                xa = _at(mp.seed_x[i * b:(i + 1) * b], an)
                pre_all = xa.to(_ot_W.dtype) @ _ot_W.T + _ot_b
                if pat.offtarget_ref is not None:
                    ref = pat.offtarget_ref[s:s + pre_all.shape[0]]
                    excess = torch.relu(pre_all - ref)
                    loss = loss + float(offtarget_weight) * (excess ** 2).sum(-1).mean()
                if rank_weight > 0:
                    _sd = int(seed_latent_idx)
                    _others = pre_all.index_fill(-1, (_sd_idx if _RANK_NOSYNC else
                                                      torch.tensor([_sd], device=pre_all.device)),
                                                 float("-inf"))
                    _ps = pre_all[:, _sd]
                    if rank_mode == "cut":
                        _tau = torch.topk(_others, int(bank.k), dim=-1).values[:, -1]
                        _pen = torch.relu(_tau + float(rank_delta) - _ps) ** 2
                    elif rank_mode == "keep":
                        _r = _rank_clean[s:s + _ps.shape[0]]
                        _tops = torch.topk(_others, (_r_kmax if _RANK_NOSYNC else int(_r.max())),
                                           dim=-1).values
                        _tau = _tops.gather(1, (_r - 1).long()[:, None]).squeeze(1)
                        _pen = torch.relu(_tau + float(rank_delta) - _ps) ** 2
                    else:
                        _T = _rank_T[s:s + _ps.shape[0]]
                        _above = torch.sigmoid((_others - _ps[:, None]) / _T[:, None]).sum(-1)
                        _pen = float(dual_norm) * (_above / float(bank.k)) ** float(rank_power)
                    loss = loss + float(rank_weight) * _pen.mean()
                    _rank_log.append((float(rank_weight) * _pen.mean()).detach())
            parts.append(w * loss / accum)
        return parts

'''


def _sub(src, old, new, label):
    n = src.count(old)
    if n != 1:
        raise RuntimeError("fastlm patch %s: expected 1 match, found %d" % (label, n))
    return src.replace(old, new)


def build_source(flags):
    import circuit.instrument.learned_mask as LM
    src = Path(LM.__file__).read_text(encoding="utf-8")
    fl = set(flags)
    group = 4 if "batch4" in fl else 2 if "batch2" in fl else 1
    hdr = ("\n_EARLY_EXIT = %r\n_DONOR_GPU = %r\n_RANK_NOSYNC = %r\n_BATCH_GROUP = %d\n\n\n"
           "class _SeedReached(Exception):\n"
           "    \"\"\"106 early exit: raised by the seed tap once seed_pre is captured.\"\"\"\n\n"
           % ("early_exit" in fl, "donor_gpu" in fl, "rank_nosync" in fl, group))
    src = _sub(src, "Site = Tuple[int, str]\n", "Site = Tuple[int, str]\n" + hdr, "header")
    # --- compile_sp ---------------------------------------------------------------------------------------------
    if "compile_sp" in fl:
        src = src + _COMPILE_SRC
        src = _sub(src,
                   "                H = sparse_keep_mask(dense, member, alpha.detach(), f, int(self.bank.k))\n",
                   "                H = _sparse_keep_mask_c(dense, member, alpha.detach(), f, int(self.bank.k))\n",
                   "compile_sp.keep")
        src = _sub(src,
                   "        code = m * alpha * dense + (1.0 - m) * f\n"
                   "        return sparse_code(code, H, self.sparse_grad) - dense\n",
                   "        if self.sparse_grad == \"ste\":\n"
                   "            return _sp_tail_c(m, alpha, dense, f, H)\n"
                   "        code = m * alpha * dense + (1.0 - m) * f\n"
                   "        return sparse_code(code, H, self.sparse_grad) - dense\n", "compile_sp.tail")
    # --- batch_terms (inert at _BATCH_GROUP 1: every group is one term through the original data_loss) ----------
    src = src + _MULTI_SRC
    src = _sub(src, "    # TERM SCALING. Both dual terms measure", _BATCH_FNS + "    # TERM SCALING. Both dual terms measure",
               "batch.fns")
    src = _sub(src,
               "                for ti, (tokens_t, anchors_t, targets_t, w, pat_t) in enumerate(terms):\n"
               "                    with _phase(\"fit.fwd\"):\n"
               "                        part = w * data_loss(tokens_t, anchors_t, targets_t,\n"
               "                                             mi, pat_t) / accum\n"
               "                    with _phase(\"fit.bwd\"):\n"
               "                        part.backward()\n"
               "                    _pd = part.detach()\n",
               "                if _BATCH_GROUP > 1 and _batchable(terms):\n"
               "                    _groups = [list(range(g_, min(g_ + _BATCH_GROUP, len(terms))))\n"
               "                               for g_ in range(0, len(terms), _BATCH_GROUP)]\n"
               "                else:\n"
               "                    _groups = [[ti_] for ti_ in range(len(terms))]\n"
               "                _gparts = []\n"
               "                for _g in _groups:\n"
               "                    with _phase(\"fit.fwd\"):\n"
               "                        if len(_g) == 1:\n"
               "                            tokens_t, anchors_t, targets_t, w, pat_t = terms[_g[0]]\n"
               "                            _ps_ = [w * data_loss(tokens_t, anchors_t, targets_t,\n"
               "                                                  mi, pat_t) / accum]\n"
               "                        else:\n"
               "                            _ps_ = _batched_parts([terms[t_] for t_ in _g], mi)\n"
               "                    _tot_ = _ps_[0]\n"
               "                    for _p_ in _ps_[1:]:\n"
               "                        _tot_ = _tot_ + _p_\n"
               "                    with _phase(\"fit.bwd\"):\n"
               "                        _tot_.backward()\n"
               "                    _gparts += [p_.detach() for p_ in _ps_]\n"
               "                for ti, _pd in enumerate(_gparts):\n", "batch.loop")
    # --- early_exit -------------------------------------------------------------------------------------------
    src = _sub(src,
               "            self.seed_pre = _pre\n"
               "            if self.capture_site:\n"
               "                self.seed_x = x\n"
               "            return x\n",
               "            self.seed_pre = _pre\n"
               "            if self.capture_site:\n"
               "                self.seed_x = x\n"
               "            if getattr(self, \"_stop_at_seed\", False):\n"
               "                raise _SeedReached()\n"
               "            return x\n", "early_exit.tap")
    src = _sub(src,
               "    try:\n"
               "        inference.forward(tokens, patcher=patcher, grad_enabled=grad,\n"
               "                          return_activations=False, tokenize_final=False)\n"
               "    finally:\n"
               "        if was_compiled:\n"
               "            inference.enable_compile()\n"
               "    if patcher.seed_pre is None:\n",
               "    patcher._stop_at_seed = _EARLY_EXIT\n"
               "    try:\n"
               "        inference.forward(tokens, patcher=patcher, grad_enabled=grad,\n"
               "                          return_activations=False, tokenize_final=False)\n"
               "    except _SeedReached:\n"
               "        pass\n"
               "    finally:\n"
               "        patcher._stop_at_seed = False\n"
               "        if was_compiled:\n"
               "            inference.enable_compile()\n"
               "    if patcher.seed_pre is None:\n", "early_exit.forward")
    # --- donor_gpu --------------------------------------------------------------------------------------------
    src = _sub(src,
               "        self.codes, self.d_sae, self.device = codes, int(d_sae), device\n",
               "        self.codes, self.d_sae, self.device = codes, int(d_sae), device\n"
               "        self.rows_dev = None\n"
               "        if _DONOR_GPU:\n"
               "            self.codes = {s: (a.to(device), i.to(device)) for s, (a, i) in codes.items()}\n",
               "donor_gpu.init")
    src = _sub(src,
               "        self.rows = torch.randint(self.n, (int(B),), generator=self.gen)\n"
               "        self.n_draws += 1\n",
               "        self.rows = torch.randint(self.n, (int(B),), generator=self.gen)\n"
               "        if _DONOR_GPU:\n"
               "            self.rows_dev = (self.rows.pin_memory().to(self.device, non_blocking=True)\n"
               "                             if torch.device(self.device).type == \"cuda\" else self.rows.to(self.device))\n"
               "        self.n_draws += 1\n", "donor_gpu.draw")
    src = _sub(src,
               "        r = self.rows.to(acts.device)\n",
               "        r = self.rows_dev if self.rows_dev is not None else self.rows.to(acts.device)\n",
               "donor_gpu.get")
    # --- rank_nosync ------------------------------------------------------------------------------------------
    src = _sub(src,
               "        _rank_clean = ((_clean_others > _clean_seed[:, None]).sum(-1) + 1).clamp(1, int(bank.k)).to(device)\n",
               "        _rank_clean = ((_clean_others > _clean_seed[:, None]).sum(-1) + 1).clamp(1, int(bank.k)).to(device)\n"
               "        _r_kmax = int(_rank_clean.max())\n"
               "        _sd_idx = torch.tensor([int(seed_latent_idx)], device=device)\n", "rank_nosync.pre")
    src = _sub(src,
               "                _others = pre_all.index_fill(-1, torch.tensor([_sd], device=pre_all.device),\n",
               "                _others = pre_all.index_fill(-1, (_sd_idx if _RANK_NOSYNC else\n"
               "                                                  torch.tensor([_sd], device=pre_all.device)),\n",
               "rank_nosync.index")
    src = _sub(src,
               "                    _tops = torch.topk(_others, int(_r.max()), dim=-1).values\n",
               "                    _tops = torch.topk(_others, (_r_kmax if _RANK_NOSYNC else int(_r.max())),\n"
               "                                       dim=-1).values\n", "rank_nosync.topk")
    return src


_STATE = {"mod": None, "orig_run": None, "orig_encode": None}


def install(flags):
    """Install the patched engine (flags subset of FLAGS_ALL). Returns the new module. install(()) restores src."""
    import circuit.instrument.learned_mask as LM
    if _STATE["orig_run"] is None:
        _STATE["orig_run"] = LM.run_learned_mask
    bad = set(flags) - set(FLAGS_ALL)
    if bad:
        raise ValueError("unknown fastlm flags %s" % sorted(bad))
    _set_triton_topk("triton_topk" in flags)
    eng = [f for f in flags if f != "triton_topk"]
    if not eng:
        LM.run_learned_mask = _STATE["orig_run"]
        _STATE["mod"] = LM
        return LM
    src = build_source(eng)
    mod = types.ModuleType("circuit.instrument.learned_mask_106")
    mod.__file__ = LM.__file__ + "<106>"
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    LM.run_learned_mask = mod.run_learned_mask
    _STATE["mod"] = mod
    return mod


def _set_triton_topk(on):
    import torch
    import sae.topk_sae as TS
    if _STATE["orig_encode"] is None:
        _STATE["orig_encode"] = TS.SAE.encode
    orig = _STATE["orig_encode"]
    if not on:
        TS.SAE.encode = orig
        return
    from sae.triton_topk import topk_nonneg_bf16

    def encode(self, x):
        if not torch.is_grad_enabled():
            return orig(self, x)
        if x.dtype != self.encoder.weight.dtype:
            x = x.to(self.encoder.weight.dtype)
        pre = torch.relu(torch.nn.functional.linear(x, self.encoder.weight, self._get_bias_eff()))
        if pre.dtype != torch.bfloat16 or not pre.is_cuda:
            return pre.topk(self.k, sorted=False, dim=-1)
        with torch.no_grad():
            _v, idx = topk_nonneg_bf16(pre.detach(), self.k)
        return pre.gather(-1, idx), idx
    TS.SAE.encode = encode


def current_module():
    import circuit.instrument.learned_mask as LM
    return _STATE["mod"] or LM
