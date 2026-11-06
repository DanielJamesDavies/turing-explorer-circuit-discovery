"""092: experiment-local in-memory patches of src/circuit/instrument/learned_mask.py (src is NOT edited), on top of
090's sp_engine (v2 = sp_ste) and, for test 3, 091's combined engine (sp + 079's donor-resample term).

SPEED switches (module dict, read at fit time; all False = the parent engine, bit-identical):
  truncate   test 1. In data_loss, each training micro-batch's token windows are cut after the LATEST anchor in that
             batch: tk = tk[:, :max(an) + 1]. The model is causal (turingllm.py: SDPA is_causal=True, absolute wpe,
             RMSNorm per token, no padding / attention mask), every SAE encode / Top-K / fill is per position, and the
             loss reads only (a) the target pre-activation at each row's anchor and (b) the rank-keep term's other site
             latents AT the anchor (pat.seed_x at an). Means / floors / _rank_clean are constants precomputed over full
             windows before training, read per latent or at anchors. So positions after max(an) cannot change the loss.
             Batches are contiguous slices tokens[s:s+4] in a fixed order; sorting rows by anchor would change which
             rows share a batch, so the cut is per batch as is.
  combine    test 2. The Z, C and A terms (same rows, anchors, targets; three patchers over the same thetas / amps)
             run as ONE forward over the micro-batch repeated 3x along the batch dimension; CombinedPatcher encodes
             and decodes once per site and applies each row group's own floor (Z: none, C: contrast mean, A: posctx
             mean) and, for C / A, the sparsity-preserving fill (090 _sp_delta, per group). Each group's loss is its
             term's loss exactly as data_loss builds it (mse at the anchors + rank-keep), weighted as now; one
             backward of the sum (= the sum of the three separate backwards, up to float order).
  grab       step-0 diagnostics: the data terms (full precision) and the gradients of thetas / amps from the data
             terms alone (before the penalty), stored in SPEED["grads"].
  drop_floor test 3. The C term (patcher = the contrast-mean floor) is removed from the training terms.
  sp_donor   test 3. The donor-resample patcher (079, is_resample) gets the sparsity-preserving fill, the training
             analogue of 073's ResamplePatcher: members at alpha x live, then the donor's non-member active entries at
             that position, largest first, up to the Top-K budget K - (#active members); everything else 0
             (sp_keep_mask_donor; forward hard, backward straight-through under SPFILL ste).
"""
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "090-sparse-fill-training"))
sys.path.insert(0, str(EXP / "091-v2-resample-small"))
import sp_engine  # noqa: E402

SRC = sp_engine.SRC
MODNAME = sp_engine.MODNAME

EDITS = [
    # 1. truncate each micro-batch after its latest anchor (test 1)
    ("        if tk.shape[0] == 0:\n"
     "            s = 0\n"
     "            tk, an, tg = tokens[:micro_bs], anchors[:micro_bs], targets[:micro_bs].to(device)\n",
     "        if tk.shape[0] == 0:\n"
     "            s = 0\n"
     "            tk, an, tg = tokens[:micro_bs], anchors[:micro_bs], targets[:micro_bs].to(device)\n"
     "        if SPEED[\"truncate\"] and objective != \"logit\":           # 092 test 1\n"
     "            _L = int(an.max()) + 1\n"
     "            SPEED[\"tok_full\"] += int(tk.shape[0]) * int(tk.shape[1]); SPEED[\"tok_used\"] += int(tk.shape[0]) * _L\n"
     "            tk = tk[:, :_L]\n"),
    # 2. the combined Z / C / A forward (test 2); the original per-term loop runs over () when it is on
    ("                for ti, (tokens_t, anchors_t, targets_t, w, pat_t) in enumerate(terms):\n",
     "                _comb = bool(SPEED[\"combine\"])                      # 092 test 2\n"
     "                if _comb:\n"
     "                    with _phase(\"fit.fwd\"):                    # forward + backward inside\n"
     "                        _parts = _combined_parts(terms, mi, micro_bs, accum, pt_tr, objective, device, inference,\n"
     "                                                 _rank_clean if rank_weight > 0 else None, rank_weight, rank_mode,\n"
     "                                                 rank_delta, _ot_W, _ot_b, seed_latent_idx, bank, _rank_log)\n"
     "                    for ti, part in enumerate(_parts):\n"
     "                        _pd = part.detach()\n"
     "                        step_total = _pd if step_total is None else step_total + _pd\n"
     "                        if ti < len(step_terms):\n"
     "                            step_terms[ti] = step_terms[ti] + _pd\n"
     "                        else:\n"
     "                            step_terms.append(_pd)\n"
     "                for ti, (tokens_t, anchors_t, targets_t, w, pat_t) in enumerate(() if _comb else terms):\n"),
    # 3. step-0 diagnostics: data terms and data-term gradients, before the penalty's backward
    ("            if objective in (\"negctx\", \"raise\"):\n"
     "                penalty = l1_lambda * edit_sum()\n",
     "            if SPEED[\"grab\"] and step == 0:                          # 092 diagnostics\n"
     "                SPEED[\"grads\"] = dict(\n"
     "                    terms=[float(t_) for t_ in step_terms],\n"
     "                    theta={s_: t_.grad.detach().clone() for s_, t_ in thetas.items() if t_.grad is not None},\n"
     "                    amp={s_: t_.grad.detach().clone() for s_, t_ in (amps or {}).items() if t_.grad is not None})\n"
     "            if objective in (\"negctx\", \"raise\"):\n"
     "                penalty = l1_lambda * edit_sum()\n"),
    # 4. drop the C term (test 3)
    ("                if not term_names:\n",
     "                if SPEED[\"drop_floor\"]:                             # 092 test 3: no C term\n"
     "                    terms = [t_ for t_ in terms if t_[4] is not patcher]\n"
     "                if not term_names:\n"),
]

# test 3, applied to 091's combined source: the donor patcher gets the sparsity-preserving donor fill
DONOR_OLD = "                    if SPFILL[\"mode\"] != \"dense\" and not getattr(self, \"is_resample\", False):  # 091\n"
DONOR_NEW = ("                    if SPFILL[\"mode\"] != \"dense\" and getattr(self, \"is_resample\", False) and SPEED[\"sp_donor\"]:\n"
             "                        delta_code = _sp_delta_donor(self, (layer_idx, kind), dense, m, alpha, floor)  # 092\n"
             "                    elif SPFILL[\"mode\"] != \"dense\" and not getattr(self, \"is_resample\", False):  # 091\n")

FOOTER = '''

# ---------------------------------------------------------------------------------------------- 092 additions
SPEED = {"truncate": False, "combine": False, "grab": False, "grads": None, "drop_floor": False, "sp_donor": False,
         "tok_full": 0, "tok_used": 0, "combine_rows": None}


class CombinedPatcher:
    """The Z / C / A patchers (same thetas, amps, temperature, binarize) run on ONE batch of G x B rows: row group g
    uses pats[g]'s floor (and the sparsity-preserving fill when it has one). One encode, one decode per site."""

    def __init__(self, pats, B):
        p0 = pats[0]
        for p in pats[1:]:
            assert p.thetas is p0.thetas and p.amps is p0.amps and p.binarize == p0.binarize
            assert p.bin_threshold == p0.bin_threshold and p.temperature == p0.temperature
            assert p.code_dtype == p0.code_dtype and p.tap_seed == p0.tap_seed
        for p in pats:
            assert not p.deltas and not p.pins and p.offtarget_ref is None
        assert _MARGIN[0] is None
        self.pats, self.B, self.p0 = pats, int(B), p0
        self.seed_pre = None
        self.seed_x = None
        self.capture_site = any(p.capture_site for p in pats)

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        p0 = self.p0
        if p0.tap_seed and layer_idx == p0.seed_layer and kind == p0.seed_kind:
            w = p0.w_seed.to(device=x.device, dtype=x.dtype)
            b = p0.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b
            if self.capture_site:
                self.seed_x = x
            return x
        site = (layer_idx, kind)
        theta = p0.thetas.get(site)
        if theta is None:
            return x
        ta, ti = p0.bank.encode(x, kind, layer_idx)
        code_dt = torch.float32 if p0.code_dtype == "fp32" else x.dtype
        dense = sparse_topk_to_dense(ta, ti, p0.bank.d_sae, dtype=code_dt)
        if p0.binarize == "anneal":
            m = torch.sigmoid(theta / max(p0.temperature, 1e-4))
        else:
            m = torch.sigmoid(theta)
        m = m.to(device=dense.device, dtype=dense.dtype)
        if p0.binarize == "ste":
            hard = (m > p0.bin_threshold).to(m.dtype)
            m = hard + m - m.detach()
        amp = p0.amps.get(site)
        assert amp is not None
        alpha = _amp_of(amp).to(device=dense.device, dtype=dense.dtype)
        B = self.B
        out_d = []
        for g, p in enumerate(self.pats):
            d = dense[g * B:(g + 1) * B]
            dc = d * (m * alpha - 1.0)
            floor = p.floors.get(site)
            if floor is not None:
                dc = dc + (1.0 - m) * p._const("floor", site, floor, d)
                if SPFILL["mode"] != "dense":
                    dc = _sp_delta(p, site, d, dc, m, alpha, floor)
            out_d.append(dc)
        delta_code = torch.cat(out_d, 0)
        out = p0.bank.decode(delta_code, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


def _combined_parts(terms, mi, micro_bs, accum, pt_tr, objective, device, inference, rank_clean, rank_weight,
                    rank_mode, rank_delta, ot_W, ot_b, seed_idx, bank, rank_log):
    """Per-term (w * data_loss / accum) for all terms from ONE forward. Same slice, same loss as data_loss."""
    assert objective == "pos", objective
    toks, anc, tgs = terms[0][0], terms[0][1], terms[0][2]
    for t in terms:
        assert t[0] is toks and t[1] is anc and t[2] is tgs and t[4] is not None
    s = (mi * micro_bs) % max(int(toks.shape[0]), 1)
    tk, an = toks[s:s + micro_bs], anc[s:s + micro_bs]
    tg = tgs[s:s + micro_bs].to(device)
    if tk.shape[0] == 0:
        s = 0
        tk, an, tg = toks[:micro_bs], anc[:micro_bs], tgs[:micro_bs].to(device)
    if SPEED["truncate"]:
        _L = int(an.max()) + 1
        SPEED["tok_full"] += int(tk.shape[0]) * int(tk.shape[1]); SPEED["tok_used"] += int(tk.shape[0]) * _L
        tk = tk[:, :_L]
    G, B = len(terms), int(tk.shape[0])
    # rows per group per forward: all B (one forward), or SPEED["combine_rows"] (memory fallback: several forwards,
    # each chunk's mean weighted by its share of the B rows, so the sum is the same mean over B)
    cr = int(SPEED.get("combine_rows") or B)
    totals = [None] * G
    for c0 in range(0, B, cr):
        tkc, anc_ = tk[c0:c0 + cr], an[c0:c0 + cr]
        tgc = tg[c0:c0 + cr]
        nb = int(tkc.shape[0])
        frac = float(nb) / float(B)
        cp = CombinedPatcher([t[4] for t in terms], nb)
        pre = _forward_preact(inference, cp, tkc.repeat(G, 1), grad=True)
        parts = []
        for g, (_k, _a, _t, w, pat) in enumerate(terms):
            vals = _at(pre[g * nb:(g + 1) * nb], anc_)
            tgt = tgc[:vals.shape[0]].to(vals.device, vals.dtype)
            loss = ((vals - tgt) ** 2).mean()
            if pat.capture_site and toks is pt_tr:
                xa = _at(cp.seed_x[g * nb:(g + 1) * nb], anc_)
                pre_all = xa.to(ot_W.dtype) @ ot_W.T + ot_b
                if rank_weight > 0:
                    assert rank_mode == "keep", rank_mode
                    _others = pre_all.index_fill(-1, torch.tensor([int(seed_idx)], device=pre_all.device),
                                                 float("-inf"))
                    _ps = pre_all[:, int(seed_idx)]
                    _r = rank_clean[s + c0:s + c0 + _ps.shape[0]]
                    _tops = torch.topk(_others, int(_r.max()), dim=-1).values
                    _tau = _tops.gather(1, (_r - 1).long()[:, None]).squeeze(1)
                    _pen = torch.relu(_tau + float(rank_delta) - _ps) ** 2
                    loss = loss + float(rank_weight) * _pen.mean()
                    if cr >= B:
                        rank_log.append((float(rank_weight) * _pen.mean()).detach())
            parts.append(w * loss * frac / accum)
        torch.stack([p_.float() for p_ in parts]).sum().backward()
        cp.seed_pre = cp.seed_x = None
        for g, p_ in enumerate(parts):
            totals[g] = p_.detach() if totals[g] is None else totals[g] + p_.detach()
        del pre, parts, cp
    return totals


def sp_keep_mask_donor(dense, member, alpha, donor, K):
    """H [B, T, d] bool, the training analogue of 073 ResamplePatcher's fill: members kept; the donor's non-member
    entries with value > 0 at that position, largest first, up to K - (#active members); the rest dropped.
    dense / donor [B, T, d] in the code dtype, member [d] bool."""
    D = int(dense.shape[-1])
    mem = member.view(1, 1, D)
    a_c = alpha.to(device=dense.device, dtype=dense.dtype)
    act = ((dense * a_c) != 0) & mem
    budget = (int(K) - act.sum(dim=-1)).clamp(min=0)
    valid = (donor > 0) & ~mem
    vals, idx = donor.float().masked_fill(~valid, float("-inf")).topk(int(K), dim=-1)
    use = (torch.arange(int(K), device=dense.device).view(1, 1, -1) < budget.unsqueeze(-1)) & torch.isfinite(vals)
    fill = torch.zeros_like(valid).scatter_(-1, idx, use)
    return fill | mem


def _sp_delta_donor(self, site, dense, m, alpha, floor):
    mode = SPFILL["mode"]
    if mode == "curriculum":
        raise ValueError("092: donor sp fill supports hard / ste only")
    K = int(SPFILL["k"] if SPFILL["k"] is not None else self.bank.k)
    f = self._const("floor", site, floor, dense)
    with torch.no_grad():
        member = (m.detach() > self.bin_threshold).to(dense.device)
        H = sp_keep_mask_donor(dense, member, alpha.detach(), f, K)
        st = SPFILL.setdefault("stats", {})
        st["donor_calls"] = st.get("donor_calls", 0) + 1
    code = m * alpha * dense + (1.0 - m) * f
    return sp_code(code, H, mode) - dense
'''


def _apply(text, edits, tag):
    for anchor, repl in edits:
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError("092 %s: anchor found %d times (need 1):\n%r" % (tag, n, anchor[:200]))
        text = text.replace(anchor, repl)
    return text


def source_v2():
    """090 sp_engine + the 092 edits (tests 1, 2)."""
    return _apply(sp_engine.patched_source(), EDITS, "v2") + FOOTER


def source_v2r():
    """091 combined engine (sp + 079 donor term) + the 092 edits + the donor sp fill (test 3)."""
    import engine as e91
    text = _apply(e91.patched_source(), EDITS + [(DONOR_OLD, DONOR_NEW)], "v2r")
    return text + FOOTER


def install(source, tag="092"):
    import circuit.instrument as pkg
    mod = types.ModuleType(MODNAME)
    mod.__file__ = str(SRC)
    mod.__package__ = "circuit.instrument"
    sys.modules[MODNAME] = mod
    exec(compile(source, "<%s patched %s>" % (tag, SRC.name), "exec"), mod.__dict__)
    pkg.learned_mask = mod
    return mod


def reset(mod):
    mod.SPEED.update(truncate=False, combine=False, grab=False, grads=None, drop_floor=False, sp_donor=False,
                     tok_full=0, tok_used=0, combine_rows=None)
    sp_engine.set_mode(mod, "dense")
    if hasattr(mod, "RESAMPLE"):
        mod.RESAMPLE.update(weight=0.0, make=None)
        mod.SILENT.update(weight=0.0, fn=None)
