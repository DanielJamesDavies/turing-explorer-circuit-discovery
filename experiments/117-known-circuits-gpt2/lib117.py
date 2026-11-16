"""Shared pieces of 117 (DAN-215): known-circuit check on GPT-2 small with WCM v4 latent targets.

  setup_env()       env for 062's driver.Runner BEFORE it is imported: FAMILY gpt2-small, the production data-pass stores
                    (unpacked from the 2026-11-07 pod tarball into out/stores), WCM_VERSION v4, OUT out/wcm, lazy SAEs,
                    DEEP_BS 1 (the 16 GB card; same effective batch)
  HEADS / CLASSES   the published components (Wang et al. 2022 IOI; Hanna et al. 2023 greater-than)
  task rows         081's task files (IOI: Wang's templates, disjoint name halves; greater-than: Hanna's years, disjoint
                    noun halves), the role label per token, END = the prompt's last token (the prediction position)
  task_record()     a production-format context record (circuit/protocol_contexts) built from task prompts instead of
                    corpus windows, so the fit and the scorer run unchanged
  Attrib            per-slot attribution of the target's pre-activation at its anchor, in a chosen frame (members only =
                    the circuit's Z frame; all latents = the full model), with the exact per-head decomposition of
                    attention-out latents (081 engine.head_drive)
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments"
OUT = HERE / "out"
STORES_ROOT = OUT / "stores"
TASKS = EXP / "081-task-circuits" / "tasks" / "gpt2-small"
EOS = 50256
WIN = 64
N_HEADS = 12

for _p in (str(ROOT / "src"), str(EXP / "062-h100-protocol-v1"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def setup_env():
    e = os.environ
    e["FAMILY"] = "gpt2-small"
    e["DATA"] = str(STORES_ROOT / "data_public" / "gpt2-small")
    e["STORES"] = str(STORES_ROOT / "outputs_public" / "gpt2-small")
    e["WCM_VERSION"] = "v4"
    e.setdefault("OUT", str(OUT / "wcm"))
    e["SAE_PRELOAD"] = "0"
    e.setdefault("DEEP_BS", "1")
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    e["EVAL_SPLIT"] = "explicit"
    e.setdefault("TAG", "dan215_unused")
    e.setdefault("HF_HUB_OFFLINE", "1")
    e["LAM"] = "1e-3"


# ------------------------------------------------------------------------------------------- published components
def _hs(*xs):
    return [tuple(int(v) for v in x.split(".")) for x in xs]


# Wang et al. 2022 (Fig. 2 / Sec. 3): heads and the position they act at. "end" = the prediction position.
# Hanna et al. 2023 (Sec. 3-4): attention heads moving YY to END, and MLPs 8-11 at END.
CLASSES = {
    "name_mover": dict(task="ioi", heads=_hs("9.6", "9.9", "10.0"), role="end"),
    "s_inhibition": dict(task="ioi", heads=_hs("7.3", "7.9", "8.6", "8.10"), role="end"),
    "induction": dict(task="ioi", heads=_hs("5.5", "5.8", "5.9", "6.9"), role="s2"),
    "duplicate_token": dict(task="ioi", heads=_hs("0.1", "0.10", "3.0"), role="s2"),
    "previous_token": dict(task="ioi", heads=_hs("2.2", "4.11"), role="s1_next"),
    "gt_heads": dict(task="greater_than", heads=_hs("5.1", "5.5", "6.1", "6.9", "7.10", "8.8", "8.11", "9.1"),
                     role="end"),
}
# MLP and position checks (shares of all attribution mass, any kind): (task, kind filter, layers, role)
POSITION_CHECKS = {
    "ioi_io_pos": dict(task="ioi", role="io"),
    "ioi_s2_pos": dict(task="ioi", role="s2"),
    "gt_yy_pos": dict(task="greater_than", role="y1_yy"),
    "gt_late_mlp_end": dict(task="greater_than", role="end", kind="mlp", layers=(8, 9, 10, 11)),
}

# Target pathways (selection rule in select_targets.py / README, stated before any circuit was fitted)
PATHWAYS = {
    "IOI-NM": dict(task="ioi", kind="attn", layers=(9, 10), role="end", cls="name_mover"),
    "IOI-SI": dict(task="ioi", kind="attn", layers=(7, 8), role="end", cls="s_inhibition"),
    "IOI-IND": dict(task="ioi", kind="attn", layers=(5, 6), role="s2", cls="induction"),
    "GT-MLP": dict(task="greater_than", kind="mlp", layers=(8, 9, 10), role="end", cls=None),
}


# ------------------------------------------------------------------------------------------- tasks
def load_task(name):
    return json.load(open(TASKS / ("%s.json" % name), encoding="utf-8"))


def roles_of(row):
    """Per-token roles with the last token (the prediction position) labelled "end"."""
    r = list(row["roles"])
    r[-1] = "end"
    return r


def role_pos(row, role):
    r = roles_of(row)
    return r.index(role) if role in r else None


def tokens_of(row, io_token=None):
    """The row's prompt tokens; with io_token (IOI-NM per-name targets) the IO name token replaced by io_token. Returns
    None when the row cannot take it (its S name is io_token)."""
    t = list(row["tokens"])
    if io_token is None:
        return t
    r = row["roles"]
    if "io" not in r or int(row["bad"][0]) == int(io_token):
        return None
    t[r.index("io")] = int(io_token)
    return t


def task_items(task, split, role, io_token=None):
    """(item ids, token lists) of a split's rows that have the role (and can take io_token)."""
    ids, toks = [], []
    for i in task["split"][split]:
        r = task["rows"][i]
        if role_pos(r, role) is None:
            continue
        t = tokens_of(r, io_token)
        if t is None:
            continue
        ids.append(i); toks.append(t)
    return ids, toks


def pad_batch(rows_tokens, dev, pad=EOS):
    import torch
    T = max(len(t) for t in rows_tokens)
    x = torch.full((len(rows_tokens), T), pad, dtype=torch.long)
    for j, t in enumerate(rows_tokens):
        x[j, :len(t)] = torch.tensor(t, dtype=torch.long)
    return x.to(dev)


def parse(key):
    l, k, i = key.split(".")
    return int(l), k, int(i)


# ------------------------------------------------------------------------------------------- forward helpers
def site_acts(inf, bank, tokens, layer, kind, latent, bs=32):
    """The target's post-Top-K activation at every position, [n, T] (CPU)."""
    import torch
    from sae.dense import target_latent_activations
    kinds = list(bank.kinds)
    out = []

    def hook(li, acts):
        if li == layer:
            ta, ti = bank.encode(acts[kinds.index(kind)], kind, li)
            out.append(target_latent_activations(ta, ti, latent).float().cpu())
    inf.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), bs):
                inf.forward(tokens[s0:s0 + bs], activations_callback=hook, return_activations=False,
                            tokenize_final=False)
    finally:
        inf.enable_compile()
    return torch.cat(out)


def c_proj(inf, layer):
    return inf.model.hf.transformer.h[layer].attn.c_proj


def filler(loader, n, seed, n_corpus):
    """n corpus windows (64 tokens, production corpus), seeded: the text appended after each task prompt (after an EOS
    document separator) so a record's 64-token rows are OpenWebText-like, as the production records are."""
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    ids = sorted(set(int(x) for x in rng.integers(1, n_corpus + 1, size=int(n * 1.2) + 8)))
    rows = []
    for _, tk in loader.get_batches_by_ids(ids, max_length=65):
        rows.append(tk[:, :64].cpu())
    t = torch.cat(rows)
    perm = torch.from_numpy(rng.permutation(int(t.shape[0])))
    return t[perm][:n]


def windows(prompts, fill):
    """[n, 64] rows: prompt tokens, EOS, then filler text (truncated to 64)."""
    import torch
    out = torch.empty((len(prompts), WIN), dtype=torch.long)
    for j, p in enumerate(prompts):
        seq = list(p) + [EOS] + fill[j % fill.shape[0]].tolist()
        out[j] = torch.tensor(seq[:WIN], dtype=torch.long)
    return out


# ------------------------------------------------------------------------------------------- task record
def task_record(inf, bank, loader, n_corpus, key, task, role, seed=20261115, n_scan=1600, n_held_scan=1600,
                io_token=None):
    """A context record in protocol_contexts' format from task prompts:
      strong  48 strongest firing training-split prompts (train) + up to 64 strongest firing held-split prompts (held)
      mid     16 random firing training-split prompts not in strong (train = B_mid) + up to 64 random firing held-split
              prompts not in strong-held (held)
      D_train the production's stratified 32 of strongest-train (build_record's rule); training arm B = 32 + 16
      neg     64 verified-silent contrast rows: the task's own counterfactual prompts (IOI ABC / greater-than XX01) of
              training-split items, with filler, silent at every position; train first, held last
    Anchors are the role position (END or S2), not the window argmax. Held-split items use names / nouns disjoint from
    the training split (081 make_tasks)."""
    import torch
    from circuit.context_split import stratified_split
    from circuit.protocol_contexts import strong_holdout_frac
    from config import config
    cfg = config.discovery.context_v1
    layer, kind, latent = parse(key)
    rows = task["rows"]
    tr_items, tr_tok = task_items(task, "train", role, io_token)
    ho_items, ho_tok = task_items(task, "held", role, io_token)
    tr_items, ho_items = tr_items[:n_scan], ho_items[:n_held_scan]
    toks = dict(zip(tr_items, tr_tok)); toks.update(zip(ho_items, ho_tok))
    rng = torch.Generator().manual_seed(seed)

    def pool(items, fill_seed):
        prompts = [toks[i] for i in items]
        anc = torch.tensor([role_pos(rows[i], role) for i in items], dtype=torch.long)
        fill = filler(loader, len(items), fill_seed, n_corpus)
        pos = windows(prompts, fill)
        a = site_acts(inf, bank, pos.to(inf.device), layer, kind, latent)
        v = a[torch.arange(len(items)), anc]
        return pos, anc, v

    tp, ta, tv = pool(tr_items, seed + 1)
    hp, ha, hv = pool(ho_items, seed + 2)
    t_fire = [j for j in torch.argsort(-tv).tolist() if tv[j] > 0]
    h_fire = [j for j in torch.argsort(-hv).tolist() if hv[j] > 0]
    s_tr, s_ho = t_fire[:48], h_fire[:64]
    rest_t = [j for j in t_fire[48:]]
    rest_h = [j for j in h_fire[64:]]
    m_tr = [rest_t[j] for j in torch.randperm(len(rest_t), generator=rng).tolist()[:16]]
    m_ho = [rest_h[j] for j in torch.randperm(len(rest_h), generator=rng).tolist()[:64]]

    def mk(t_idx, h_idx):
        pos = torch.cat([tp[t_idx], hp[h_idx]])
        tgt = torch.cat([pos[:, 1:], torch.full((pos.shape[0], 1), EOS, dtype=torch.long)], 1)
        arg = torch.cat([ta[t_idx], ha[h_idx]])
        peak = torch.cat([tv[t_idx], hv[h_idx]])
        ids = [int(tr_items[j]) for j in t_idx] + [int(ho_items[j]) for j in h_idx]
        n_t = len(t_idx)
        return dict(pos=pos, tgt=tgt, arg=arg, peak=peak, train=list(range(n_t)),
                    held=list(range(n_t, n_t + len(h_idx))), ids=ids)

    rec = dict(n_top=len(t_fire), n_mid=len(rest_t), source="task:%s" % task["name"], role=role, io_token=io_token,
               fire_rate_train=float((tv > 0).float().mean()), fire_rate_held=float((hv > 0).float().mean()))
    if len(s_tr) < 8 or len(s_ho) < 4:
        rec["strong"] = None
        return rec
    rec["strong"] = mk(s_tr, s_ho)
    rec["mid"] = mk(m_tr, m_ho) if len(m_tr) >= 8 and len(m_ho) >= 4 else None
    s = rec["strong"]
    hf = strong_holdout_frac(int(cfg.pool_size), cfg.holdout_frac, cfg.train_strong)
    d_keep, _ = stratified_split(s["peak"][s["train"]], holdout_frac=hf, keep_top=cfg.keep_top)
    rec["D_train"] = [s["train"][j] for j in d_keep]
    if rec["mid"] is not None:
        rec["B_mid"] = list(rec["mid"]["train"])
    # contrast: the item's own counterfactual prompt (+ filler), verified silent at every position
    cf_items = tr_items
    cf_prompts = [rows[i]["cf_tokens"] for i in cf_items]
    cfill = filler(loader, len(cf_items), seed + 3, n_corpus)
    cpos = windows(cf_prompts, cfill)
    ca = site_acts(inf, bank, cpos.to(inf.device), layer, kind, latent)
    silent = [j for j in range(len(cf_items)) if float(ca[j].max()) <= 0.0]
    rec["contrast_silent_share"] = len(silent) / max(1, len(cf_items))
    rec["contrast_source"] = "task_cf"
    if len(silent) < 64:
        # top up with corpus windows (the filler pool), verified silent
        extra = filler(loader, 512, seed + 4, n_corpus)
        ea = site_acts(inf, bank, extra.to(inf.device), layer, kind, latent)
        es = [j for j in range(extra.shape[0]) if float(ea[j].max()) <= 0.0][:64 - len(silent)]
        neg = torch.cat([cpos[silent], extra[es]]) if es else cpos[silent]
        rec["contrast_source"] = "task_cf+corpus(%d)" % len(es)
    else:
        sel = [silent[j] for j in torch.randperm(len(silent), generator=rng).tolist()[:64]]
        neg = cpos[sel]
    if neg.shape[0] < int(cfg.min_contrast):
        rec["strong"] = None
        rec["no_contrast"] = True
        return rec
    rec["neg"] = neg[:64]
    rec["neg_ids"] = None
    return rec


def fold(rec, pool, k, size=16):
    """A shallow copy of rec whose `pool` held-out set is the k-th block of `size` rows (the production scorer uses the
    first 16 held-out rows; folds 1-3 extend the task held-out set to 64 prompts)."""
    r = dict(rec)
    p = dict(rec[pool])
    h = list(p["held"])[k * size:(k + 1) * size]
    if not h:
        return None
    p["held"] = h
    r[pool] = p
    return r


# ------------------------------------------------------------------------------------------- attribution
class _Stop(Exception):
    pass


def attribution(inf, bank, key, tokens, anchors, roles, members=None, n_heads=N_HEADS):
    """Attribution of the target's pre-activation at `anchors` to every Top-K slot at every in-scope (upstream) site:
    d pre / d gain at gain 1 (= the slot's first-order contribution), in the frame given by `members`:
      members = {site: {latent: alpha}}: the circuit's Z frame (non-members at every in-scope site set to 0, members
                at alpha x their live value, SAE error preserved; the production Z ablation)
      members = None: the full model (every latent live at alpha 1; the forward is the unedited model)
    Attention-out slots are split over heads by |exact head drive| (081 engine.head_drive) at that position.
    Returns (mass {(layer, kind, head, role): sum |attr|}, signed {(layer, kind, role): sum attr}, pre [B], natural
    flag-free target post-Top-K activation [B])."""
    import torch
    from eval.ablation_faithfulness import upstream_sites
    from model.hooks import multi_patch
    from sae.dense import sparse_topk_to_dense, target_latent_activations
    from head_drive_lib import head_drive_slots
    layer, kind, latent = parse(key)
    scope = upstream_sites(bank, layer, kind)
    D = int(bank.d_sae)
    sae_t = bank.saes[kind][layer]
    w_t = sae_t.encoder.weight[latent].detach()
    b_t = sae_t._get_bias_eff()[latent].detach()
    B, T = int(tokens.shape[0]), int(tokens.shape[1])
    dev = tokens.device
    mem = None
    if members is not None:
        mem = {}
        for s, d in members.items():
            if s in scope and d:
                v = torch.zeros(D, device=dev)
                m = torch.zeros(D, dtype=torch.bool, device=dev)
                idx = torch.tensor(sorted(d), device=dev, dtype=torch.long)
                v[idx] = torch.tensor([float(d[i]) for i in sorted(d)], device=dev)
                m[idx] = True
                mem[s] = (m, v)
    gains, slots, out = {}, {}, {}
    zin = {}

    def transform(li, kd, x):
        s = (li, kd)
        if s == (layer, kind):
            pre = x @ w_t.to(x.dtype) + b_t.to(x.dtype)
            rr = torch.arange(B, device=x.device)
            out["pre"] = pre[rr, anchors]
            with torch.no_grad():
                ta, ti = bank.encode(x.detach(), kd, li)
                out["tk"] = target_latent_activations(ta, ti, latent)[rr, anchors].detach()
            raise _Stop()
        if s not in scope:
            return x
        ta, ti = bank.encode(x, kd, li)
        dense_nat = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        error = x - bank.decode(dense_nat, kd, li)
        if mem is None:
            coef = torch.ones_like(ta)
        elif s in mem:
            m, v = mem[s]
            coef = (m[ti].to(ta.dtype) * v[ti].to(ta.dtype))
        else:
            coef = torch.zeros_like(ta)
        g = torch.ones_like(ta, requires_grad=True)
        gains[s] = g
        slots[s] = (ta.detach(), ti.detach(), coef.detach())
        vals = ta * coef * g
        dense = torch.zeros(B, T, D, device=x.device, dtype=x.dtype).scatter(-1, ti.long(), vals.to(x.dtype))
        return bank.decode(dense, kd, li) + error

    hs = []
    attn_layers = sorted(l for (l, k) in scope if k == "attn")
    for l in attn_layers:
        def pre_hook(_m, args, _l=l):
            zin[_l] = args[0].detach()
        hs.append(c_proj(inf, l).register_forward_pre_hook(pre_hook))

    class P:
        def __call__(self, model):
            return multi_patch(model, transform)

    inf.disable_compile()
    try:
        try:
            inf.forward(tokens, patcher=P(), grad_enabled=True, return_activations=False, tokenize_final=False)
        except _Stop:
            pass
    finally:
        for h in hs:
            h.remove()
        inf.enable_compile()
    pre = out["pre"]
    sl = sorted(gains)
    grads = torch.autograd.grad(pre.sum(), [gains[s] for s in sl], allow_unused=True)
    mass = defaultdict(float)
    signed = defaultdict(float)
    eps = float(getattr(bank, "norm_eps", 1e-5))
    rl = [list(r) + ["pad"] * (T - len(r)) for r in roles]
    role_ids = sorted(set(x for r in rl for x in r))
    rid = {r: j for j, r in enumerate(role_ids)}
    R = torch.tensor([[rid[x] for x in r] for r in rl], device=dev)            # [B, T]
    for s, g in zip(sl, grads):
        if g is None:
            continue
        ta, ti, coef = slots[s]
        a = g.detach().float()                                                  # [B, T, k]
        if s[1] == "attn" and s[0] in zin:
            cp = c_proj(inf, s[0])
            sae = bank.saes["attn"][s[0]]
            drv, _ = head_drive_slots(zin[s[0]], cp.weight.detach(), cp.bias.detach(), sae.encoder.weight.detach(), ti,
                                   n_heads, eps)                                # [B, T, H, k]
            w = drv.abs()
            w = w / w.sum(2, keepdim=True).clamp(min=1e-12)
            hm = (a.abs()[:, :, None, :] * w).sum(-1)                           # [B, T, H]
            for r, j in rid.items():
                sel = (R == j)
                if not bool(sel.any()):
                    continue
                v = hm[sel].sum(0)
                for h in range(n_heads):
                    mass[(s[0], s[1], h, r)] += float(v[h])
        else:
            pm = a.abs().sum(-1)
            for r, j in rid.items():
                sel = (R == j)
                if bool(sel.any()):
                    mass[(s[0], s[1], -1, r)] += float(pm[sel].sum())
        ps = a.sum(-1)
        for r, j in rid.items():
            sel = (R == j)
            if bool(sel.any()):
                signed[(s[0], s[1], r)] += float(ps[sel].sum())
    return dict(mass), dict(signed), pre.detach().float().cpu(), out["tk"].float().cpu()


def circuit_members(c):
    """{(layer, kind): {latent: alpha}} of a circuit object (seed node skipped)."""
    from eval.wcm_eval import circuit_alphas
    return circuit_alphas(c)
