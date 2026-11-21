"""129 C (CPU only): exact per-head and per-error attribution for 117's targets, in the full model and in 117's circuits'
Z frames, plus forward-only partial error-zeroing.

What 117's structure check could not see: its head shares only credit attention-out SAE *member* slots (|attr| split
over heads by |head drive|), so a head whose output reaches the target through the live SAE error (or through a
non-attention route) scores zero. Here, for every head h at every upstream attention layer l and position, with
G = d pre / d (site output, SAE space) in the frame and oc_h = head h's exact additive share of the normalised attn-out
(head_drive_lib's identity, x_s = sum_h oc_h + const):

  T_h = G . oc_h                                  the head's first-order effect on the target in the frame
  D_h = sum_{i active} coef_i (w_i . oc_h)(G . d_i)  the part routed through the site's dictionary slots (members only in
                                                  a Z frame; every slot in the full model)
  E_h = G . oc_h - sum_{i active} (w_i . oc_h)(G . d_i)   the part routed through the site's live SAE error

(G . d_i is read off a zero-valued additive gain on each active slot, so non-members get it too.) The same pass gives,
per site kind, the slot mass sum |a_i coef_i G.d_i| vs the error mass |G . err| (signed and abs), and 117's own
|drive|-split head mass (to check this pipeline reproduces 117's numbers).

Forward-only: the full model's target pre-activation with the SAE error zeroed at chosen site kinds only
({attn}, {mlp}, {attn, mlp}, {resid_mid, resid}, all = 120's regime).

  CUDA_VISIBLE_DEVICES= ./.venv/bin/python -I experiments/129-known-circuits-diagnosis/c_heads.py
  env: N_PROMPTS (64), BS (16), KEYS (comma list), THREADS (16)
  -> out/c_heads.jsonl (one row per target x frame)
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E117 = ROOT / "experiments" / "117-known-circuits-gpt2"
sys.path.insert(0, str(E117))
import lib117 as L  # noqa: E402  (adds src to sys.path)
import torch  # noqa: E402

N_PROMPTS = int(os.environ.get("N_PROMPTS", "64"))
BS = int(os.environ.get("BS", "16"))
OUTP = Path(os.environ.get("OUTP", str(HERE / "out" / "c_heads.jsonl")))
ERR_SETS = {"attn": ("attn",), "mlp": ("mlp",), "attn+mlp": ("attn", "mlp"), "resid": ("resid_mid", "resid"),
            "all": ("attn", "resid_mid", "mlp", "resid")}


class _Stop(Exception):
    pass


def frame_pass(inf, bank, key, tokens, anchors, roles, members=None, err_zero=(), grad=True):
    from eval.ablation_faithfulness import upstream_sites
    from model.hooks import multi_patch
    from sae.dense import sparse_topk_to_dense
    from head_drive_lib import head_drive_slots
    layer, kind, latent = L.parse(key)
    scope = upstream_sites(bank, layer, kind)
    D = int(bank.d_sae)
    sae_t = bank.saes[kind][layer]
    w_t = sae_t.encoder.weight[latent].detach()
    b_t = sae_t._get_bias_eff()[latent].detach()
    B, T = int(tokens.shape[0]), int(tokens.shape[1])
    mem = None
    if members is not None:
        mem = {}
        for s, d in members.items():
            if s in scope and d:
                v = torch.zeros(D); m = torch.zeros(D, dtype=torch.bool)
                idx = torch.tensor(sorted(d), dtype=torch.long)
                v[idx] = torch.tensor([float(d[i]) for i in sorted(d)]); m[idx] = True
                mem[s] = (m, v)
    saved, out, zin = {}, {}, {}

    def transform(li, kd, x):
        s = (li, kd)
        if s == (layer, kind):
            rr = torch.arange(B)
            out["pre"] = (x @ w_t + b_t)[rr, anchors]
            raise _Stop()
        if s not in scope:
            return x
        ta, ti = bank.encode(x, kd, li)
        recon = bank.decode(sparse_topk_to_dense(ta, ti, D, dtype=x.dtype), kd, li)
        error = x - recon
        if mem is None:
            coef = torch.ones_like(ta)
        elif s in mem:
            m, v = mem[s]
            coef = m[ti].to(ta.dtype) * v[ti].to(ta.dtype)
        else:
            coef = torch.zeros_like(ta)
        u = torch.zeros_like(ta, requires_grad=grad)
        vals = ta * coef + u
        y = bank.decode(torch.zeros(B, T, D, dtype=x.dtype).scatter(-1, ti.long(), vals), kd, li)
        if kd not in err_zero:
            y = y + error
        if grad:
            y.retain_grad()
        saved[s] = dict(y=y, u=u, err=error.detach(), ta=ta.detach(), ti=ti.detach(), coef=coef.detach())
        return y

    hs = []
    for l in sorted(l for (l, k) in scope if k == "attn"):
        def pre_hook(_m, args, _l=l):
            zin[_l] = args[0].detach()
        hs.append(L.c_proj(inf, l).register_forward_pre_hook(pre_hook))

    class P:
        def __call__(self, model):
            return multi_patch(model, transform)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    try:
        with ctx:
            try:
                inf.forward(tokens, patcher=P(), grad_enabled=grad, return_activations=False, tokenize_final=False)
            except _Stop:
                pass
    finally:
        for h in hs:
            h.remove()
    pre = out["pre"]
    if not grad:
        return dict(pre=pre.detach().float())
    sl = sorted(saved)
    us = [saved[s]["u"] for s in sl]
    pre.sum().backward(inputs=us + [saved[s]["y"] for s in sl])
    rl = [list(r) + ["pad"] * (T - len(r)) for r in roles]
    role_ids = sorted(set(x for r in rl for x in r))
    R = [[x for x in r] for r in rl]
    import numpy as np
    Rn = np.array(R)
    heads = defaultdict(lambda: [0.0] * 6)   # (l, h, role) -> [T_abs, T_sg, D_abs, D_sg, E_abs, E_sg]
    old = defaultdict(float)                 # 117-style |drive| split of |slot attr|: (l, h, role)
    kinds = defaultdict(lambda: [0.0] * 4)   # (l, kind) -> [slot_abs, slot_sg, err_abs, err_sg]
    eps = float(getattr(bank, "norm_eps", 1e-5))
    for s in sl:
        sv = saved[s]
        G = sv["y"].grad
        if G is None:
            continue
        gd = sv["u"].grad.detach()                                  # [B, T, k] = G . d_i
        attr = sv["ta"] * sv["coef"] * gd                           # slot attribution (117's)
        ge = (G * sv["err"]).sum(-1) if s[1] not in err_zero else torch.zeros(B, T)
        kinds[s][0] += float(attr.abs().sum()); kinds[s][1] += float(attr.sum())
        kinds[s][2] += float(ge.abs().sum()); kinds[s][3] += float(ge.sum())
        if s[1] != "attn":
            continue
        cp = L.c_proj(inf, s[0]); sae = bank.saes["attn"][s[0]]
        z = zin[s[0]]
        H = L.N_HEADS
        drv, _ = head_drive_slots(z, cp.weight.detach(), cp.bias.detach(), sae.encoder.weight.detach(), sv["ti"], H, eps)
        zh = z.reshape(B, T, H, -1).float()
        o = torch.einsum("bthk,hkd->bthd", zh, cp.weight.detach().float().reshape(H, zh.shape[-1], -1))
        a = o.sum(2) + cp.bias.detach().float()
        sd = (a - a.mean(-1, keepdim=True)).std(-1, keepdim=True) + eps
        oc = (o - o.mean(-1, keepdim=True)) / sd[..., None]          # [B, T, H, d]
        Th = torch.einsum("bthd,btd->bth", oc, G.float())
        Dall = (drv * gd[:, :, None, :]).sum(-1)                    # through every active slot
        Dh = (drv * (sv["coef"] * gd)[:, :, None, :]).sum(-1)       # through member slots (coef)
        Eh = Th - Dall
        w = drv.abs(); w = w / w.sum(2, keepdim=True).clamp(min=1e-12)
        hm = (attr.abs()[:, :, None, :] * w).sum(-1)
        for r in role_ids:
            sel = torch.from_numpy(Rn == r)
            if not bool(sel.any()):
                continue
            for h in range(H):
                t_, d_, e_ = Th[..., h][sel], Dh[..., h][sel], Eh[..., h][sel]
                c = heads[(s[0], h, r)]
                c[0] += float(t_.abs().sum()); c[1] += float(t_.sum())
                c[2] += float(d_.abs().sum()); c[3] += float(d_.sum())
                c[4] += float(e_.abs().sum()); c[5] += float(e_.sum())
                old[(s[0], h, r)] += float(hm[..., h][sel].sum())
    return dict(pre=pre.detach().float(), heads=dict(heads), old=dict(old), kinds=dict(kinds))


def merge(acc, res):
    for name in ("heads", "kinds"):
        for k, v in res[name].items():
            a = acc[name].setdefault(k, [0.0] * len(v))
            for j in range(len(v)):
                a[j] += v[j]
    for k, v in res["old"].items():
        acc["old"][k] = acc["old"].get(k, 0.0) + v
    acc["pre"].append(res["pre"])


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    from config import config
    from model.inference import Inference
    from sae.public_bank import PublicSAEBank
    config.model.family = "gpt2-small"
    dev = torch.device("cpu")
    inf = Inference(dev, family="gpt2-small")
    inf.disable_compile(); inf.enable_compile = lambda: None
    bank = PublicSAEBank(device=dev, family="gpt2-small", preload=False)
    sel = json.loads((E117 / "out" / "targets.json").read_text())
    keys = [t for t in (E117 / "out" / "targets.txt").read_text().split() if t]
    if os.environ.get("KEYS"):
        keys = [k for k in os.environ["KEYS"].split(",") if k]
    done = set()
    if OUTP.exists():
        for line in open(OUTP):
            r = json.loads(line); done.add((r["seed"], r["frame"]))
    fh = open(OUTP, "a")
    tasks = {}
    for key in keys:
        pname = next(p for p, d in sel["pathways"].items() if any(c["key"] == key for c in d["chosen"]))
        pw = L.PATHWAYS[pname]
        if pw["task"] not in tasks:
            tasks[pw["task"]] = L.load_task(pw["task"])
        task = tasks[pw["task"]]
        ent = next(c for c in sel["pathways"][pname]["chosen"] if c["key"] == key)
        items, toks = L.task_items(task, "held", pw["role"], ent.get("io_token"))
        items, toks = items[:N_PROMPTS], toks[:N_PROMPTS]
        batches = []
        for s0 in range(0, len(items), BS):
            rb = [task["rows"][i] for i in items[s0:s0 + BS]]
            tok = L.pad_batch(toks[s0:s0 + BS], dev)
            anc = torch.tensor([L.role_pos(r, pw["role"]) for r in rb])
            batches.append((tok, anc, [L.roles_of(r) for r in rb]))
        frames = [("full", None)]
        for cname, sub in (("corpus", "main"), ("task", "task")):
            cp = E117 / "out" / "wcm" / sub / "circuits" / ("%s.pt" % key)
            if cp.exists():
                c = torch.load(cp, weights_only=False, map_location="cpu")
                if c is not None:
                    frames.append((cname, L.circuit_members(c)))
        t0 = time.time()
        for fname, mem in frames:
            if (key, fname) in done:
                continue
            acc = dict(heads={}, kinds={}, old={}, pre=[])
            for tok, anc, roles in batches:
                merge(acc, frame_pass(inf, bank, key, tok, anc, roles, members=mem))
            row = dict(seed=key, pathway=pname, frame=fname, n_prompts=len(items),
                       pre=float(torch.cat(acc["pre"]).mean()),
                       heads=[[l, h, r] + v for (l, h, r), v in acc["heads"].items()],
                       old=[[l, h, r, v] for (l, h, r), v in acc["old"].items()],
                       kinds=[[l, k] + v for (l, k), v in acc["kinds"].items()])
            if fname == "full":
                ez = {}
                for nm, ks in ERR_SETS.items():
                    ps = [frame_pass(inf, bank, key, tok, anc, roles, None, err_zero=ks, grad=False)["pre"]
                          for tok, anc, roles in batches]
                    ez[nm] = float(torch.cat(ps).mean())
                row["err_zero_pre"] = ez
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("[129 C] %s %s pre %.3f (%.0fs)" % (key, fname, row["pre"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
