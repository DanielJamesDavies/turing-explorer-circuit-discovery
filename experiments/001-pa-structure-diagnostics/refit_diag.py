"""Quick refit diagnostic: full 16-cluster match cosines + mass-weighted
stability, to disambiguate 'clusters dissolve' vs 'hard-labeling noise on a
stable soft structure'. 3 seeds (shallow/mid/deep)."""
import sys
from pathlib import Path
REPO = "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
sys.path.insert(0, REPO + "/src")
import torch
from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method
from analysis.circuits.gradient_method_neg_mode_grid_runner import _candidate_with_index
from circuit.instrument.ig_baseline import collect_natural_codes
from circuit.instrument.restoration import MaskedRestorationInstrument
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import upstream_sites
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RR = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
          "20260531-152059-37117a33/20260531-152059-37117a33")
load_discovery_artifacts(RR, candidates_path=RR / "candidates.pt")
dev = detect_devices()[0]
loader = DataLoader(device=dev, pin_memory=is_fast_memory())
inf = Inference(device=dev, compile=should_compile())
bank = SAEBank(devices=detect_devices(), load_decoders=is_fast_memory(), compile=should_compile())
pb = ProbeDatasetBuilder(inf, bank, loader)
avg = torch.zeros((bank.n_layer * len(bank.kinds), bank.d_sae), device=bank.device)
nk = len(bank.kinds)
cand = torch.load(RR / "candidates.pt", map_location="cpu", weights_only=False)
_apply_sweep_config(max_per_site=24)
config.discovery.probe_sequence_count = 64
config.discovery.eval_sequence_count = 64


def nmf(V, r, iters=150, s=0, eps=1e-10):
    g = torch.Generator().manual_seed(s)
    W = (torch.rand(V.shape[0], r, generator=g) * .1 + .01).to(V.device)
    H = (torch.rand(r, V.shape[1], generator=g) * .1 + .01).to(V.device)
    for _ in range(iters):
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        W *= (V @ H.T) / (W @ (H @ H.T) + eps)
    return W, H


def greedy(HA, HB):
    A = HA / HA.norm(dim=1, keepdim=True).clamp_min(1e-9)
    B = HB / HB.norm(dim=1, keepdim=True).clamp_min(1e-9)
    C = (A @ B.T).clone(); r = C.shape[0]; perm = [-1] * r; cs = [0.] * r
    for _ in range(r):
        idx = torch.argmax(C); i, j = int(idx // r), int(idx % r)
        perm[i] = j; cs[i] = float(C[i, j]); C[i, :] = -1; C[:, j] = -1
    return perm, cs


for target in [(8, 30122), (14, 16202), (23, 31114)]:
    idx = next(i for i, c in enumerate(cand)
               if int(c['comp_idx']) == target[0] and int(c['latent_idx']) == target[1])
    c = _candidate_with_index(cand[idx], idx)
    sc, sl = int(c['comp_idx']), int(c['latent_idx'])
    L, ki = split_component_idx(sc, nk); sk = bank.kinds[ki]
    m0 = _build_mode_method("counterfactual_gradient", "local", inf, bank, avg, pb)
    pd = m0.build_probe_dataset(sc, sl)
    sites = sorted(upstream_sites(bank, L, sk)); pt, pa = pd.pos_tokens[:64], pd.pos_argmax[:64]
    B, T = pt.shape; sidx = {s: i for i, s in enumerate(sites)}
    sae = bank.saes[sk][L]; ws = sae.encoder.weight[sl].detach(); bs = sae._get_bias_eff()[sl].detach()
    _, res = collect_natural_codes(inf, bank, pt, set(sites))
    f0 = {s: torch.zeros(bank.d_sae) for s in sites}
    m1 = {s: torch.ones(bank.d_sae, dtype=torch.bool) for s in sites}
    er, ec, ev = [], [], []
    for st in range(0, B, 8):
        tk = pt[st:st + 8]
        rc = {s: (r[st:st + 8] if r.dim() == 3 else r) for s, r in res.items()}
        ins = MaskedRestorationInstrument(bank, set(sites), rc, f0, m1, L, sk, ws, bs)
        inf.disable_compile()
        try:
            inf.forward(tk, patcher=ins, grad_enabled=True, return_activations=False, tokenize_final=False)
        finally:
            inf.enable_compile()
        pre = ins.seed_pre_act; Bc = tk.shape[0]
        pac = pa[st:st + Bc].to(pre.device).clamp(0, pre.shape[1] - 1)
        peak = pre[torch.arange(Bc, device=pre.device), pac]
        order = sorted(ins.leaves)
        gr = torch.autograd.grad(peak.mean(), [ins.leaves[s] for s in order], allow_unused=True)
        rb = (torch.arange(Bc, device=dev) + st).view(-1, 1) * T + torch.arange(T, device=dev).view(1, -1)
        for s, g in zip(order, gr):
            if g is None:
                continue
            a = (g.float() * ins.leaves[s].detach().float()).abs()
            k = min(128, a.shape[-1]); v, la = a.topk(k, dim=-1)
            co = sidx[s] * bank.d_sae + la; ro = rb.unsqueeze(-1).expand_as(co); nz = v > 0
            er.append(ro[nz]); ec.append(co[nz]); ev.append(v[nz])
    rf = torch.cat(er); cf = torch.cat(ec); vf = torch.cat(ev); ns = len(sites) * bank.d_sae
    ma_all = torch.zeros(ns, device=dev); ma_all.index_add_(0, cf, vf)
    mc = min(24000, int((ma_all > 0).sum())); tc = ma_all.topk(mc).indices
    cm = torch.full((ns,), -1, dtype=torch.long, device=dev); cm[tc] = torch.arange(mc, device=dev)
    kp = cm[cf] >= 0; rk, ck, vk = rf[kp], cm[cf[kp]], vf[kp]
    Vf = torch.zeros(B * T * mc, device=dev); Vf.index_add_(0, rk * mc + ck, vk); Vf = Vf.view(B * T, mc)
    rsq = torch.arange(B * T, device=dev) // T; hA = rsq % 2 == 0

    def nr(M):
        return M / M.sum(dim=1, keepdim=True).clamp_min(1e-12)
    VA, VB = Vf[hA], Vf[~hA]; lA, lB = VA.sum(1) > 0, VB.sum(1) > 0
    WA, HA = nmf(nr(VA[lA]), 16, s=0); WB, HB = nmf(nr(VB[lB]), 16, s=0)
    mA, mB = VA.sum(0), VB.sum(0)
    perm, cs = greedy(HA, HB); HBp = HB[perm]

    def mem(H):
        return H / H.sum(0, keepdim=True).clamp_min(1e-12)
    MA, MB = mem(HA), mem(HBp); both = (mA > 0) & (mB > 0)
    a, b = MA[:, both], MB[:, both]
    lc = (a * b).sum(0) / (a.norm(0) * b.norm(0)).clamp_min(1e-9)
    persist = (a.argmax(0) == b.argmax(0)).float()
    w = mA[both]; w = w / w.sum()
    thr = torch.quantile(mA[both], 0.75); hi = mA[both] >= thr
    print(f"seed {sc}/{sl} L{L} {sk}: all-16 match_cos {[round(x,2) for x in sorted(cs,reverse=True)]}")
    print(f"   latent-membership cosine: unweighted {float(lc.mean()):.3f} | "
          f"mass-weighted {float((lc*w).sum()):.3f} | top-25%-mass {float(lc[hi].mean()):.3f}")
    print(f"   top-cluster persistence : unweighted {float(persist.mean()):.3f} | "
          f"mass-weighted {float((persist*w).sum()):.3f} | top-25%-mass {float(persist[hi].mean()):.3f}")
