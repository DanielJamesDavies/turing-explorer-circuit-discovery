"""117 stage 1 (GPU, ~2 min): choose the target latents by the pre-stated rule (README "Selection rule").

On the first 512 TRAINING-split items of each task (081 task files), at the pathway's position:
  IOI-NM   attn-out L9, L10 at END     class heads = name movers 9.6, 9.9, 10.0
  IOI-SI   attn-out L7, L8 at END      class heads = S-inhibition 7.3, 7.9, 8.6, 8.10
  IOI-IND  attn-out L5, L6 at S2       class heads = induction 5.5, 5.8, 5.9, 6.9
  eligible: fires (post-Top-K > 0) on >= 50% of the items AND its pre-activation's head drive (exact per-head
  decomposition) comes >= 50% from the class heads (mean over the items it fires on, |drive| shares over that layer's
  12 heads); ranked by fire rate x mean activation when firing; the top 3 per pathway. Fallback (recorded): if fewer
  than 3 are eligible, the head-share bar drops to 0.3.
  GT-MLP   mlp-out L8, L9, L10 at END on greater-than prompts: eligible if it fires on >= 50% of the items; ranked by
  |Spearman rho| between its activation (0 when silent) and YY; the top 2 per layer.
L11 MLP is left out: GPT-2's deepest targets do not fit the 16 GB card in the fit (107 arena A notes).

  PYTHONPATH=src python -X utf8 experiments/117-known-circuits-gpt2/select_targets.py   -> out/targets.json, targets.txt
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib117 as L  # noqa: E402
from head_drive_lib import head_drive_slots  # noqa: E402

N_ITEMS = 512
BS = 64


def select_nm_per_name(inf, bank, kinds, eps, eligible, layers=(9, 10)):
    """IOI-NM, per IO name: over ALL training-split items, grouped by the IO name token. A latent qualifies for name n
    if the group has >= 8 items, it fires at END on >= 50% of them and on <= 10% of the other names' items, its head
    drive at END (on the group items it fires on) comes >= 50% from the name movers (fallback 0.3), and it is production
    frame-eligible; ranked by within-group fire rate x mean activation; the top 3 with distinct names and latents."""
    task = L.load_task("ioi")
    items = list(task["split"]["train"])
    rows = [task["rows"][i] for i in items]
    names = [int(r["good"][0]) for r in rows]
    grp = defaultdict(list)
    for j, n in enumerate(names):
        grp[n].append(j)
    cls_heads = set(L.CLASSES["name_mover"]["heads"])
    dev = inf.device
    fire = defaultdict(lambda: defaultdict(int))        # (l, lat) -> name -> n_fire
    acts = defaultdict(lambda: defaultdict(float))
    shs = defaultdict(lambda: defaultdict(float))
    for s0 in range(0, len(rows), BS):
        rb = rows[s0:s0 + BS]
        tok = L.pad_batch([r["tokens"] for r in rb], dev)
        pb = torch.tensor([len(r["tokens"]) - 1 for r in rb], device=dev)
        cap, zin, hs = {}, {}, []
        for l in layers:
            hs.append(L.c_proj(inf, l).register_forward_pre_hook(lambda m, a, _l=l: zin.__setitem__(_l, a[0].detach())))

        def cb(li, a):
            if li in layers:
                cap[li] = a[kinds.index("attn")]
        inf.disable_compile()
        try:
            with torch.no_grad():
                inf.forward(tok, activations_callback=cb, return_activations=False, tokenize_final=False)
        finally:
            for h in hs:
                h.remove()
            inf.enable_compile()
        rr = torch.arange(len(rb), device=dev)
        for l in layers:
            with torch.no_grad():
                ta, ti = bank.encode(cap[l], "attn", l)
                ta, ti = ta[rr, pb], ti[rr, pb]
                cp = L.c_proj(inf, l)
                drv, _ = head_drive_slots(zin[l][rr, pb][:, None], cp.weight, cp.bias,
                                          bank.saes["attn"][l].encoder.weight, ti[:, None], L.N_HEADS, eps)
                w = drv[:, 0].abs()
                hmask = torch.tensor([(l, h) in cls_heads for h in range(L.N_HEADS)], device=dev)
                share = (w[:, hmask].sum(1) / w.sum(1).clamp(min=1e-12)).cpu()
            ta, ti = ta.cpu(), ti.cpu()
            for b in range(len(rb)):
                n = names[s0 + b]
                for q in range(ta.shape[1]):
                    a = float(ta[b, q])
                    if a <= 0:
                        continue
                    k = (l, int(ti[b, q]))
                    fire[k][n] += 1; acts[k][n] += a; shs[k][n] += float(share[b, q])
    N = len(rows)
    cands = []
    for k, fn in fire.items():
        tot = sum(fn.values())
        for n, nf in fn.items():
            g = len(grp[n])
            if g < 8 or nf / g < 0.5:
                continue
            other = (tot - nf) / max(1, N - g)
            if other > 0.1:
                continue
            ok, sc = eligible(k[0], "attn", k[1])
            if not ok:
                continue
            cands.append(dict(key="%d.attn.%d" % k, layer=k[0], latent=k[1], io_token=int(n),
                              io_name=task["rows"][items[grp[n][0]]]["meta"]["io"], group_items=g,
                              fire_rate=round(nf / g, 4), fire_rate_other=round(other, 4),
                              mean_act=round(acts[k][n] / nf, 4), class_head_share=round(shs[k][n] / nf, 4),
                              corpus_windows=sc))
    chosen, bar = [], 0.5
    for bar in (0.5, 0.3):
        el = sorted([c for c in cands if c["class_head_share"] >= bar], key=lambda c: -c["fire_rate"] * c["mean_act"])
        chosen, used_n, used_l = [], set(), set()
        for c in el:
            if c["io_token"] in used_n or c["key"] in used_l:
                continue
            chosen.append(c); used_n.add(c["io_token"]); used_l.add(c["key"])
            if len(chosen) == 3:
                break
        if len(chosen) == 3:
            break
    return dict(share_bar=bar, n_candidates=len(cands), chosen=chosen,
                runners_up=sorted(cands, key=lambda c: -c["class_head_share"])[:10])


def main():
    from config import config
    from model.inference import Inference
    from sae.public_bank import PublicSAEBank
    config.model.family = "gpt2-small"
    import os
    dev = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_float32_matmul_precision("high")
    inf = Inference(dev, family="gpt2-small")
    bank = PublicSAEBank(device=dev, family="gpt2-small", preload=False)
    kinds = list(bank.kinds)
    eps = float(bank.norm_eps)
    # production frame eligibility (082 make_targets_public, frame "fired"): fires in >= 200 of the 4M corpus windows
    # and in no more than 50% of them (hub cap); a hub latent has no verified-silent contrast context
    from pipeline.component_index import component_idx
    ls = torch.load(L.STORES_ROOT / "outputs_public" / "gpt2-small" / "latent_stats.pt", weights_only=False,
                    map_location="cpu")
    seq_count = ls["seq_count"]
    n_win = 4_000_000

    def eligible(layer, kind, lat):
        sc = int(seq_count[component_idx(layer, kinds.index(kind), len(kinds)), lat])
        return 200 <= sc <= 0.5 * n_win, sc
    out = dict(rule="see README Selection rule", n_items=N_ITEMS, pathways={})
    hubs = {}
    for pname, pw in L.PATHWAYS.items():
        task = L.load_task(pw["task"])
        items = list(task["split"]["train"])[:N_ITEMS]
        rows = [task["rows"][i] for i in items]
        pos = [L.role_pos(r, pw["role"]) for r in rows]
        keep = [j for j, p in enumerate(pos) if p is not None]
        rows = [rows[j] for j in keep]
        pos = torch.tensor([pos[j] for j in keep])
        n = len(rows)
        cls_heads = set(L.CLASSES[pw["cls"]]["heads"]) if pw["cls"] else set()
        stats = {l: defaultdict(lambda: [0, 0.0, 0.0]) for l in pw["layers"]}      # latent -> [n_fire, sum act, sum share]
        actmat = {l: defaultdict(lambda: np.zeros(n)) for l in pw["layers"]}
        for s0 in range(0, n, BS):
            rb = rows[s0:s0 + BS]
            tok = L.pad_batch([r["tokens"] for r in rb], dev)
            pb = pos[s0:s0 + BS].to(dev)
            cap, zin = {}, {}
            hs = []
            if pw["kind"] == "attn":
                for l in pw["layers"]:
                    hs.append(L.c_proj(inf, l).register_forward_pre_hook(
                        lambda m, a, _l=l: zin.__setitem__(_l, a[0].detach())))

            def cb(li, acts):
                if li in pw["layers"]:
                    cap[li] = acts[kinds.index(pw["kind"])]
            inf.disable_compile()
            try:
                with torch.no_grad():
                    inf.forward(tok, activations_callback=cb, return_activations=False, tokenize_final=False)
            finally:
                for h in hs:
                    h.remove()
                inf.enable_compile()
            rr = torch.arange(len(rb), device=dev)
            for l in pw["layers"]:
                ta, ti = bank.encode(cap[l], pw["kind"], l)
                ta, ti = ta[rr, pb], ti[rr, pb]                                       # [b, k]
                share = None
                if pw["kind"] == "attn":
                    cp = L.c_proj(inf, l)
                    z = zin[l][rr, pb][:, None]                                       # [b, 1, Dz]
                    drv, _ = head_drive_slots(z, cp.weight, cp.bias, bank.saes["attn"][l].encoder.weight,
                                              ti[:, None], L.N_HEADS, eps)
                    w = drv[:, 0].abs()                                               # [b, H, k]
                    hmask = torch.tensor([(l, h) in cls_heads for h in range(L.N_HEADS)], device=dev)
                    share = (w[:, hmask].sum(1) / w.sum(1).clamp(min=1e-12))         # [b, k]
                ta_c, ti_c = ta.cpu(), ti.cpu()
                sh_c = share.cpu() if share is not None else None
                for b in range(len(rb)):
                    for q in range(ta_c.shape[1]):
                        a = float(ta_c[b, q])
                        if a <= 0:
                            continue
                        lat = int(ti_c[b, q])
                        st = stats[l][lat]
                        st[0] += 1; st[1] += a
                        st[2] += float(sh_c[b, q]) if sh_c is not None else 0.0
                        actmat[l][lat][s0 + b] = a
        cands = []
        for l in pw["layers"]:
            for lat, (nf, sa, ss) in stats[l].items():
                fr = nf / n
                if fr < 0.5:
                    continue
                ok, sc = eligible(l, pw["kind"], lat)
                if not ok:
                    hubs[pname] = hubs.get(pname, 0) + 1
                    continue
                c = dict(key="%d.%s.%d" % (l, pw["kind"], lat), layer=l, latent=lat, fire_rate=round(fr, 4),
                         mean_act=round(sa / nf, 4), corpus_windows=sc)
                if pw["kind"] == "attn":
                    c["class_head_share"] = round(ss / nf, 4)
                else:
                    yy = np.array([r["meta"]["yy"] for r in rows], dtype=float)
                    from scipy.stats import spearmanr
                    rho = spearmanr(actmat[l][lat], yy).correlation
                    c["rho_yy"] = round(float(rho), 4) if rho == rho else 0.0
                cands.append(c)
        if pw["kind"] == "attn":
            chosen, bar = [], 0.5
            for bar in (0.5, 0.3):
                el = [c for c in cands if c["class_head_share"] >= bar]
                el.sort(key=lambda c: -c["fire_rate"] * c["mean_act"])
                chosen = el[:3]
                if len(chosen) == 3:
                    break
            out["pathways"][pname] = dict(share_bar=bar, n_fire50=len(cands), chosen=chosen,
                                          runners_up=[c for c in sorted(cands, key=lambda c: -c["class_head_share"])
                                                      [:10]])
        else:
            chosen = []
            for l in pw["layers"]:
                el = sorted([c for c in cands if c["layer"] == l], key=lambda c: -abs(c["rho_yy"]))
                chosen += el[:2]
            out["pathways"][pname] = dict(n_fire50=len(cands), chosen=chosen)
        out["pathways"][pname]["excluded_frame"] = hubs.get(pname, 0)
        print(pname, json.dumps(out["pathways"][pname]["chosen"]), flush=True)
    # ---- IOI-NM per-name amendment (README: added after the first selection pass returned no IOI-NM target, before
    # any circuit was fitted). Name-mover outputs are name-specific, so the 50%-of-all-items rule cannot select them.
    if not out["pathways"]["IOI-NM"]["chosen"]:
        out["pathways"]["IOI-NM"] = dict(per_name=True, **select_nm_per_name(inf, bank, kinds, eps, eligible))
        print("IOI-NM per-name", json.dumps(out["pathways"]["IOI-NM"]["chosen"]), flush=True)
    (L.OUT).mkdir(parents=True, exist_ok=True)
    (L.OUT / "targets.json").write_text(json.dumps(out, indent=1))
    keys = [c["key"] for p in out["pathways"].values() for c in p["chosen"]]
    (L.OUT / "targets.txt").write_text("\n".join(keys) + "\n")
    print("targets:", keys)


if __name__ == "__main__":
    main()
