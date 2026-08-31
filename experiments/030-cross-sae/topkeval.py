"""The mechanism test for the architecture-dependent null.

CLAIM UNDER TEST: dense ReLU nulls pass reconstruction at mid n because
their fitted amplitudes exploit anchor-position support that a Top-K
code could never emit. If that is the mechanism, then projecting the
reconstructed code back onto per-position top-k AT EVALUATION TIME
should collapse the nulls while leaving the discovered circuits
standing — one knob, everything else identical.

This is the same on-manifold move as the paper's `respect_topk` fill,
now used as a null repair rather than a fill correction.

Design: the three mid-n seeds where nulls passed (L2/14800 n=5,933;
L2/28259 n=5,043; L4/26044 n=2,321). Per seed, re-fit the discovered
tri-amp circuit and the SAME three random null sets (identical RNG
seeding as crosssae.run), then score every set under:
    dense  — the original evaluation (all fills dense)
    topk<K> — reconstructed code projected to per-position top-K
              (K in K_LIST) before decoding
Held-out probes throughout, as in the replication.

  PYTHONPATH=. python topkeval.py
"""
import json
import math
import random
import time

import torch
import torch.nn.functional as F

import crosssae as C

K_LIST = [32, 64, 128]
SEEDS_OF_INTEREST = [(2, 14800), (2, 28259), (4, 26044)]
HERE = C.HERE
DEV = C.DEV


def topk_project(code, k):
    """Zero all but the k largest entries per position."""
    if code.shape[-1] <= k:
        return code
    vals, idx = code.topk(k, dim=-1)
    out = torch.zeros_like(code)
    return out.scatter(-1, idx, vals)


def main():
    data = torch.load(HERE / "scan.pt", weights_only=False)
    toks, seeds = data["tokens"], data["seeds"]
    fh = (HERE / "topk_rows.jsonl").open("a")
    done = set()
    if (HERE / "topk_rows.jsonl").exists():
        for line in (HERE / "topk_rows.jsonl").open():
            try:
                r = json.loads(line)
                done.add((r["layer"], r["latent"], r["set"], r["eval"]))
            except Exception:
                pass

    for key, S in seeds.items():
        L, sl = S["layer"], S["latent"]
        if (L, sl) not in SEEDS_OF_INTEREST:
            continue
        UP = C.upstream_sites(L)
        saes = {site: C.load_sae(C.sub_name(site)) for site in UP}
        seed_sae = C.load_sae(C.sub_name(("resid", L)))
        pos = toks[S["pos_windows"]]
        neg = toks[S["neg_windows"]]
        pos_tr, pos_ho = pos[:C.N_TRAIN], pos[C.N_TRAIN:]
        neg_tr, neg_ho = neg[:C.N_TRAIN], neg[C.N_TRAIN:]

        def natural(tokens):
            return C.forward(tokens, C.Hooks({}, ("resid", L), seed_sae,
                                             sl)).detach()

        nat_tr = natural(pos_tr)
        anchors_tr = nat_tr.argmax(dim=1)
        target_tr = nat_tr.gather(1, anchors_tr[:, None]).squeeze(1)
        nat_ho = natural(pos_ho)
        anchors_ho = nat_ho.argmax(dim=1)
        a_pos_ho = float(F.relu(nat_ho.gather(1, anchors_ho[:, None])).mean())

        def site_means(tokens):
            means = {}
            for site in UP:
                cap = {}
                def grab(mod, inp, out, _s=site):
                    x = out[0] if isinstance(out, tuple) else out
                    cap[_s] = C.encode(saes[_s], x).mean(dim=(0, 1))
                hd = C.site_module(site).register_forward_hook(grab)
                with torch.no_grad():
                    C.model(tokens.to(DEV), use_cache=False)
                hd.remove()
                means[site] = cap[site].detach()
            return means

        means_pos = site_means(pos_tr)
        means_neg = site_means(neg_tr)

        def eval_transform(site, members_alpha, floor, k=None):
            sae = saes[site]
            ma = members_alpha.get(site, {})
            idx = torch.tensor(sorted(ma), device=DEV, dtype=torch.long)
            av = torch.tensor([ma[int(i)] for i in idx.tolist()], device=DEV)
            fl = floor.get(site) if floor else None
            def fn(x):
                c = C.encode(sae, x)
                chat = (fl.expand_as(c).clone() if fl is not None
                        else torch.zeros_like(c))
                if len(idx):
                    chat[..., idx] = c[..., idx] * av
                if k is not None:
                    chat = topk_project(chat, k)
                return x + (chat - c) @ sae["dec_w"].T
            return fn

        def read(transforms, tokens, anchors):
            pre = C.forward(tokens, C.Hooks(transforms, ("resid", L),
                                            seed_sae, sl))
            B = pre.shape[0]
            v = pre[torch.arange(B, device=DEV),
                    anchors.to(DEV).clamp(0, pre.shape[1] - 1)]
            return float(F.relu(v).mean())

        # empty-circuit anchors per eval regime (the floor must be measured
        # under the SAME projection the circuit is scored with)
        anchors_by_k = {}
        for k in [None] + K_LIST:
            e0 = read({s: eval_transform(s, {}, {}, k) for s in UP},
                      pos_ho, anchors_ho)
            eM = read({s: eval_transform(s, {}, means_pos, k) for s in UP},
                      pos_ho, anchors_ho)
            anchors_by_k[k] = (e0, eM)
        print("[L%d %d] a_pos_ho %.3f | floors: %s" % (
            L, sl, a_pos_ho,
            " ".join("%s:(%.2f,%.2f)" % ("dense" if k is None else "k%d" % k,
                                          v[0], v[1])
                     for k, v in anchors_by_k.items())), flush=True)

        def fit(support=None):
            params = {}
            for site in UP:
                th = torch.full((C.D_SAE,), -40.0 if support is not None
                                else 2.0, device=DEV)
                if support is not None and site in support:
                    th[support[site]] = 40.0
                th.requires_grad_(support is None)
                ps = torch.full((C.D_SAE,), math.log(math.e - 1.0),
                                device=DEV, requires_grad=True)
                params[site] = (th, ps)
            opt = torch.optim.AdamW([p for pr in params.values() for p in pr
                                     if p.requires_grad], lr=C.LR,
                                    weight_decay=0.05)
            temp = [1.0]
            tnorm = float((target_tr ** 2).mean())
            lam = 0.0 if support is not None else 1e-3
            floors3 = [(None, 1.0), (means_neg, C.DUAL_W),
                       (means_pos, C.TRIPLE_W)]
            for step in range(C.STEPS):
                temp[0] = 1.0 * (0.05 ** (step / max(C.STEPS - 1, 1)))
                s0 = (step * 4) % C.N_TRAIN
                tk, an = pos_tr[s0:s0 + 4], anchors_tr[s0:s0 + 4]
                tg = target_tr[s0:s0 + 4]
                opt.zero_grad()
                for fl, w in floors3:
                    tr = {s: C.spec_transform_factory(saes, s, temp, params,
                                                      fl or {}) for s in UP}
                    pre = C.forward(tk, C.Hooks(tr, ("resid", L), seed_sae,
                                                sl), grad=True)
                    v = pre[torch.arange(pre.shape[0], device=DEV),
                            an.to(DEV)]
                    (w * ((v - tg.to(DEV)) ** 2).mean()
                     / max(tnorm, 1e-6)).backward()
                pen = 0.0
                for site in UP:
                    th, ps = params[site]
                    m = torch.sigmoid(th / temp[0])
                    if support is None:
                        pen = pen + lam * m.sum()
                    pen = pen + lam * ((1 - m)
                                       * (F.softplus(ps) - 1).abs()).sum()
                if isinstance(pen, torch.Tensor):
                    pen.backward()
                opt.step()
            out = {}
            with torch.no_grad():
                for site in UP:
                    th, ps = params[site]
                    keep = (torch.sigmoid(th / temp[0]) > 0.5).nonzero(
                        as_tuple=True)[0]
                    al = F.softplus(ps)[keep]
                    out[site] = {int(i): float(a) for i, a in
                                 zip(keep.tolist(), al.tolist())}
            return out

        def score(ma, tag):
            n = sum(len(d) for d in ma.values())
            for k in [None] + K_LIST:
                ev = "dense" if k is None else "topk%d" % k
                if (L, sl, tag, ev) in done:
                    continue
                e0, eM = anchors_by_k[k]
                aw0 = read({s: eval_transform(s, ma, {}, k) for s in UP},
                           pos_ho, anchors_ho)
                awM = read({s: eval_transform(s, ma, means_pos, k)
                            for s in UP}, pos_ho, anchors_ho)
                row = {"layer": L, "latent": sl, "set": tag, "eval": ev,
                       "n": n,
                       "ampF0": round((aw0 - e0) / (a_pos_ho - e0), 4)
                       if abs(a_pos_ho - e0) > 1e-9 else None,
                       "ampFM": round((awM - eM) / (a_pos_ho - eM), 4)
                       if abs(a_pos_ho - eM) > 1e-9 else None}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print("  %-10s %-8s n=%-6d ampF0=%-9s ampFM=%s"
                      % (tag, ev, n, row["ampF0"], row["ampFM"]), flush=True)

        t0 = time.time()
        ma = fit()
        n_ref = sum(len(d) for d in ma.values())
        score(ma, "discovered")
        print("  (discovered fit %.0fs, n=%d)" % (time.time() - t0, n_ref),
              flush=True)

        # identical null draws to crosssae.run: same pool construction,
        # same RNG seed
        live = []
        for site in UP:
            cap = {}
            def grab(mod, inp, out, _s=site):
                x = out[0] if isinstance(out, tuple) else out
                cap[_s] = (C.encode(saes[_s], x) > 0).reshape(
                    -1, C.D_SAE).any(0)
            hd = C.site_module(site).register_forward_hook(grab)
            with torch.no_grad():
                C.model(pos_tr.to(DEV), use_cache=False)
            hd.remove()
            live += [(site, int(i)) for i in
                     cap[site].nonzero(as_tuple=True)[0].tolist()]
        rng = random.Random(1000 + sl)
        for draw in range(C.N_NULL):
            members = rng.sample(live, min(n_ref, len(live)))
            support = {}
            for s, i in members:
                support.setdefault(s, []).append(i)
            support = {s: torch.tensor(v, dtype=torch.long, device=DEV)
                       for s, v in support.items()}
            score(fit(support=support), "null%d" % draw)
        del saes, seed_sae
        torch.cuda.empty_cache()
    fh.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
