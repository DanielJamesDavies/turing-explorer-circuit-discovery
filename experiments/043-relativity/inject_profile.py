"""FORWARD INJECTION: activate a validated latent in neutral contexts
and read which downstream latents it causally drives. The forward
counterpart of our (backward) circuits; no logits anywhere.

Per seed latent A:
  * take NEUTRAL windows (random training sequences where A is silent),
  * add mag * W_dec[:, A] to the residual contribution at A's site at
    one injection position (mag = A's typical peak, from its own
    top_ctx stats), via the same encode-modify-decode delta the
    circuits use,
  * read EVERY latent at every LATER site at positions p_inj..p_inj+3,
    delta vs the un-injected run,
  * NULLS: k random directions of identical norm injected at the same
    site/position -- a latent only counts as driven if its delta beats
    the max null delta.

Output: the seed's association profile -- (site, latent, delta,
null_max) rows -- to inject_<comp>_<lat>.jsonl, top rows printed.

  COMP=29 LAT=3736 PYTHONPATH=src python .../inject_profile.py
  env: NWIN (default 64), NNULL (3), OFFS (4), MAGSCALE (1.0)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from hardware import detect_devices, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import component_idx, split_component_idx
from sae.bank import SAEBank

HERE = Path(__file__).parent
COMP = int(os.environ["COMP"])
LAT = int(os.environ["LAT"])
NWIN = int(os.environ.get("NWIN", 64))
NNULL = int(os.environ.get("NNULL", 3))
OFFS = int(os.environ.get("OFFS", 4))
MAGSCALE = float(os.environ.get("MAGSCALE", 1.0))
P_INJ = int(os.environ.get("P_INJ", 32))
BATCH = 16
SEQ = 64


def neutral_windows(n):
    shards = sorted(Path("data").glob("*.npy"))
    rng = np.random.default_rng(3)
    out = []
    for sp in rng.choice(shards, size=min(40, len(shards)), replace=False):
        sh = np.asarray(np.load(sp, mmap_mode="r"))
        sep = np.where(sh == -1)[0]
        st = np.concatenate([[0], sep + 1]) + 1
        en = np.concatenate([sep, [len(sh)]])
        keep = (en - st) == SEQ
        st, en = st[keep], en[keep]
        for a, b in zip(st[: n // 30 + 1], en[: n // 30 + 1]):
            out.append(sh[a:b].tolist())
        if len(out) >= n:
            break
    return out[:n]


class InjectPatcher:
    """Adds `vec` to the stream at `site` at position `pos` only."""

    def __init__(self, bank, site, vec, pos):
        self.bank, self.site, self.vec, self.pos = bank, site, vec, pos

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) != self.site:
            return x
        x = x.clone()
        x[:, self.pos, :] += self.vec.to(device=x.device, dtype=x.dtype)
        return x


def main():
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=True,
                   compile=should_compile())
    n_kinds = len(bank.kinds)
    n_comp = bank.n_layer * n_kinds
    layer, ki = split_component_idx(COMP, n_kinds)
    kind = bank.kinds[ki]
    site = (layer, kind)

    sae = bank.saes[kind][layer]
    dec = sae.decoder.weight[:, LAT].detach().float()      # [d_model]
    # typical peak from the latent's own stored top context values
    from pipeline.discovery_artifacts import load_discovery_artifacts
    RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 "
                    "Implementation/Runs/20260531-152059-37117a33/"
                    "20260531-152059-37117a33")
    load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
    # magnitude = the seed's measured natural peak (a_pos_ho from its
    # fitted row), NOT ctx_seq_val (which stores per-sequence means and
    # under-injects ~25x -- caught on the first run).
    mag = None
    for _f in ("rows.jsonl", "know_rows.jsonl"):
        _p = HERE / _f
        if not _p.exists():
            continue
        for _line in open(_p):
            _r = json.loads(_line)
            if (_r["comp_idx"], _r["latent"]) == (COMP, LAT)                     and "a_pos_ho" in _r:
                mag = float(_r["a_pos_ho"])
                break
        if mag is not None:
            break
    assert mag is not None, "no a_pos_ho row for %d/%d" % (COMP, LAT)
    mag *= MAGSCALE
    vec = dec * mag
    print("inject c%d/%d at %s pos %d | mag %.2f | ||vec|| %.3f"
          % (COMP, LAT, site, P_INJ, mag, float(vec.norm())), flush=True)

    wins = neutral_windows(NWIN)
    toks_all = torch.tensor([[max(t, 0) for t in w] for w in wins],
                            dtype=torch.long, device=device)

    down = [(l, k) for l in range(bank.n_layer) for k in bank.kinds
            if (l > layer or (l == layer and
                              bank.kinds.index(k) > ki))]
    down_set = set(down)

    def profile(inj_vec):
        acc = torch.zeros((n_comp, bank.d_sae), device=device)
        for s0 in range(0, len(wins), BATCH):
            toks = toks_all[s0:s0 + BATCH]

            def cb(layer_idx, activations):
                with torch.no_grad():
                    for ki_, kd in enumerate(bank.kinds):
                        if (layer_idx, kd) not in down_set:
                            continue
                        ta, ti = bank.encode(activations[ki_], kd, layer_idx)
                        c = component_idx(layer_idx, ki_, n_kinds)
                        for off in range(OFFS):
                            p = min(P_INJ + off, ta.shape[1] - 1)
                            v = ta[:, p, :].float().reshape(-1)
                            ii = ti[:, p, :].reshape(-1).long()
                            acc[c].index_add_(0, ii, v / OFFS)
            patcher = (InjectPatcher(bank, site, inj_vec, P_INJ)
                       if inj_vec is not None else None)
            inference.forward(toks, num_gen=1, tokenize_final=False,
                              activations_callback=cb, patcher=patcher,
                              return_activations=False)
        return acc / max(1, len(wins))

    base = profile(None)
    real = profile(vec)
    delta = real - base

    nulls = []
    g = torch.Generator().manual_seed(11)
    for j in range(NNULL):
        rnd = torch.randn(dec.shape[0], generator=g).float().to(vec.device)
        rnd = rnd / rnd.norm() * vec.norm()
        nulls.append(profile(rnd) - base)
    null_max = torch.stack(nulls).abs().amax(0)

    flat = delta.flatten()
    top = flat.abs().topk(300)
    out = HERE / ("inject_c%d_%d.jsonl" % (COMP, LAT))
    rows = []
    with out.open("w") as fh:
        for gi in top.indices.tolist():
            c, lat = gi // bank.d_sae, gi % bank.d_sae
            d = float(delta[c, lat])
            nm = float(null_max[c, lat])
            r = {"comp": c, "latent": lat, "delta": round(d, 4),
                 "null_max": round(nm, 4),
                 "beats_null": abs(d) > 2 * nm + 1e-3}
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
    sig = [r for r in rows if r["beats_null"]]
    print("\n%d of top-300 deltas beat 2x the null | top signal:" % len(sig))
    print("%-6s %-9s %-7s %9s %9s" % ("comp", "site", "lat", "delta", "nullmax"))
    for r in sig[:25]:
        l_, k_ = split_component_idx(r["comp"], n_kinds)
        print("%-6d L%-2d %-5s %-7d %9.3f %9.3f"
              % (r["comp"], l_, bank.kinds[k_], r["latent"], r["delta"],
                 r["null_max"]))
    print("\n-> %s" % out.name)


if __name__ == "__main__":
    main()
