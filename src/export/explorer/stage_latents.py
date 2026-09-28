"""Stage 1: dense per-latent arrays (top / mid contexts, co-activations) + the latent table.

Every array is [N_LATENTS, width], row = gid, so the backend reads a latent with
one memory-mapped row access.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from export.explorer import ids
from export.explorer.bundle import Bundle

TOPK = 128  # SAE k on every component


def _save(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def build(bundle: Bundle, outputs_dir: str) -> dict:
    t0 = time.time()
    N = ids.N_LATENTS
    info: dict = {"outputs": outputs_dir}

    # ---- contexts: ids stay 1-based sequence ids with 0 = empty slot
    for name in ("top", "mid"):
        d = torch.load(os.path.join(outputs_dir, f"{name}_ctx.pt"), map_location="cpu", weights_only=False)
        sid = d["ctx_seq_idx"].reshape(N, -1).numpy().astype(np.int32, copy=False)
        val = d["ctx_seq_val"].reshape(N, -1).float().numpy().astype(np.float16)
        _save(os.path.join(bundle.arrays, f"{name}_ids.npy"), sid)
        _save(os.path.join(bundle.arrays, f"{name}_val.npy"), val)
        info[f"{name}_width"] = int(sid.shape[1])
        print(f"  {name}_ctx -> arrays ({time.time() - t0:.0f}s)", flush=True)
        del d, sid, val

    # ---- latent stats (needed for dead rows in co-activations too)
    st = torch.load(os.path.join(outputs_dir, "latent_stats.pt"), map_location="cpu", weights_only=False)
    active = st["active_count"].reshape(N)
    dead = active == 0

    # ---- co-activations: sort each row by PMI (descending); dead rows -> gid -1, value 0
    co = torch.load(os.path.join(outputs_dir, "top_coactivation.pt"), map_location="cpu", weights_only=False)
    cid = co["top_indices"].reshape(N, -1)
    cval = co["top_values"].reshape(N, -1).float()
    del co
    cval, order = torch.sort(cval, dim=1, descending=True)
    cid = torch.gather(cid, 1, order)
    del order
    cid[dead] = -1
    cval[dead] = 0
    _save(os.path.join(bundle.arrays, "coact_ids.npy"), cid.numpy().astype(np.int32, copy=False))
    _save(os.path.join(bundle.arrays, "coact_val.npy"), cval.numpy().astype(np.float16))
    info["coact_width"] = int(cid.shape[1])
    info["coact_mode"] = "pmi"
    print(f"  coactivation -> arrays ({time.time() - t0:.0f}s)", flush=True)
    del cid, cval

    # ---- latent table
    per_comp_tokens = st["active_count"].sum(dim=1) // TOPK
    tokens_seen = int(per_comp_tokens.max())  # zero activations are not counted, so some comps undercount
    n = active.double()
    mean = st["mean"].reshape(N).double()
    m2 = st["m2"].reshape(N).double()
    std = torch.where(n > 1, (m2 / (n - 1).clamp(min=1)).sqrt(), torch.zeros_like(m2))
    freq = n / tokens_seen
    seq_count = st["seq_count"].reshape(N)
    mean_seq = st["mean_seq"].reshape(N).double()
    gids = np.arange(N, dtype=np.int64)
    comp, lat = np.divmod(gids, ids.D_SAE)
    layer, kind_idx = np.divmod(comp, ids.N_KINDS)
    kinds = np.asarray(ids.KINDS)[kind_idx]

    conn = bundle.connect()
    conn.execute("DELETE FROM latent")
    rows = zip(gids.tolist(), layer.tolist(), kinds.tolist(), lat.tolist(),
               active.tolist(), freq.tolist(), mean.tolist(), std.tolist(),
               seq_count.tolist(), mean_seq.tolist())
    conn.executemany("INSERT INTO latent(gid, layer, kind, idx, active_count, frequency, mean, std, seq_count, mean_seq)"
                     " VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    info.update(tokens_seen=tokens_seen, n_latents=N, n_dead=int(dead.sum()),
                stats_run_id=st.get("metadata", {}).get("run_id"), secs=round(time.time() - t0, 1))
    print(f"  latent table: {N} rows, {info['n_dead']} dead ({time.time() - t0:.0f}s)", flush=True)
    return info
