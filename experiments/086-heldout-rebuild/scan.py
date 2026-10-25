"""TOP-N CORPUS SCAN (DAN-157): each target latent's N strongest corpus windows, for topping up the held-out set.

The stores hold only the 64 strongest per latent and the 15k run used all of them (085 README §3), so the rebuild needs
a deeper list. This repeats the first pass's top-context computation (pipeline/first_pass.py + store/context.py) for
the needed latents only, with a deeper buffer:
  forward  Inference.forward on 64-token windows (the loader's skip_first_token rows), SAEBank.encode per site: the
           first pass's model / SAE path and numerics (bf16 model, the configured encode / Top-K backends)
  score    the top store's ranking statistic, store/context.compute_seq_scores: the MEAN of the target's post-Top-K
           activation over the 64 positions (fp32 here; the store kept bf16). Windows with score 0 (never fire) are
           never kept, as in the store. Also recorded: the peak (max over positions) and its position
  corpus   every window whose content is new: one id per content hash (the first, by 085's corpus_hash.npy), so the
           ~6.6M windows written twice do not take two slots. Twins are still caught by the rule (exact match)
  targets  every 062 target (targets_full.txt; the sweep, nulls, alpha = 1, 073, 083, 084 targets are all subsets)

Sharded over corpus shards (SHARD=i/k takes shards i::k), resumable (an atomic checkpoint every CKPT_EVERY shards),
then MODE=merge takes the exact top N over the shard files (the top N of a union is the top N of the parts' top Ns).

  CUDA_VISIBLE_DEVICES=0 SHARD=0/1 PYTHONPATH=src python experiments/086-heldout-rebuild/scan.py
  MODE=merge PYTHONPATH=src python experiments/086-heldout-rebuild/scan.py
  MODE=check PYTHONPATH=src python experiments/086-heldout-rebuild/scan.py     (scan vs the top store, see check())
  env: MODE (scan | merge | check)  SHARD (0/1)  N_TOP (512)  BATCH (512)  CKPT_EVERY (20 shards)
       LIMIT_SHARDS (first N of this shard's corpus shards: smoke / benchmark)  TARGETS (062 targets_full.txt)
       OUT_SCAN (086/out/scan)  CORPUS_DIR (085/out: corpus_hash.npy)  COMPILE=1 (torch.compile blocks)
       PARTIAL=1 (merge: allow missing corpus shards; smoke only, recorded in the meta)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
REPO = HERE.parent.parent
MODE = os.environ.get("MODE", "scan")
SHARD_I, SHARD_K = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
N_TOP = int(os.environ.get("N_TOP", "512"))
BATCH = int(os.environ.get("BATCH", "512"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "20"))
LIMIT_SHARDS = int(os.environ["LIMIT_SHARDS"]) if os.environ.get("LIMIT_SHARDS") else None
TARGETS = Path(os.environ.get("TARGETS", str(REPO / "experiments" / "062-h100-protocol-v1" / "targets_full.txt")))
OUT_SCAN = Path(os.environ.get("OUT_SCAN", str(HERE / "out" / "scan")))
CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", str(REPO / "experiments" / "085-heldout-audit" / "out")))
COMPILE = os.environ.get("COMPILE") == "1"
PARTIAL = os.environ.get("PARTIAL") == "1"
SEQ = 64
KINDS = ["attn", "mlp", "resid"]
SCAN_VERSION = "086-scan/v1 (mean over 64 positions, canonical-by-hash corpus)"


def log(s):
    print("[%s scan %d/%d] %s" % (time.strftime("%H:%M:%S"), SHARD_I, SHARD_K, s), flush=True)


def read_targets(path=TARGETS):
    return sorted({t for t in Path(path).read_text().split() if t})


def comp_latents(keys):
    """{component index: sorted latent list} for keys L.kind.i (component = L * 3 + kind index, TuringLLM)."""
    out = {}
    for k in keys:
        l, kd, i = k.split(".")
        out.setdefault(int(l) * len(KINDS) + KINDS.index(kd), set()).add(int(i))
    return {c: sorted(v) for c, v in sorted(out.items())}


def canonical_mask(corpus_dir=CORPUS_DIR):
    """bool [n + 1]: id is the first id carrying its content hash (index 0, the empty slot, is False)."""
    h = np.load(corpus_dir / "corpus_hash.npy", mmap_mode="r")
    _, first = np.unique(np.asarray(h[1:]), return_index=True)
    m = np.zeros(len(h), dtype=bool)
    m[first + 1] = True
    return m


def ckpt_path(i=SHARD_I, k=SHARD_K):
    return OUT_SCAN / ("scan.shard%dof%d.pt" % (i, k))


def save_atomic(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp-%d" % os.getpid())
    torch.save(obj, tmp)
    os.replace(tmp, path)


def meta_of(keys, n_corpus_shards):
    import hashlib
    return dict(version=SCAN_VERSION, n_top=N_TOP, n_targets=len(keys), targets_sha1=hashlib.sha1(
        "\n".join(keys).encode()).hexdigest(), n_corpus_shards=n_corpus_shards, canonical="first id per content hash")


class TopN:
    """Per-component running top-N buffers on the GPU: mean, id, peak, peak position for each needed latent."""

    def __init__(self, comps, n_top, device, d_sae):
        self.comps, self.n, self.dev = comps, n_top, device
        self.lut, self.val, self.ids, self.peak, self.arg = {}, {}, {}, {}, {}
        for c, lats in comps.items():
            lut = torch.full((d_sae,), len(lats), dtype=torch.long, device=device)   # not needed -> dummy column
            lut[torch.tensor(lats, device=device)] = torch.arange(len(lats), device=device)
            self.lut[c] = lut
            self.val[c] = torch.full((len(lats), n_top), float("-inf"), device=device)
            self.ids[c] = torch.zeros((len(lats), n_top), dtype=torch.long, device=device)
            self.peak[c] = torch.zeros((len(lats), n_top), device=device)
            self.arg[c] = torch.zeros((len(lats), n_top), dtype=torch.int16, device=device)

    def update(self, c, ids, ta, ti):
        """ids [B] long, ta / ti [B, 64, k] (the SAE's Top-K values / indices at this site)."""
        B, T, k = ti.shape
        nc = len(self.comps[c])
        slot = self.lut[c][ti.reshape(B * T, k)]
        dense = torch.zeros(B * T, nc + 1, device=ta.device, dtype=torch.float32)
        dense.scatter_add_(1, slot, ta.reshape(B * T, k).float())
        dense = dense.view(B, T, nc + 1)[..., :nc]                   # [B, 64, nc]
        mean = dense.sum(1) / T                                       # compute_seq_scores: sum / seq_len
        pk, pa = dense.max(1)
        mean = torch.where(mean > 0, mean, torch.full_like(mean, float("-inf")))
        V = torch.cat([self.val[c], mean.T], 1)
        top = torch.topk(V, self.n, dim=1).indices
        I = torch.cat([self.ids[c], ids[None, :].expand(nc, B)], 1)
        P = torch.cat([self.peak[c], pk.T], 1)
        A = torch.cat([self.arg[c], pa.T.to(torch.int16)], 1)
        self.val[c], self.ids[c] = V.gather(1, top), I.gather(1, top)
        self.peak[c], self.arg[c] = P.gather(1, top), A.gather(1, top)

    def state(self):
        return {c: dict(latents=self.comps[c], mean=self.val[c].cpu(), ids=self.ids[c].cpu(), peak=self.peak[c].cpu(),
                        arg=self.arg[c].cpu()) for c in self.comps}

    def load(self, st):
        for c, d in st.items():
            assert d["latents"] == self.comps[c]
            self.val[c] = d["mean"].to(self.dev); self.ids[c] = d["ids"].to(self.dev)
            self.peak[c] = d["peak"].to(self.dev); self.arg[c] = d["arg"].to(self.dev)


def shard_windows(loader, shard_index, canon):
    """(ids [n] int64, tokens [n, 64] int64) of the canonical windows of one corpus shard (the loader's rows)."""
    start_id, _ = loader.shard_id_ranges[shard_index]
    if start_id == -1:
        return np.zeros(0, np.int64), np.zeros((0, SEQ), np.int64)
    a = np.load(os.path.join(loader.data_path, loader.shard_files[shard_index]), mmap_mode="r")
    index = loader._shard_indices[shard_index]
    lens = index[:, 1] - index[:, 0]
    if not (lens == SEQ).all():
        raise SystemExit("shard %d: windows of length %s (scan assumes %d)" % (shard_index, np.unique(lens), SEQ))
    ids = start_id + np.arange(len(index), dtype=np.int64)
    keep = canon[ids]
    rows = np.asarray(a[index[keep, 0][:, None] + np.arange(SEQ)[None, :]], dtype=np.int64)
    return ids[keep], rows


def scan():
    from config import config
    from data.loader import DataLoader
    from model.inference import Inference
    from sae.bank import SAEBank
    device = torch.device("cuda")
    keys = read_targets()
    comps = comp_latents(keys)
    loader = DataLoader(device=torch.device("cpu"))
    n_cs = len(loader.shard_files)
    meta = meta_of(keys, n_cs)
    canon = canonical_mask()
    if len(canon) != sum(loader._shard_sequence_counts) + 1:
        raise SystemExit("corpus index (%d ids) does not match the loader (%d sequences); rerun 085 corpus_index.py"
                         % (len(canon) - 1, sum(loader._shard_sequence_counts)))
    mine = list(range(n_cs))[SHARD_I::SHARD_K]
    inference = Inference(device=device, compile=COMPILE)
    bank = SAEBank(device=device, load_decoders=False)
    buf = TopN(comps, N_TOP, device, int(bank.d_sae))
    done, stats = [], dict(windows=0, seconds=0.0)
    p = ckpt_path()
    if p.exists():
        st = torch.load(p, weights_only=False)
        if st["meta"] != meta:
            raise SystemExit("checkpoint %s was made with other settings: %s vs %s" % (p, st["meta"], meta))
        buf.load(st["comps"]); done, stats = st["done"], st["stats"]
    todo = [s for s in mine if s not in set(done)]
    if LIMIT_SHARDS is not None:
        todo = todo[:max(0, LIMIT_SHARDS - len(done))]
    log("%d targets in %d components | N %d | batch %d | corpus shards: %d mine, %d done, %d to do | compile %s" % (
        len(keys), len(comps), N_TOP, BATCH, len(mine), len(done), len(todo), COMPILE))
    k2c = {(l, kd): l * len(KINDS) + j for l in range(int(bank.n_layer)) for j, kd in enumerate(KINDS)}

    def run_batch(ids, rows):
        tok = torch.from_numpy(rows).to(device, non_blocking=True)
        idt = torch.from_numpy(ids).to(device)

        def hook(layer_idx, activations):
            for j, kd in enumerate(KINDS):
                c = k2c[(layer_idx, kd)]
                if c in comps:
                    ta, ti = bank.encode(activations[j], kd, layer_idx)
                    buf.update(c, idt, ta, ti)
        with torch.no_grad():
            inference.forward(tok, activations_callback=hook, return_activations=False, tokenize_final=False)

    t_last, since = time.time(), 0
    for n_, s in enumerate(todo):
        t0 = time.time()
        ids, rows = shard_windows(loader, s, canon)
        for b0 in range(0, len(ids), BATCH):
            run_batch(ids[b0:b0 + BATCH], rows[b0:b0 + BATCH])
        torch.cuda.synchronize()
        dt = time.time() - t0
        stats["windows"] += int(len(ids)); stats["seconds"] += dt
        done.append(s); since += 1
        if since >= CKPT_EVERY or n_ == len(todo) - 1:
            save_atomic(dict(meta=meta, done=sorted(done), stats=stats, comps=buf.state()), p)
            since = 0
        if n_ % 5 == 0 or n_ == len(todo) - 1:
            rate = stats["windows"] / max(stats["seconds"], 1e-9)
            left = sum(1 for x in mine if x not in set(done))
            log("shard %d (%d windows, %.1fs) | %d/%d done | %.0f windows/s | ~%.1f h left on this process" % (
                s, len(ids), dt, len(done), len(mine), rate, left * (stats["windows"] / max(1, len(done))) / max(rate, 1e-9) / 3600))
    log("DONE %d shards, %d windows, %.0f windows/s" % (len(done), stats["windows"],
                                                        stats["windows"] / max(stats["seconds"], 1e-9)))


def merge():
    """Exact top N over every shard file -> OUT_SCAN/scan_top<N>.pt; refuses overlap, mixed settings or (unless
    PARTIAL=1) missing corpus shards."""
    files = sorted(OUT_SCAN.glob("scan.shard*of*.pt"))
    if not files:
        raise SystemExit("no shard files in %s" % OUT_SCAN)
    parts = [torch.load(f, weights_only=False) for f in files]
    meta = parts[0]["meta"]
    for f, p in zip(files, parts):
        if p["meta"] != meta:
            raise SystemExit("%s: settings differ from %s" % (f, files[0]))
    seen = {}
    for f, p in zip(files, parts):
        for s in p["done"]:
            if s in seen:
                raise SystemExit("corpus shard %d scanned twice (%s, %s)" % (s, seen[s], f))
            seen[s] = f
    missing = sorted(set(range(meta["n_corpus_shards"])) - set(seen))
    if missing and not PARTIAL:
        raise SystemExit("%d corpus shards not scanned (first %s); PARTIAL=1 to merge anyway" % (len(missing), missing[:5]))
    comps = {}
    for c in parts[0]["comps"]:
        cat = {k: torch.cat([p["comps"][c][k] for p in parts], 1) for k in ("mean", "ids", "peak", "arg")}
        # deterministic order: mean descending, then id ascending (empty slots, mean -inf, last)
        order = torch.stack([torch.from_numpy(np.lexsort((cat["ids"][r].numpy(), -cat["mean"][r].numpy())))
                             for r in range(cat["ids"].shape[0])])[:, :meta["n_top"]]
        comps[c] = dict(latents=parts[0]["comps"][c]["latents"], **{k: v.gather(1, order) for k, v in cat.items()})
    out = OUT_SCAN / ("scan_top%d.pt" % meta["n_top"])
    windows = sum(p["stats"]["windows"] for p in parts)
    save_atomic(dict(meta=dict(meta, partial=bool(missing), n_missing_shards=len(missing), files=[f.name for f in files],
                               windows_scanned=windows, scanned_shards=sorted(seen)), comps=comps), out)
    print("merged %d files (%d corpus shards, %d missing, %d windows) -> %s" % (len(files), len(seen), len(missing),
                                                                                windows, out))


def load_scan(path=None):
    """The merged scan, with a lookup: rows(key) -> dict(ids, mean, peak, arg) for one target, strongest first, only
    windows that fire (mean > 0)."""
    path = Path(path) if path else OUT_SCAN / ("scan_top%d.pt" % N_TOP)
    sc = torch.load(path, weights_only=False)
    where = {}
    for c, d in sc["comps"].items():
        l, j = divmod(int(c), len(KINDS))
        for r, lat in enumerate(d["latents"]):
            where["%d.%s.%d" % (l, KINDS[j], lat)] = (c, r)

    def rows(key):
        c, r = where[key]
        d = sc["comps"][c]
        ok = torch.isfinite(d["mean"][r]) & (d["mean"][r] > 0)
        return {k: d[k][r][ok] for k in ("ids", "mean", "peak", "arg")}
    sc["rows"], sc["where"] = rows, where
    return sc


def check():
    """The scan against the top store (outputs/top_ctx.pt): every stored top-64 id whose canonical twin lies in a
    scanned shard must be in the scan's list (a partial scan is still exact for the shards it covers: a window in the
    global top 64 is in the top 64 of any subset), and the scan's mean must match the store's bf16 value. Writes
    OUT_SCAN/check.json."""
    from data.loader import DataLoader
    sc = load_scan()
    loader = DataLoader(device=torch.device("cpu"))
    h = np.load(CORPUS_DIR / "corpus_hash.npy", mmap_mode="r")
    hv = np.asarray(h[1:])
    _, first, inv = np.unique(hv, return_index=True, return_inverse=True)
    canon_of = np.concatenate([[0], first[inv] + 1])                 # id -> the canonical id of its content
    shard_of_id = np.zeros(len(h), dtype=np.int32)
    for s, (a, b) in enumerate(loader.shard_id_ranges):
        if a > 0:
            shard_of_id[a:b + 1] = s
    scanned = set(sc["meta"]["scanned_shards"])                     # as merged (the shard files may have moved on)
    top = torch.load(REPO / "outputs" / "top_ctx.pt", map_location="cpu", weights_only=False)
    tid, tval = top["ctx_seq_idx"], top["ctx_seq_val"].float()
    n_in, n_found, rel, rank_hi = 0, 0, [], []
    per_key = []
    for key in sorted(sc["where"]):
        c, r = sc["where"][key]
        lat = sc["comps"][c]["latents"][r]
        rows = sc["rows"](key)
        pos = {int(i): j for j, i in enumerate(rows["ids"].tolist())}
        k_in = k_found = 0
        for sid, v in zip(tid[c, lat].tolist(), tval[c, lat].tolist()):
            if sid <= 0:
                continue
            cid = int(canon_of[sid])
            if int(shard_of_id[cid]) not in scanned:
                continue
            k_in += 1
            j = pos.get(cid)
            if j is not None:
                k_found += 1
                rel.append(float(rows["mean"][j]) / max(v, 1e-12) - 1)
                rank_hi.append(j)
        n_in += k_in; n_found += k_found
        per_key.append((key, k_in, k_found))
    rel = np.asarray(rel)
    meta = {k: v for k, v in sc["meta"].items() if k != "scanned_shards"}
    out = dict(scan=meta, n_scanned_shards=len(scanned), store_ids_in_scanned_shards=n_in,
               found_in_scan=n_found, recall=n_found / max(1, n_in),
               mean_rel_err=dict(median=float(np.median(np.abs(rel))) if len(rel) else None,
                                 p99=float(np.quantile(np.abs(rel), 0.99)) if len(rel) else None,
                                 max=float(np.abs(rel).max()) if len(rel) else None),
               scan_rank_of_store_rows_max=int(max(rank_hi)) if rank_hi else None,
               keys_with_missing=[k for k, a, b in per_key if b < a][:50])
    (OUT_SCAN / "check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("scan", "keys_with_missing")}, indent=1))
    print("keys with a store row missing from the scan: %d %s" % (len(out["keys_with_missing"]), out["keys_with_missing"][:10]))


if __name__ == "__main__":
    {"scan": scan, "merge": merge, "check": check}[MODE]()
