"""Stage 10: per-target direct output effect (DAN-136) -> bundle/prediction.sqlite + target_dirs.npy.

Source: experiments/064-prediction-circuits/results (DAN-134), written only by that experiment's
target_effects.py (target_effects.parquet, target_topk.npz) and summarise.py (target_tiers.parquet). This stage only
reads them; nothing is recomputed. Per circuit target:

  direct logit effect  l = W_U[:32064] (g * d) / rms_typ, centred over the vocabulary (d = unit decoder column,
                       rms_typ = median final-residual rms); top/bottom 20 tokens, nats per unit activation
  z1, kurt, gain       top-1 z-score of l, excess kurtosis, unembedding gain vs random unit directions
  boost_peak           l_max x the target's peak activation: nats the top token gains over the average token at peak
  ctx_z                mean z-score of l over the latent's top-10 distinct logit_ctx next tokens
  tiers (TIER_RULES)   writer / pred_like / focused, from target_tiers.parquet as summarise.py wrote them

Also writes the target decoder directions for the inference DLA (DAN-135): target_dirs.npy, fp16 [n_targets, 1024]
(unit-norm decoder columns read from the fp32 SAE checkpoints), with target_dirs_gid.npy (int64, ascending) giving
each row's gid. Both plain .npy, so the back end can np.load(mmap_mode="r").

All three files are rebuilt on every run, written to temporary files and swapped in with atomic replaces.
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import time
from typing import Optional

import numpy as np

from export.explorer import ids
from export.explorer.bundle import Bundle

PREDICTION_FILE = "prediction.sqlite"
DIRS_FILE = "target_dirs.npy"
DIRS_GID_FILE = "target_dirs_gid.npy"
SRC_DIR = os.path.join("experiments", "064-prediction-circuits", "results")
SOURCES = dict(effects="target_effects.parquet", tiers="target_tiers.parquet", topk="target_topk.npz",
               rms_typ="rms_typ.json")
METHOD = "direct effect: W_U[:32064] (g * d) / rms_typ, centred (064)"
TIER_RULES = dict(
    writer="boost_peak >= 1 nat: at the target's peak activation its direct push on its top promoted token "
           "(vs the average token, typical final-norm scale) is at least 1 nat",
    pred_like="writer AND ctx_z >= 1: the latent's empirical next tokens (logit_ctx top-10) sit >= 1 sd above the "
              "average token in its direct logit vector",
    focused="pred_like AND z1 >= z95, the 95th percentile of z1 over 1,024 random live non-target latents at the "
            "same site",
    source="experiments/064-prediction-circuits/summarise.py tiers()",
)
# 064 README / summary.json; stage 6 checks them exactly (all targets, then passing circuits only)
EXPECTED_TIERS = dict(writer=2390, pred_like=1293, focused=140)
EXPECTED_TIERS_PASS = dict(writer=1246, pred_like=675, focused=74)
TIERS = ("writer", "pred_like", "focused")

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE target_prediction(
    gid INTEGER PRIMARY KEY, cid INTEGER, key TEXT, layer INTEGER, kind TEXT, latent INTEGER, pass INTEGER,
    boost_peak REAL, peak REAL, z1 REAL, z95 REAL, kurt REAL, gain REAL, lmax REAL, lmin REAL, mean_act REAL,
    ctx_z REAL, ctx_overlap20 REAL, ctx_hit1 INTEGER, has_ctx INTEGER,
    writer INTEGER, pred_like INTEGER, focused INTEGER,
    logit_up TEXT, logit_down TEXT
);
CREATE INDEX target_prediction_tier ON target_prediction(pred_like, focused);
CREATE INDEX target_prediction_boost ON target_prediction(boost_peak);
"""


def prediction_path(bundle: Bundle) -> str:
    return os.path.join(bundle.root, PREDICTION_FILE)


def _js(x) -> str:
    return json.dumps(x, separators=(",", ":"))


def _replace(tmp: str, path: str, tries: int = 8) -> None:
    """os.replace, retried: on Windows it fails while a reader has the old file open."""
    for t in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if t == tries - 1:
                raise
            time.sleep(0.25 * (t + 1))


def _num(x) -> Optional[float]:
    x = float(x)
    return None if np.isnan(x) else x


# ---------------------------------------------------------------------------------------------- coverage + checks
def coverage(bundle: Bundle) -> Optional[dict]:
    """Summary of prediction.sqlite for the manifest (None when it does not exist)."""
    path = prediction_path(bundle)
    if not os.path.isfile(path):
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    meta = {k: json.loads(v) for k, v in conn.execute("SELECT key, value FROM meta")}
    out = dict(file=PREDICTION_FILE, targets=q("SELECT COUNT(*) FROM target_prediction"),
               tiers={t: q(f"SELECT SUM({t}) FROM target_prediction") for t in TIERS},
               tiers_pass={t: q(f"SELECT SUM({t} * pass) FROM target_prediction") for t in TIERS},
               method=meta.get("method"), tier_rules=meta.get("tier_rules"), n_top=meta.get("n_top"),
               dirs=dict(file=DIRS_FILE, gid_file=DIRS_GID_FILE, dtype="float16", shape=meta.get("dirs_shape")))
    conn.close()
    return out


def validate(bundle: Bundle, main, ck, sample: int = 50) -> None:
    """Stage 6 and end-of-stage-10 checks. ck(name, ok, detail)."""
    path = prediction_path(bundle)
    if not os.path.isfile(path):
        return
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    q = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731
    circ = {g: (cid, p) for cid, g, p in main.execute(
        "SELECT c.cid, c.seed_gid, c.pass FROM circuit c JOIN target t ON t.gid = c.seed_gid AND t.run = c.run "
        "WHERE t.status = 'circuit'")}
    rows = {g: (cid, p) for g, cid, p in conn.execute("SELECT gid, cid, pass FROM target_prediction")}
    ck("prediction row per circuit target", rows.keys() == circ.keys(),
       f"{len(rows)} rows, {len(circ)} circuit targets, {len(circ.keys() - rows.keys())} missing, "
       f"{len(rows.keys() - circ.keys())} extra")
    bad = [g for g in rows if g in circ and (rows[g][0] != circ[g][0] or int(rows[g][1]) != int(circ[g][1] or 0))]
    ck("prediction cid and pass == bundle circuit (cid, pass)", not bad, bad[:5] or "ok")
    bad_k = [g for g, k in conn.execute("SELECT gid, key FROM target_prediction")
             if ids.gid_of(*ids.parse_key(k)) != g]
    ck("prediction key -> gid", not bad_k, bad_k[:5] or "ok")
    got = {t: q(f"SELECT SUM({t}) FROM target_prediction") for t in TIERS}
    ck("prediction tier counts (all targets)", got == EXPECTED_TIERS, got)
    got_p = {t: q(f"SELECT SUM({t} * pass) FROM target_prediction") for t in TIERS}
    ck("prediction tier counts (passing circuits)", got_p == EXPECTED_TIERS_PASS, got_p)
    ck("prediction tiers follow their rules",
       q("SELECT COUNT(*) FROM target_prediction WHERE writer != (boost_peak >= 1.0) "
         "OR pred_like != (writer AND COALESCE(ctx_z >= 1.0, 0)) OR focused != (pred_like AND z1 >= z95)") == 0)
    ck("prediction token lists well formed",
       q("SELECT COUNT(*) FROM target_prediction WHERE json_array_length(logit_up) != ? "
         "OR json_array_length(logit_down) != ?", *(2 * [json.loads(q("SELECT value FROM meta WHERE key = 'n_top'"))])) == 0)

    dp, gp = os.path.join(bundle.root, DIRS_FILE), os.path.join(bundle.root, DIRS_GID_FILE)
    if not (os.path.isfile(dp) and os.path.isfile(gp)):
        ck("target_dirs files present", False, [f for f in (DIRS_FILE, DIRS_GID_FILE)
                                                 if not os.path.isfile(os.path.join(bundle.root, f))])
    else:
        D = np.load(dp, mmap_mode="r")
        G = np.load(gp, mmap_mode="r")
        want = np.array(sorted(circ), np.int64)
        ck("target_dirs shape / dtype", D.dtype == np.float16 and D.shape == (len(want), 1024) and G.dtype == np.int64,
           f"{D.shape} {D.dtype}, gid {G.shape} {G.dtype}")
        ck("target_dirs gid order == sorted circuit targets", G.shape == want.shape and bool(np.array_equal(G, want)))
        nrm = np.linalg.norm(np.asarray(D, np.float32), axis=1)
        ck("target_dirs unit norm", bool(np.abs(nrm - 1).max() < 2e-3), f"norm in [{nrm.min():.4f}, {nrm.max():.4f}]")

    # promoted tokens vs stage 8's reading_latent.logit_up (same 064 maths, computed independently on the GPU)
    from export.explorer import stage_reading
    rp = stage_reading.reading_path(bundle)
    if os.path.isfile(rp):
        r = sqlite3.connect(f"file:{rp}?mode=ro&immutable=1", uri=True)
        have = [g for (g,) in r.execute("SELECT gid FROM reading_latent")]
        pick = random.Random(0).sample(sorted(set(have) & rows.keys()), min(sample, len(set(have) & rows.keys())))
        bad = []
        for g in pick:
            a = [t for t, _ in json.loads(r.execute("SELECT logit_up FROM reading_latent WHERE gid = ?", (g,)).fetchone()[0])][:6]
            b = [t for t, _ in json.loads(conn.execute("SELECT logit_up FROM target_prediction WHERE gid = ?", (g,)).fetchone()[0])][:6]
            if a != b:
                bad.append(g)
        r.close()
        ck(f"top-6 promoted tokens == reading_latent.logit_up ({len(pick)} targets)", not bad, bad[:5] or "ok")
    conn.close()


def _update_manifest(bundle: Bundle, cov: dict, info: dict) -> None:
    path = os.path.join(bundle.root, "manifest.json")
    if not os.path.isfile(path):
        print("  no manifest.json yet; stage 6 will record the prediction data", flush=True)
        return
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest.setdefault("features", {})["has_prediction"] = True
    manifest["prediction"] = cov
    manifest.setdefault("stages", {})["10"] = info
    for f in (PREDICTION_FILE, DIRS_FILE, DIRS_GID_FILE):
        manifest.setdefault("sizes", {})[f] = os.path.getsize(os.path.join(bundle.root, f))
        if manifest.get("checksums"):                   # stage 6 wrote checksums: keep these files' current
            from export.explorer.stage_finalize import _sha256
            manifest["checksums"][f] = _sha256(os.path.join(bundle.root, f))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, path)
    print("  manifest.json: features.has_prediction, prediction section updated", flush=True)


# ---------------------------------------------------------------------------------------------- decoder rows
def _target_dirs(gids: np.ndarray, sae_path: str) -> np.ndarray:
    """Decoder columns of the given gids (ascending) -> fp16 [n, 1024]; one checkpoint at a time, memory-mapped."""
    import torch
    out = np.empty((len(gids), 1024), np.float16)
    site = {}
    for j, g in enumerate(gids.tolist()):
        l, k, i = ids.split_gid(g)
        site.setdefault((l, k), []).append((j, i))
    for (l, k), items in sorted(site.items()):
        p = os.path.join(sae_path, f"sae-{k}", f"sae_{k}_layer_{l}.pth")
        sd = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        W = sd["decoder.weight"]                                    # [d_model, d_sae]
        assert W.shape == (1024, ids.D_SAE), (p, tuple(W.shape))
        js = np.array([j for j, _ in items])
        cols = W[:, torch.tensor([i for _, i in items])].T.float()
        out[js] = cols.numpy().astype(np.float16)
        del sd, W, cols
    return out


def _sae_path(repo_root: str) -> str:
    try:
        from config import config
        return str(config.weights.sae_path)
    except Exception:  # noqa: BLE001
        return os.path.join(repo_root, "models", "TuringLLM", "SAE")


# ---------------------------------------------------------------------------------------------- build
def build(bundle: Bundle, repo_root: str, source_dir: str = "") -> dict:
    import pandas as pd
    t0 = time.time()
    src = source_dir or os.path.join(repo_root, SRC_DIR)
    E = pd.read_parquet(os.path.join(src, SOURCES["effects"]))
    T = pd.read_parquet(os.path.join(src, SOURCES["tiers"]))
    K = np.load(os.path.join(src, SOURCES["topk"]))
    with open(os.path.join(src, SOURCES["rms_typ"]), encoding="utf-8") as fh:
        rms = json.load(fh)
    n_top = int(K["top_ids"].shape[1])
    if not (E.gid.is_unique and len(E) == len(T) == len(K["gid"])
            and (T.gid.to_numpy() == E.gid.to_numpy()).all() and (K["gid"] == E.gid.to_numpy()).all()):
        raise RuntimeError("064 effects / tiers / topk are not aligned row for row by gid")
    E = E.merge(T, on="gid", how="left", validate="one_to_one")

    main = sqlite3.connect(f"file:{bundle.sqlite}?mode=ro", uri=True)
    circ = {g: (cid, key, p) for cid, g, key, p in main.execute(
        "SELECT c.cid, c.seed_gid, t.key, c.pass FROM circuit c JOIN target t ON t.gid = c.seed_gid AND t.run = c.run "
        "WHERE t.status = 'circuit'")}
    missing = sorted(set(circ) - set(E.gid.tolist()))
    extra = sorted(set(E.gid.tolist()) - set(circ))
    if missing or extra:
        raise RuntimeError(f"064 targets != bundle circuit targets: {len(missing)} missing {missing[:5]}, "
                           f"{len(extra)} extra {extra[:5]}")
    pass_mismatch = [int(g) for g, p, c in zip(E.gid, E["pass"], E.cid)
                     if bool(p) != bool(circ[int(g)][2]) or int(c) != circ[int(g)][0]]
    if pass_mismatch:
        raise RuntimeError(f"064 pass / cid disagree with the bundle for {len(pass_mismatch)} targets: "
                           f"{pass_mismatch[:10]}")

    rows = []
    for j, r in enumerate(E.itertuples(index=False)):
        g = int(r.gid)
        l, k, i = ids.split_gid(g)
        cid, key, p = circ[g]
        up = [[int(t), round(float(v), 4)] for t, v in zip(K["top_ids"][j], K["top_vals"][j])]
        dn = [[int(t), round(float(v), 4)] for t, v in zip(K["bot_ids"][j], K["bot_vals"][j])]
        rows.append((g, cid, key, l, k, i, int(bool(p)),
                     _num(r.boost_peak), _num(r.peak), _num(r.z1), _num(r.z95), _num(r.kurt), _num(r.gain),
                     _num(r.lmax), _num(r.lmin), _num(r.mean_act), _num(r.ctx_z), _num(r.ctx_overlap20),
                     int(bool(r.ctx_hit1)), int(bool(r.has_ctx)),
                     int(bool(r.writer)), int(bool(r.pred_like)), int(bool(r.focused)), _js(up), _js(dn)))
    print(f"  {src}: {len(rows)} targets, {n_top} promoted / suppressed tokens each", flush=True)

    # decoder directions, ascending gid
    t1 = time.time()
    gids = np.array(sorted(circ), np.int64)
    D = _target_dirs(gids, _sae_path(repo_root))
    nrm = np.linalg.norm(D.astype(np.float32), axis=1)
    print(f"  target_dirs {D.shape} fp16 ({time.time() - t1:.0f}s), norm in [{nrm.min():.4f}, {nrm.max():.4f}]",
          flush=True)
    for name, arr in ((DIRS_FILE, D), (DIRS_GID_FILE, gids)):
        path = os.path.join(bundle.root, name)
        tmp = path + ".tmp.npy"
        np.save(tmp, arr)
        _replace(tmp, path)

    path = prediction_path(bundle)
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    conn = sqlite3.connect(tmp)
    conn.executescript(SCHEMA)
    conn.executemany(f"INSERT INTO target_prediction VALUES ({','.join('?' * 25)})", rows)
    st = {f: os.stat(os.path.join(src, f)) for f in SOURCES.values()}
    meta = dict(
        source=SRC_DIR.replace(os.sep, "/"), files=SOURCES,
        source_mtime={f: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s.st_mtime)) for f, s in st.items()},
        method=METHOD, rms_typ=rms.get("rms_typ"), n_top=n_top,
        logit_lists="[[token_id, value], ...]: value = centred logit per unit activation (nats); logit_up most "
                    "promoted first, logit_down most suppressed first (same format as reading_latent)",
        tier_rules=TIER_RULES, n_targets=len(rows),
        tiers={t: int(E[t].sum()) for t in TIERS}, tiers_pass={t: int((E[t] & E["pass"]).sum()) for t in TIERS},
        dirs=dict(file=DIRS_FILE, gid_file=DIRS_GID_FILE, dtype="float16",
                  source="SAE checkpoint decoder.weight [d_model, d_sae] (fp32) columns, unit norm"),
        dirs_shape=list(D.shape), built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        note="direct path only (attn / mlp directions add into the residual at their layer; later blocks can "
             "transform them); tiers fixed before the 064 inference results were seen")
    conn.executemany("INSERT INTO meta VALUES (?, ?)", [(k, _js(v)) for k, v in meta.items()])
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    _replace(tmp, path)

    from export.explorer.stage_finalize import Checks
    ck = Checks()
    validate(bundle, main, ck)
    main.close()
    if ck.failures:
        raise RuntimeError(f"prediction validation failed ({len(ck.failures)}): " + "; ".join(ck.failures))
    cov = coverage(bundle)
    info = dict(source=meta["source"], targets=len(rows), tiers=meta["tiers"], tiers_pass=meta["tiers_pass"],
                dirs_shape=list(D.shape), secs=round(time.time() - t0, 1))
    _update_manifest(bundle, cov, info)
    return info
