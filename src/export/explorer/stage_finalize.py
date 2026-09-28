"""Stage 6: derived latent fields, indexes, manifest and validation.

Validation failures raise, so a bundle that does not hold together is never
marked finished.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from typing import Any, Dict, List

import numpy as np
import torch

from export.explorer import SCHEMA_VERSION, ids
from export.explorer.bundle import INDEXES, Bundle, clean, unpack
from export.explorer.corpus import Corpus
from export.explorer.run_source import RunSource


def _derive(conn) -> None:
    conn.executescript(INDEXES)
    conn.execute("UPDATE latent SET is_target = 0, target_status = NULL, target_reason = NULL, "
                 "seed_cid = NULL, n_memberships = 0")
    conn.execute("UPDATE latent SET is_target = 1, target_status = t.status, target_reason = t.reason "
                 "FROM target t WHERE t.gid = latent.gid")
    conn.execute("UPDATE latent SET seed_cid = c.cid FROM circuit c "
                 "WHERE c.seed_gid = latent.gid AND c.run = (SELECT run FROM target WHERE target.gid = latent.gid)")
    conn.execute("CREATE TEMP TABLE mc AS SELECT gid, COUNT(*) AS n FROM member GROUP BY gid")
    conn.execute("UPDATE latent SET n_memberships = mc.n FROM mc WHERE mc.gid = latent.gid")
    conn.commit()


def _git(repo: str) -> Dict[str, Any]:
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", "src"],
                                    capture_output=True, text=True).stdout.strip())
        return dict(commit=head, src_dirty=dirty)
    except OSError:
        return {}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class Checks:
    def __init__(self):
        self.failures: List[str] = []
        self.passed: List[str] = []

    def __call__(self, name: str, ok: bool, detail: Any = "") -> None:
        (self.passed if ok else self.failures).append(f"{name}: {detail}" if detail != "" else name)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)


def validate(bundle: Bundle, conn, outputs_dir: str, run: RunSource, run_name: str, data_dir: str) -> Checks:
    ck = Checks()
    q = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731
    n_lat = q("SELECT COUNT(*) FROM latent")
    ck("latent rows == N_LATENTS", n_lat == ids.N_LATENTS, n_lat)
    ck("member gids are latents", q("SELECT COUNT(*) FROM member WHERE gid < 0 OR gid >= ?", ids.N_LATENTS) == 0)
    ck("members reference circuits", q("SELECT COUNT(*) FROM member m LEFT JOIN circuit c USING(cid) WHERE c.cid IS NULL") == 0)
    ck("no circuit lists its seed as a member",
       q("SELECT COUNT(*) FROM member m JOIN circuit c USING(cid) WHERE m.gid = c.seed_gid") == 0)
    n_circ = q("SELECT COUNT(*) FROM circuit WHERE run = ?", run_name)
    ck("one circuit per non-skipped target", n_circ == q("SELECT COUNT(*) FROM target WHERE status = 'circuit'"), n_circ)
    ck("every circuit target has strong contexts",
       q("SELECT COUNT(*) FROM circuit c WHERE NOT EXISTS (SELECT 1 FROM target_ctx t WHERE t.gid = c.seed_gid AND t.pool = 'strong')") == 0)
    ck("contrast contexts only for targets",
       q("SELECT COUNT(*) FROM target_ctx t LEFT JOIN target g USING(gid) WHERE g.gid IS NULL") == 0)
    n_seq = np.load(os.path.join(bundle.tokens, "tokens.npy"), mmap_mode="r").shape[0]
    ck("every context seq_id inside the token array", q("SELECT COUNT(*) FROM target_ctx WHERE seq_id < 1 OR seq_id > ?", n_seq) == 0)
    ck("latent.seed_cid set for every circuit target",
       q("SELECT COUNT(*) FROM latent WHERE target_status = 'circuit' AND seed_cid IS NULL") == 0)
    ck("n_memberships sums to member rows", q("SELECT SUM(n_memberships) FROM latent") == q("SELECT COUNT(*) FROM member"))

    # graph structure
    bad_star, bad_wired, bad_members = [], [], []
    for cid, key, topo, seed, nn, blob in conn.execute("SELECT cid, key, topology, seed_gid, n_nodes, graph FROM circuit"):
        g = unpack(blob)
        gids = {n["gid"] for n in g["nodes"]}
        if len(g["nodes"]) != nn or g["seed"] != seed or seed not in gids:
            bad_members.append(key)
        att = [e for e in g["edges"] if e["type"] == "attribution"]
        if any(e["dst"] != seed or e["src"] not in gids for e in att):
            bad_star.append(key)
        if topo == "wired" and not any(e["type"] == "wired" for e in g["edges"]):
            bad_wired.append(key)
        if topo == "star" and len(att) != nn - 1:
            bad_star.append(key)
    ck("graph nodes consistent with circuit row", not bad_members, bad_members[:5] or "ok")
    ck("attribution edges point member -> seed", not bad_star, bad_star[:5] or "ok")
    ck("wired circuits have wired edges", not bad_wired, bad_wired[:5] or "ok")

    # round trip: arrays vs stores
    rng = random.Random(0)
    sample = rng.sample(range(ids.N_LATENTS), 200)
    for name in ("top", "mid"):
        d = torch.load(os.path.join(outputs_dir, f"{name}_ctx.pt"), map_location="cpu", weights_only=False)
        src_ids = d["ctx_seq_idx"].reshape(ids.N_LATENTS, -1)
        src_val = d["ctx_seq_val"].reshape(ids.N_LATENTS, -1).float()
        arr_ids = np.load(os.path.join(bundle.arrays, f"{name}_ids.npy"), mmap_mode="r")
        arr_val = np.load(os.path.join(bundle.arrays, f"{name}_val.npy"), mmap_mode="r")
        ok_ids = all(np.array_equal(arr_ids[g], src_ids[g].numpy()) for g in sample)
        ok_val = all(np.allclose(arr_val[g].astype(np.float32), src_val[g].numpy(), rtol=2e-3, atol=1e-3) for g in sample)
        ck(f"{name} arrays == store (200 latents)", ok_ids and ok_val, f"ids {ok_ids}, values {ok_val}")
        del d, src_ids, src_val
    co = torch.load(os.path.join(outputs_dir, "top_coactivation.pt"), map_location="cpu", weights_only=False)
    c_ids = co["top_indices"].reshape(ids.N_LATENTS, -1)
    c_val = co["top_values"].reshape(ids.N_LATENTS, -1).float()
    a_ids = np.load(os.path.join(bundle.arrays, "coact_ids.npy"), mmap_mode="r")
    a_val = np.load(os.path.join(bundle.arrays, "coact_val.npy"), mmap_mode="r")
    dead = {r[0] for r in conn.execute("SELECT gid FROM latent WHERE active_count = 0 AND gid IN (%s)"
                                       % ",".join(map(str, sample)))}
    bad = []
    for g in sample:
        if g in dead:
            ok = bool((a_ids[g] == -1).all())
        else:
            src = dict(zip(c_ids[g].tolist(), c_val[g].tolist()))
            got = dict(zip(a_ids[g].tolist(), a_val[g].astype(np.float32).tolist()))
            ok = src.keys() == got.keys() and all(abs(src[k] - got[k]) <= 1e-2 + 2e-3 * abs(src[k]) for k in src) \
                and bool((np.diff(a_val[g].astype(np.float32)) <= 1e-6).all())
        if not ok:
            bad.append(g)
    ck("coact arrays == store as sorted sets (200 latents)", not bad, bad[:5] or "ok")
    del co, c_ids, c_val

    # round trip: tokens
    corpus = Corpus(data_dir)
    tok = np.load(os.path.join(bundle.tokens, "tokens.npy"), mmap_mode="r")
    probe = [1, 8192, 8193, n_seq] + [rng.randint(1, n_seq) for _ in range(200)]
    ck("tokens[id-1] == corpus (204 ids)", all(np.array_equal(tok[s - 1], corpus.get(s)) for s in probe))

    # round trip: targets (contexts, members, headline metrics)
    keys = rng.sample(run.targets, min(50, len(run.targets)))
    bad_ctx, bad_mem, bad_head = [], [], []
    for key in keys:
        gid = ids.gid_of(*ids.parse_key(key))
        rec = run.ctx(key) or {}
        for pool, src in (("strong", (rec.get("strong") or {}).get("ids")), ("mid", (rec.get("mid") or {}).get("ids")),
                          ("neg", rec.get("neg_ids"))):
            got = [r[0] for r in conn.execute("SELECT seq_id FROM target_ctx WHERE gid = ? AND pool = ? ORDER BY rank", (gid, pool))]
            if [int(s) for s in (src or [])] != got:
                bad_ctx.append(f"{key}:{pool}")
            if src and pool != "neg":
                p = rec[pool]
                rows = conn.execute("SELECT seq_id, peak, arg FROM target_ctx WHERE gid = ? AND pool = ? ORDER BY rank",
                                    (gid, pool)).fetchall()
                if any(abs(r[1] - float(p["peak"][i])) > 1e-5 or r[2] != int(p["arg"][i]) for i, r in enumerate(rows)):
                    bad_ctx.append(f"{key}:{pool}:peak")
                if not np.array_equal(tok[np.asarray(got) - 1].astype(np.int64), p["pos"].numpy()):
                    bad_ctx.append(f"{key}:{pool}:tokens")
        circ = run.circuit(key)
        if circ is not None:
            want = {}
            for n in circ.nodes.values():
                if n.metadata.get("role") != "seed":
                    f = n.feature_id
                    want[ids.gid_of(f.layer, f.kind, f.index)] = (n.metadata["attribution_score"], n.metadata["amplitude"])
            got = {g: (a, al) for g, a, al in conn.execute(
                "SELECT m.gid, m.attribution, m.alpha FROM member m JOIN circuit c USING(cid) WHERE c.seed_gid = ? AND c.run = ?",
                (gid, run_name))}
            if want.keys() != got.keys() or any(abs(want[g][0] - got[g][0]) > 1e-6 or abs(want[g][1] - got[g][1]) > 1e-6 for g in want):
                bad_mem.append(key)
            head = run.headline.get(key)
            if head:
                row = conn.execute("SELECT free0_tk, freeM_topk_tk, freeN_topk_tk FROM circuit WHERE seed_gid = ? AND run = ?",
                                   (gid, run_name)).fetchone()
                want_h = [float(head[c]) if head[c] not in ("", "nan") else None for c in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")]
                if any((a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9) for a, b in zip(row, want_h)):
                    bad_head.append(key)
    ck(f"target contexts == run files ({len(keys)} targets: ids, peak, arg, tokens)", not bad_ctx, bad_ctx[:5] or "ok")
    ck(f"members == circuit files ({len(keys)} targets: gid, attribution, alpha)", not bad_mem, bad_mem[:5] or "ok")
    ck(f"headline metrics == targets.csv ({len(keys)} targets)", not bad_head, bad_head[:5] or "ok")

    # search index (stage 7), when built
    search = bundle.stage_info("7")
    if search:
        from export.explorer import stage_search
        files = stage_search.search_files(bundle)
        ck("search index files present", len(files) == 1 + len(stage_search.ARRAYS), files)
        if len(files) == 1 + len(stage_search.ARRAYS):
            import sqlite3
            s = sqlite3.connect(os.path.join(stage_search.search_dir(bundle), "search.sqlite"))
            k = json.loads(s.execute("SELECT value FROM meta WHERE key = 'k'").fetchone()[0])
            s.close()
            n_u = np.load(os.path.join(stage_search.search_dir(bundle), "seq_ids.npy"), mmap_mode="r").shape[0]
            ck("search index matches its stage marker", k == search["k"] and n_u == search["n_sequences"],
               f"k {k}, {n_u} sequences")
    return ck


def build(bundle: Bundle, bundle_id: str, outputs_dir: str, run: RunSource, run_name: str, data_dir: str,
          repo_root: str, stage_infos: Dict[str, Any], checksums: bool = True) -> dict:
    t0 = time.time()
    conn = bundle.connect()
    _derive(conn)
    print(f"  derived latent fields ({time.time() - t0:.0f}s)", flush=True)
    ck = validate(bundle, conn, outputs_dir, run, run_name, data_dir)
    if ck.failures:
        conn.close()
        raise RuntimeError(f"validation failed ({len(ck.failures)}): " + "; ".join(ck.failures))
    conn.execute("ANALYZE")
    conn.commit()
    conn.execute("VACUUM")
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    counts = dict(
        latents=q("SELECT COUNT(*) FROM latent"),
        dead_latents=q("SELECT COUNT(*) FROM latent WHERE active_count = 0"),
        targets=q("SELECT COUNT(*) FROM target"),
        targets_by_status=dict(conn.execute("SELECT status, COUNT(*) FROM target GROUP BY status").fetchall()),
        circuits=q("SELECT COUNT(*) FROM circuit"),
        circuits_wired=q("SELECT COUNT(*) FROM circuit WHERE topology = 'wired'"),
        circuits_pass=q("SELECT COUNT(*) FROM circuit WHERE pass = 1"),
        members=q("SELECT COUNT(*) FROM member"),
        latents_in_any_circuit=q("SELECT COUNT(*) FROM latent WHERE n_memberships > 0"),
        target_ctx_rows=q("SELECT COUNT(*) FROM target_ctx"),
        sequences=int(np.load(os.path.join(bundle.tokens, "tokens.npy"), mmap_mode="r").shape[0]),
    )
    features = dict(
        has_token_acts=bool(q("SELECT COUNT(*) FROM target_ctx WHERE acts IS NOT NULL")),
        has_wired=counts["circuits_wired"] > 0,
        has_train_curves=bool(q("SELECT COUNT(*) FROM train_curve")),
        has_search=bool(stage_infos.get("7")),
    )
    conn.close()
    from export.explorer import stage_search
    files = ["explorer.sqlite"] + [f"arrays/{f}" for f in sorted(os.listdir(bundle.arrays))] + ["tokens/tokens.npy"]
    if features["has_search"]:
        files += stage_search.search_files(bundle)
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        bundle_id=bundle_id,
        built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        primary_run=run_name,
        sources=dict(stores=outputs_dir, data=data_dir, run=run.dir, repo=_git(repo_root)),
        stages=stage_infos,
        counts=counts,
        features=features,
        search=stage_search.manifest_fields(stage_infos["7"]) if features["has_search"] else None,
        conventions=dict(
            gid="(layer * 3 + kind_idx) * 40960 + latent, kinds attn, mlp, resid",
            seq_id="1-based; tokens row = seq_id - 1; 0 = empty context slot",
            coact="sorted by PMI desc; gid -1 = empty slot",
            tokenizer="microsoft/Phi-3-mini-4k-instruct",
        ),
        sizes={f: os.path.getsize(os.path.join(bundle.root, f)) for f in files},
        checksums={f: _sha256(os.path.join(bundle.root, f)) for f in files} if checksums else {},
        validation=dict(passed=ck.passed),
    )
    with open(os.path.join(bundle.root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(clean(manifest), fh, indent=1)
    return dict(counts=counts, features=features, n_checks=len(ck.passed), secs=round(time.time() - t0, 1))
