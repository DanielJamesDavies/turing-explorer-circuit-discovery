"""Stage 7: keyword-search index over every latent's top contexts.

Layout (bundle/search/):
    search.sqlite    FTS5 table `fts` over the decoded text of each unique sequence
                     (contentless, rowid = seq_id; the text itself is not stored, the
                     backend re-decodes snippets from tokens/tokens.npy) + a `meta` table
    seq_ids.npy      int32  [n_seq_u]      unique sequence ids, ascending
    ref_ptr.npy      int64  [n_seq_u + 1]  CSR offsets into ref_gid / ref_rank
    ref_gid.npy      int32  [n_refs]       latent (gid) whose top-k contains the sequence
    ref_rank.npy     uint8  [n_refs]       its 0-based rank in that latent's top contexts

Only the first `k` slots of arrays/top_ids.npy are indexed. Many latents share
contexts, so each sequence is decoded and indexed once; the CSR arrays map a matching
sequence back to the latents (and ranks) that hold it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

import numpy as np

from export.explorer import ids
from export.explorer.bundle import Bundle

TOKENIZER_ID = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_K = 16
NEWLINE_ID = 13          # <0x0A>; special tokens (<s>, </s>, <unk>, <|...|>) decode as a newline
FIRST_ADDED_ID = 32000   # Phi-3 chat / padding tokens
FTS_OPTIONS = ("content='', columnsize=0, detail=full, "
               "tokenize='unicode61 remove_diacritics 2'")
DECODE_CHUNK = 50_000
ARRAYS = ("seq_ids", "ref_ptr", "ref_gid", "ref_rank")


def search_dir(bundle: Bundle) -> str:
    return os.path.join(bundle.root, "search")


def search_files(bundle: Bundle) -> list:
    """Bundle-relative paths of the search index files that exist."""
    d = search_dir(bundle)
    names = ["search.sqlite"] + [f"{a}.npy" for a in ARRAYS]
    return [f"search/{n}" for n in names if os.path.isfile(os.path.join(d, n))]


def _save(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _load_tokenizer():
    import logging

    from transformers import AutoTokenizer
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)


def _refs(bundle: Bundle, k: int):
    """-> (seq_ids, ref_ptr, ref_gid, ref_rank) CSR grouped by sequence id."""
    top = np.load(os.path.join(bundle.arrays, "top_ids.npy"), mmap_mode="r")
    if top.shape != (ids.N_LATENTS, top.shape[1]) or k > top.shape[1]:
        raise ValueError(f"top_ids shape {top.shape} does not allow k={k}")
    sid = np.asarray(top[:, :k])
    gid, rank = np.nonzero(sid)
    seq = sid[gid, rank]
    del sid
    order = np.argsort(seq, kind="stable")  # stable: refs of one sequence stay in gid order
    seq, gid, rank = seq[order], gid[order].astype(np.int32), rank[order].astype(np.uint8)
    del order
    seq_ids, starts = np.unique(seq, return_index=True)
    ref_ptr = np.append(starts, len(seq)).astype(np.int64)
    return seq_ids.astype(np.int32), ref_ptr, gid, rank


def _texts(tokenizer, rows: np.ndarray) -> list:
    rows = rows.astype(np.int64)
    rows[(rows <= 2) | (rows >= FIRST_ADDED_ID)] = NEWLINE_ID
    return tokenizer.batch_decode(rows.tolist())


def _validate(conn, tokenizer, tokens, top, k, seq_ids, ref_ptr, ref_gid, ref_rank) -> list:
    """Spot checks of the FTS table and the CSR refs; returns failure strings."""
    import re
    failures = []
    # 1) a word of a random indexed sequence finds that sequence
    bad = []
    for s in np.random.default_rng(0).choice(seq_ids, size=200, replace=False):
        words = re.findall(r"[^\W_]{4,}", _texts(tokenizer, np.asarray(tokens[[int(s) - 1]]))[0])
        if not words:
            continue
        w = words[len(words) // 2]
        hit = conn.execute("SELECT 1 FROM fts WHERE fts MATCH ? AND rowid = ?", (f'"{w}"', int(s))).fetchone()
        if hit is None:
            bad.append(f"{s}:{w}")
    if bad:
        failures.append(f"fts misses words of its own sequences: {bad[:5]}")
    # 2) CSR refs == top_ids[:, :k] for random latents
    bad = []
    n_u = len(seq_ids)
    for g in np.random.default_rng(1).choice(ids.N_LATENTS, size=500, replace=False):
        want = {(int(s), r) for r, s in enumerate(top[g, :k]) if s != 0}
        got = set()
        for s, _ in want:
            p = int(np.searchsorted(seq_ids, s))
            if p < n_u and seq_ids[p] == s:
                sl = slice(ref_ptr[p], ref_ptr[p + 1])
                got |= {(s, int(r)) for gg, r in zip(ref_gid[sl], ref_rank[sl]) if gg == g}
        if got != want:
            bad.append(int(g))
    if bad:
        failures.append(f"CSR refs != top_ids for latents {bad[:5]}")
    return failures


def build(bundle: Bundle, k: int = DEFAULT_K) -> dict:
    t0 = time.time()
    out = search_dir(bundle)
    os.makedirs(out, exist_ok=True)

    seq_ids, ref_ptr, ref_gid, ref_rank = _refs(bundle, k)
    n_u, n_refs = len(seq_ids), len(ref_gid)
    print(f"  {n_refs} refs (k={k}) -> {n_u} unique sequences ({time.time() - t0:.0f}s)", flush=True)

    # ---- FTS over decoded sequence text, built in a temp file then swapped in
    db_path = os.path.join(out, "search.sqlite")
    tmp = db_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    conn = sqlite3.connect(tmp)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-1000000")
    conn.execute(f"CREATE VIRTUAL TABLE fts USING fts5(text, {FTS_OPTIONS})")
    conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    tokenizer = _load_tokenizer()
    tokens = np.load(os.path.join(bundle.tokens, "tokens.npy"), mmap_mode="r")
    t_dec = t_ins = 0.0
    for i in range(0, n_u, DECODE_CHUNK):
        chunk = seq_ids[i:i + DECODE_CHUNK]
        t = time.time()
        texts = _texts(tokenizer, np.asarray(tokens[chunk.astype(np.int64) - 1]))
        t_dec += time.time() - t
        t = time.time()
        conn.executemany("INSERT INTO fts(rowid, text) VALUES (?, ?)", zip(chunk.tolist(), texts))
        t_ins += time.time() - t
        if (i // DECODE_CHUNK) % 20 == 0:
            print(f"    fts: {i + len(chunk)}/{n_u} ({time.time() - t0:.0f}s)", flush=True)
    conn.commit()
    t = time.time()
    conn.execute("INSERT INTO fts(fts) VALUES ('optimize')")
    conn.commit()
    t_opt = time.time() - t

    top = np.load(os.path.join(bundle.arrays, "top_ids.npy"), mmap_mode="r")
    failures = _validate(conn, tokenizer, tokens, top, k, seq_ids, ref_ptr, ref_gid, ref_rank)
    if failures:
        conn.close()
        os.remove(tmp)
        raise RuntimeError("search index validation failed: " + "; ".join(failures))

    meta = dict(k=k, n_sequences=n_u, n_refs=n_refs, tokenizer=TOKENIZER_ID, fts=FTS_OPTIONS,
                built_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    conn.executemany("INSERT INTO meta VALUES (?, ?)", [(key, json.dumps(v)) for key, v in meta.items()])
    conn.commit()
    t = time.time()
    conn.execute("VACUUM")
    t_vac = time.time() - t
    conn.close()
    os.replace(tmp, db_path)

    for name, arr in zip(ARRAYS, (seq_ids, ref_ptr, ref_gid, ref_rank)):
        _save(os.path.join(out, f"{name}.npy"), arr)
    sizes = {f: os.path.getsize(os.path.join(bundle.root, f)) for f in search_files(bundle)}
    info = dict(k=k, n_sequences=n_u, n_refs=n_refs, tokenizer=TOKENIZER_ID, fts=FTS_OPTIONS,
                sizes=sizes, bytes=sum(sizes.values()),
                secs_decode=round(t_dec, 1), secs_insert=round(t_ins, 1), secs_optimize=round(t_opt, 1),
                secs_vacuum=round(t_vac, 1), secs=round(time.time() - t0, 1))
    _update_manifest(bundle, info)
    return info


def manifest_fields(info: dict) -> dict:
    """The manifest `search` section for a stage 7 marker."""
    return dict(k=info["k"], n_sequences=info["n_sequences"], n_refs=info["n_refs"],
                tokenizer=info["tokenizer"], fts=info["fts"], bytes=info["bytes"],
                files=sorted(info["sizes"]))


def _update_manifest(bundle: Bundle, info: dict) -> None:
    """Record the index in an existing manifest (stage 6 writes it the same way on re-runs)."""
    path = os.path.join(bundle.root, "manifest.json")
    if not os.path.isfile(path):
        print("  no manifest.json yet; stage 6 will record the search index", flush=True)
        return
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest.setdefault("features", {})["has_search"] = True
    manifest["search"] = manifest_fields(info)
    manifest.setdefault("stages", {})["7"] = info
    manifest.setdefault("sizes", {}).update(info["sizes"])
    if manifest.get("checksums"):
        from export.explorer.stage_finalize import _sha256
        manifest["checksums"].update({f: _sha256(os.path.join(bundle.root, f)) for f in info["sizes"]})
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, path)
    print("  manifest.json: features.has_search = true, search section added", flush=True)
