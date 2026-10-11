"""Tokenise a text corpus into the pipeline's shard format for a public model family (DAN-37).

Output (under --out, e.g. data_public/pythia-70m):

  shard_<N>.npy            int64, flat: [lead, t_1 .. t_L, -1, lead, t_1 .. t_L, -1, ...]
                           exactly what src/data/loader.py reads (segments between -1 separators, the first token of each
                           segment dropped: skip_first_token=True, the default every consumer uses). The loaded sequence
                           is therefore t_1 .. t_L, L = the SAEs' context (64), with NO BOS (both public SAE families were
                           trained with prepend_bos=False). `lead` is the tokenizer's BOS/EOS id, so a reader that keeps
                           the first token (skip_first_token=False) sees the prepend_bos form.
                           Sequence ids are positional over the sorted shard list, 1-based, as for TuringLLM.
  provenance/shard_<N>.docs.npy   int64 [n_seq]: source document index of each sequence (first token's document).
                           In a subdirectory: the loader lists every top-level *.npy as a shard.
  corpus_manifest.json     source, licence (placeholder until confirmed), tokenizer, sequence format, counts,
                           duplicate statistics, per-shard sequence counts and bytes, command line, git commit.

Windowing: documents are tokenised without special tokens and concatenated with the tokenizer's EOS between documents
(standard packing; --no-doc-sep reproduces 033/070's plain concatenation), then cut into
consecutive non-overlapping L-token windows. Exact duplicate windows (same L tokens) are dropped (--no-dedupe keeps
them): "dedupe contexts by sequence" happens here, once, so every stored context is unique by content.

Sources (one of):
  --input PATH [PATH ...]      local files or directories: *.parquet, *.jsonl, *.jsonl.zst (needs `zstandard`),
                               *.txt (one document per file)
  --hf-dataset REPO --hf-pattern GLOB [...]   files of an HF dataset repo via snapshot_download (honours
                               HF_HUB_OFFLINE; this is where a real run would download)
Document boundaries: --doc-mode row (each parquet / jsonl row is a document; default) | wikitext (rows are lines;
a top-level " = Title = " heading starts a new article) | file (a whole file is one document).

Budget: --max-tokens (tokens kept in windows) and / or --max-sequences.

Example (the 069 smoke corpus, cached wikitext only):
  PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python experiments/069-public-data-pass/prepare_corpus.py \
      --family pythia-70m --out data_public/pythia-70m \
      --hf-dataset Salesforce/wikitext --hf-pattern "wikitext-103-raw-v1/train-00000-of-00002.parquet" \
      --doc-mode wikitext --max-sequences 4096 --seqs-per-shard 1024 \
      --source-name "wikitext-103-raw-v1 train (smoke)" --licence "CC BY-SA 3.0 / GFDL (wikitext-103)"
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from typing import Iterator, List, Optional

import numpy as np

_WIKITEXT_TITLE = re.compile(r"^ = [^=].* = \s*$")


# ----------------------------------------------------------------------------------------------- sources
def _expand_inputs(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                out.extend(os.path.join(root, f) for f in files)
        else:
            matched = glob.glob(p)
            out.extend(matched if matched else [p])
    keep = (".parquet", ".jsonl", ".jsonl.zst", ".json", ".txt")
    return sorted(f for f in out if f.endswith(keep))


def _hf_files(repo: str, patterns: List[str]) -> List[str]:
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns)
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root, pat), recursive=True))
    return sorted(set(files))


def _rows(path: str, text_field: str) -> Iterator[str]:
    """Text rows of one file, in file order."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for t in pf.read_row_group(rg, columns=[text_field])[text_field].to_pylist():
                yield t or ""
    elif path.endswith(".jsonl") or path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line).get(text_field) or ""
    elif path.endswith(".jsonl.zst"):
        try:
            import zstandard
        except ImportError as error:  # pragma: no cover - environment dependent
            raise SystemExit("reading .jsonl.zst needs `pip install zstandard`") from error
        with open(path, "rb") as raw:
            reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
            for line in reader:
                if line.strip():
                    yield json.loads(line).get(text_field) or ""
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield fh.read()
    else:
        raise ValueError("unsupported input file %s" % path)


def documents(files: List[str], text_field: str, doc_mode: str) -> Iterator[str]:
    for path in files:
        if doc_mode == "file":
            yield "".join(_rows(path, text_field))
        elif doc_mode == "row":
            for t in _rows(path, text_field):
                if t.strip():
                    yield t
        elif doc_mode == "wikitext":
            buf: List[str] = []
            for line in _rows(path, text_field):
                if _WIKITEXT_TITLE.match(line) and buf:
                    yield "".join(buf)
                    buf = []
                if line.strip():
                    buf.append(line)
            if buf:
                yield "".join(buf)
        else:
            raise ValueError("unknown doc mode %r" % doc_mode)


# ----------------------------------------------------------------------------------------------- writer
class ShardWriter:
    def __init__(self, out_dir: str, seq_len: int, lead: int, seqs_per_shard: int) -> None:
        self.out_dir, self.seq_len, self.lead, self.per = out_dir, seq_len, int(lead), int(seqs_per_shard)
        os.makedirs(os.path.join(out_dir, "provenance"), exist_ok=True)
        existing = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
        if existing:
            raise SystemExit("%s already holds %d shard files; refusing to mix corpora" % (out_dir, len(existing)))
        self.windows: List[np.ndarray] = []
        self.docs: List[int] = []
        self.shards: List[dict] = []

    def add(self, window: np.ndarray, doc: int) -> None:
        self.windows.append(window)
        self.docs.append(doc)
        if len(self.windows) == self.per:
            self.flush()

    def flush(self) -> None:
        if not self.windows:
            return
        n = len(self.windows)
        seg = np.empty((n, self.seq_len + 2), dtype=np.int64)
        seg[:, 0] = self.lead
        seg[:, 1:-1] = np.stack(self.windows)
        seg[:, -1] = -1
        idx = len(self.shards)
        name = "shard_%d.npy" % idx
        path = os.path.join(self.out_dir, name)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, seg.reshape(-1))
        os.replace(tmp, path)
        np.save(os.path.join(self.out_dir, "provenance", "shard_%d.docs.npy" % idx), np.asarray(self.docs, np.int64))
        self.shards.append(dict(file=name, n_sequences=n, bytes=os.path.getsize(path)))
        self.windows, self.docs = [], []


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", required=True, choices=["pythia-70m", "gpt2-small"])
    ap.add_argument("--out", required=True, help="shard directory, e.g. data_public/<family>")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", nargs="+", help="local files / directories / globs")
    src.add_argument("--hf-dataset", help="HF dataset repo id (files fetched with snapshot_download)")
    ap.add_argument("--hf-pattern", nargs="+", default=None, help="allow_patterns inside --hf-dataset")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--doc-mode", default="row", choices=["row", "wikitext", "file"])
    ap.add_argument("--max-tokens", type=int, default=None, help="stop after this many window tokens are kept")
    ap.add_argument("--max-sequences", type=int, default=None, help="stop after this many sequences are kept")
    ap.add_argument("--seq-len", type=int, default=None, help="default: the family's SAE context size (64)")
    ap.add_argument("--seqs-per-shard", type=int, default=8192, help="TuringLLM's shards hold 8,192")
    ap.add_argument("--no-dedupe", action="store_true", help="keep exact duplicate windows")
    ap.add_argument("--no-doc-sep", action="store_true", help="no EOS between documents (033/070 windows)")
    ap.add_argument("--tokenize-batch", type=int, default=512, help="documents per tokenizer call")
    ap.add_argument("--source-name", default=None, help="human-readable corpus name for the manifest")
    ap.add_argument("--licence", default="UNCONFIRMED - must be confirmed before the corpus is used for the paper",
                    help="corpus licence for the manifest")
    args = ap.parse_args(argv)
    if args.max_tokens is None and args.max_sequences is None:
        ap.error("give a budget: --max-tokens and/or --max-sequences")

    from transformers import AutoTokenizer
    from model.families import FAMILIES

    fam = FAMILIES[args.family]
    assert fam.sae is not None
    seq_len = int(args.seq_len or fam.sae.context_size)
    if fam.sae.prepend_bos:
        raise SystemExit("%s SAEs were trained with a BOS; this tool writes no-BOS windows" % fam.name)
    tok = AutoTokenizer.from_pretrained(fam.tokenizer_id)
    tok.model_max_length = int(1e12)                          # silence the long-document warning; we cut windows
    eos = tok.eos_token_id
    lead = tok.bos_token_id if tok.bos_token_id is not None else eos
    if eos is None or lead is None:
        raise SystemExit("tokenizer %s has no EOS/BOS id" % fam.tokenizer_id)

    if args.input:
        files = _expand_inputs(args.input)
        source = dict(kind="local", inputs=args.input)
    else:
        if not args.hf_pattern:
            ap.error("--hf-dataset needs --hf-pattern")
        files = _hf_files(args.hf_dataset, args.hf_pattern)
        source = dict(kind="hf_dataset", repo=args.hf_dataset, patterns=args.hf_pattern)
    if not files:
        raise SystemExit("no input files found")
    source["files"] = [dict(path=f, bytes=os.path.getsize(f)) for f in files]

    max_seqs = args.max_sequences if args.max_sequences is not None else float("inf")
    if args.max_tokens is not None:
        max_seqs = min(max_seqs, args.max_tokens // seq_len)
    writer = ShardWriter(args.out, seq_len, lead, args.seqs_per_shard)
    seen: set = set()
    n_docs = n_windows = n_dup = n_kept = 0
    n_doc_tokens = 0
    carry = np.zeros(0, dtype=np.int64)                        # < seq_len tokens left over from earlier documents
    carry_doc = -1                                             # document of carry[0]
    sep = np.asarray([] if args.no_doc_sep else [eos], dtype=np.int64)
    t0 = time.perf_counter()

    def batches() -> Iterator[List[str]]:
        batch: List[str] = []
        for d in documents(files, args.text_field, args.doc_mode):
            batch.append(d)
            if len(batch) == args.tokenize_batch:
                yield batch
                batch = []
        if batch:
            yield batch

    done = False
    for batch in batches():
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            doc = n_docs
            n_docs += 1
            n_doc_tokens += len(ids)
            parts = [carry, sep, np.asarray(ids, dtype=np.int64)] if doc > 0 else [np.asarray(ids, dtype=np.int64)]
            stream = np.concatenate(parts)
            n_full = len(stream) // seq_len
            first_doc = carry_doc if len(carry) else doc       # carry < seq_len, so only window 0 starts in it
            if n_full == 0:
                carry, carry_doc = stream, first_doc
                continue
            windows = stream[:n_full * seq_len].reshape(n_full, seq_len)
            carry, carry_doc = stream[n_full * seq_len:], doc
            for j in range(n_full):
                w = windows[j]
                wdoc = first_doc if j == 0 else doc
                n_windows += 1
                if not args.no_dedupe:
                    h = hashlib.blake2b(w.tobytes(), digest_size=8).digest()
                    if h in seen:
                        n_dup += 1
                        continue
                    seen.add(h)
                writer.add(w, wdoc)
                n_kept += 1
                if n_kept >= max_seqs:
                    done = True
                    break
            if done:
                break
        if done:
            break
        if n_docs and n_docs % (args.tokenize_batch * 50) == 0:
            print("  %d docs, %d sequences kept (%.0f s)" % (n_docs, n_kept, time.perf_counter() - t0), flush=True)
    writer.flush()

    manifest = dict(
        schema="public_corpus_manifest_v1",
        created=_dt.datetime.now().isoformat(timespec="seconds"),
        family=fam.name,
        corpus=dict(name=args.source_name or (args.hf_dataset or ",".join(args.input)), licence=args.licence,
                    source=source, text_field=args.text_field, doc_mode=args.doc_mode),
        tokenizer=dict(id=fam.tokenizer_id, vocab_size=len(tok), eos_token_id=eos, lead_token_id=lead),
        format=dict(seq_len=seq_len, prepend_bos=False, lead_token="tokenizer BOS/EOS id, dropped by the loader "
                    "(skip_first_token=True)", separator=-1, dtype="int64", doc_separator=None if args.no_doc_sep
                    else "eos", windows="consecutive, non-overlapping", sequence_ids="1-based, positional over the "
                    "sorted shard list"),
        counts=dict(documents_read=n_docs, document_tokens_read=n_doc_tokens, windows_cut=n_windows,
                    duplicate_windows_dropped=n_dup,
                    duplicate_window_rate=(n_dup / n_windows) if n_windows else 0.0,
                    sequences=n_kept, tokens=n_kept * seq_len),
        dedupe=not args.no_dedupe,
        shards=writer.shards,
        n_shards=len(writer.shards),
        command=" ".join([os.path.basename(sys.executable)] + sys.argv),
        git_commit=_git_commit(),
        elapsed_s=round(time.perf_counter() - t0, 1),
    )
    with open(os.path.join(args.out, "corpus_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("wrote %d sequences (%d tokens) in %d shards to %s; %d duplicate windows dropped (%.2f%%), %.0f s"
          % (n_kept, n_kept * seq_len, len(writer.shards), args.out, n_dup,
             100.0 * manifest["counts"]["duplicate_window_rate"], manifest["elapsed_s"]))
    return manifest


if __name__ == "__main__":
    main()
