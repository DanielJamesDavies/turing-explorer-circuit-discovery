"""Tokenise a text corpus into the pipeline's shard format for a public model family (DAN-37).

Output (under --out, e.g. data_public/pythia-70m):

  shard_<N>.npy            int64, flat: [lead, t_1 .. t_L, -1, lead, t_1 .. t_L, -1, ...]
                           exactly what src/data/loader.py reads (segments between -1 separators, the first token of each
                           segment dropped: skip_first_token=True, the default every consumer uses). The loaded sequence
                           is therefore t_1 .. t_L, L = the SAEs' context (64), with NO BOS (both public SAE families were
                           trained with prepend_bos=False). `lead` is the tokenizer's BOS/EOS id, so a reader that keeps
                           the first token (skip_first_token=False) sees the prepend_bos form.
                           Sequence ids are positional over the sorted shard list, 1-based, as for TuringLLM.
  provenance/shard_<N>.docs.npy   int64 [n_seq]: source document index of each sequence (first token's document;
                           with --token-field: the source ROW index, rows counted over all files in order).
  provenance/shard_<N>.offsets.npy   int64 [n_seq], --token-field only: the window's token offset inside its row.
                           In a subdirectory: the loader lists every top-level *.npy as a shard.
  corpus_manifest.json     source (incl. the HF snapshot revision), licence (placeholder until confirmed), tokenizer,
                           sequence format, counts, duplicate statistics, per-shard sequence counts and bytes, command
                           line, git commit.

Windowing (text, the default): documents are tokenised without special tokens and concatenated with the tokenizer's
EOS between documents (standard packing; --no-doc-sep reproduces 033/070's plain concatenation), then cut into
consecutive non-overlapping L-token windows.
Windowing (--token-field FIELD, pre-tokenised sources such as apollo-research/Skylion007-openwebtext-tokenizer-gpt2):
each row's FIELD is a token-id list already packed by the source (apollo: 1,024-token chunks of EOS-separated
documents, no BOS). Every row is re-cut on its own into consecutive non-overlapping L-token windows; the row's last
len % L tokens are dropped (counted in the manifest) and no window spans two rows (rows are independent chunks, often
shuffled). Ids are checked against the family tokenizer's vocabulary; the tokenizer itself is not re-applied, so the
source must have been tokenised with the family's tokenizer (the apollo set: gpt2).
Both paths: exact duplicate windows (same L tokens) are dropped (--no-dedupe keeps them): "dedupe contexts by
sequence" happens here, once, before any split, so every stored context is unique by content.

Sources (one of):
  --input PATH [PATH ...]      local files or directories: *.parquet, *.jsonl, *.jsonl.zst (needs `zstandard`;
                               multi-frame and long-window zstd streams are read to the end), *.txt (one document per
                               file; text path only)
  --hf-dataset REPO --hf-pattern GLOB [...]   files of an HF dataset repo via snapshot_download (honours
                               HF_HUB_OFFLINE; this is where a real run would download). --hf-revision pins a commit.
Document boundaries (text path): --doc-mode row (each parquet / jsonl row is a document; default) | wikitext (rows
are lines; a top-level " = Title = " heading starts a new article) | file (a whole file is one document).

Budget: --max-tokens (tokens kept in windows) and / or --max-sequences.

Example (the 069 smoke corpus, cached wikitext only):
  PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python experiments/069-public-data-pass/prepare_corpus.py \
      --family pythia-70m --out data_public/pythia-70m \
      --hf-dataset Salesforce/wikitext --hf-pattern "wikitext-103-raw-v1/train-00000-of-00002.parquet" \
      --doc-mode wikitext --max-sequences 4096 --seqs-per-shard 1024 \
      --source-name "wikitext-103-raw-v1 train (smoke)" --licence "CC BY-SA 3.0 / GFDL (wikitext-103)"

The pod corpora (sources verified 2026-10-23 by HF metadata calls; see experiments/082-public-runs/chain.sh):
  Pythia: --hf-dataset monology/pile-uncopyrighted --hf-pattern "train/00.jsonl.zst" (11.15 GB, rows {text, meta})
  GPT-2:  --hf-dataset apollo-research/Skylion007-openwebtext-tokenizer-gpt2 --token-field input_ids
          --hf-pattern "data/train-0000[0-6]-of-00073.parquet" (7 x 243 MB; 120,878 rows x 1,024 tokens per file)
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
from typing import Iterator, List, Optional, Tuple

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


def _hf_files(repo: str, patterns: List[str], revision: Optional[str] = None) -> Tuple[List[str], str]:
    """The matching files of an HF dataset snapshot, and the snapshot's commit sha (its directory name)."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns, revision=revision)
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root, pat), recursive=True))
    return sorted(set(files)), os.path.basename(os.path.normpath(root))


def _json_lines(path: str) -> Iterator[str]:
    """Non-empty lines of a .jsonl / .json / .jsonl.zst file, in file order. zstd: read across frames (a stream of
    concatenated frames is common for large dumps) with a 2 GB window limit (long-mode archives)."""
    if path.endswith(".jsonl.zst"):
        try:
            import zstandard
        except ImportError as error:  # pragma: no cover - environment dependent
            raise SystemExit("reading .jsonl.zst needs `pip install zstandard`") from error
        with open(path, "rb") as raw:
            dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
            reader = io.TextIOWrapper(dctx.stream_reader(raw, read_across_frames=True), encoding="utf-8")
            for line in reader:
                if line.strip():
                    yield line
    else:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield line


def _rows(path: str, text_field: str) -> Iterator[str]:
    """Text rows of one file, in file order."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for t in pf.read_row_group(rg, columns=[text_field])[text_field].to_pylist():
                yield t or ""
    elif path.endswith(".jsonl") or path.endswith(".json") or path.endswith(".jsonl.zst"):
        for line in _json_lines(path):
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


def _list_chunk_rows(chunk) -> Iterator[np.ndarray]:
    """The rows of one pyarrow list-typed chunk as numpy arrays (offsets fast path; nulls -> empty rows)."""
    import pyarrow as pa
    t = chunk.type
    if chunk.null_count == 0 and (pa.types.is_list(t) or pa.types.is_large_list(t)):
        offs = chunk.offsets.to_numpy()
        vals = chunk.values.to_numpy(zero_copy_only=False)
        for i in range(len(chunk)):
            yield vals[offs[i]:offs[i + 1]]
    else:
        for x in chunk.to_pylist():
            yield np.asarray(x or [], dtype=np.int64)


def token_rows(files: List[str], token_field: str) -> Iterator[np.ndarray]:
    """Token-id rows (pre-tokenised sources), in file order then row order."""
    for path in files:
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                for chunk in pf.read_row_group(rg, columns=[token_field]).column(token_field).chunks:
                    yield from _list_chunk_rows(chunk)
        elif path.endswith(".jsonl") or path.endswith(".json") or path.endswith(".jsonl.zst"):
            for line in _json_lines(path):
                yield np.asarray(json.loads(line).get(token_field) or [], dtype=np.int64)
        else:
            raise ValueError("--token-field reads *.parquet / *.jsonl / *.jsonl.zst, not %s" % path)


def _load_tokenizer(tokenizer_id: str):
    """The family tokenizer (a module-level hook so CPU tests can substitute a stub)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_id)


# ----------------------------------------------------------------------------------------------- writer
class ShardWriter:
    def __init__(self, out_dir: str, seq_len: int, lead: int, seqs_per_shard: int, offsets: bool = False) -> None:
        self.out_dir, self.seq_len, self.lead, self.per = out_dir, seq_len, int(lead), int(seqs_per_shard)
        os.makedirs(os.path.join(out_dir, "provenance"), exist_ok=True)
        existing = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
        if existing:
            raise SystemExit("%s already holds %d shard files; refusing to mix corpora" % (out_dir, len(existing)))
        self.windows: List[np.ndarray] = []
        self.docs: List[int] = []
        self.with_offsets = offsets
        self.offsets: List[int] = []
        self.shards: List[dict] = []

    def add(self, window: np.ndarray, doc: int, offset: Optional[int] = None) -> None:
        self.windows.append(window)
        self.docs.append(doc)
        if self.with_offsets:
            self.offsets.append(int(offset))
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
        if self.with_offsets:
            np.save(os.path.join(self.out_dir, "provenance", "shard_%d.offsets.npy" % idx),
                    np.asarray(self.offsets, np.int64))
        self.shards.append(dict(file=name, n_sequences=n, bytes=os.path.getsize(path)))
        self.windows, self.docs, self.offsets = [], [], []


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
    ap.add_argument("--hf-revision", default=None, help="dataset commit / branch (default main; the manifest records "
                    "the snapshot's commit sha either way)")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--token-field", default=None,
                    help="pre-tokenised source: the row field holding token ids (e.g. input_ids); rows are re-cut "
                         "into --seq-len windows, no tokenizer call")
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
    if args.token_field and (args.doc_mode != "row" or args.no_doc_sep):
        ap.error("--token-field rows are already packed by the source: --doc-mode / --no-doc-sep do not apply")

    from model.families import FAMILIES

    fam = FAMILIES[args.family]
    assert fam.sae is not None
    seq_len = int(args.seq_len or fam.sae.context_size)
    if fam.sae.prepend_bos:
        raise SystemExit("%s SAEs were trained with a BOS; this tool writes no-BOS windows" % fam.name)
    tok = _load_tokenizer(fam.tokenizer_id)
    tok.model_max_length = int(1e12)                          # silence the long-document warning; we cut windows
    eos = tok.eos_token_id
    lead = tok.bos_token_id if tok.bos_token_id is not None else eos
    if eos is None or lead is None:
        raise SystemExit("tokenizer %s has no EOS/BOS id" % fam.tokenizer_id)
    vocab = len(tok)

    if args.input:
        files = _expand_inputs(args.input)
        source = dict(kind="local", inputs=args.input)
    else:
        if not args.hf_pattern:
            ap.error("--hf-dataset needs --hf-pattern")
        files, revision = _hf_files(args.hf_dataset, args.hf_pattern, args.hf_revision)
        source = dict(kind="hf_dataset", repo=args.hf_dataset, patterns=args.hf_pattern, revision=revision,
                      revision_requested=args.hf_revision or "main")
    if args.token_field:
        files = [f for f in files if not f.endswith(".txt")]
    if not files:
        raise SystemExit("no input files found")
    source["files"] = [dict(path=f, bytes=os.path.getsize(f)) for f in files]

    max_seqs = args.max_sequences if args.max_sequences is not None else float("inf")
    if args.max_tokens is not None:
        max_seqs = min(max_seqs, args.max_tokens // seq_len)
    writer = ShardWriter(args.out, seq_len, lead, args.seqs_per_shard, offsets=bool(args.token_field))
    seen: set = set()
    st = dict(windows=0, dup=0, kept=0, eos_tokens=0, eos_windows=0)
    n_docs = n_doc_tokens = 0
    t0 = time.perf_counter()

    def keep(w: np.ndarray, doc: int, offset: Optional[int] = None) -> bool:
        """Dedupe + write one window; True once the budget is reached."""
        st["windows"] += 1
        if not args.no_dedupe:
            h = hashlib.blake2b(w.tobytes(), digest_size=8).digest()
            if h in seen:
                st["dup"] += 1
                return False
            seen.add(h)
        writer.add(w, doc, offset)
        st["kept"] += 1
        n_eos = int((w == eos).sum())
        st["eos_tokens"] += n_eos
        st["eos_windows"] += int(n_eos > 0)
        return st["kept"] >= max_seqs

    row_stats = dict(rows_shorter_than_window=0, row_tail_tokens_dropped=0, row_lengths={})
    if args.token_field:
        # ---------------------------------------------------------------- pre-tokenised rows: re-cut each row alone
        for row in token_rows(files, args.token_field):
            doc = n_docs
            n_docs += 1
            n_doc_tokens += len(row)
            lens = row_stats["row_lengths"]
            if len(lens) < 16 or len(row) in lens:                 # a short histogram, not one entry per odd length
                lens[len(row)] = lens.get(len(row), 0) + 1
            n_full = len(row) // seq_len
            row_stats["row_tail_tokens_dropped"] += len(row) - n_full * seq_len
            if n_full == 0:
                row_stats["rows_shorter_than_window"] += 1
                continue
            windows = np.asarray(row[:n_full * seq_len], dtype=np.int64).reshape(n_full, seq_len)
            lo, hi = int(windows.min()), int(windows.max())
            if lo < 0 or hi >= vocab:
                raise SystemExit("row %d holds token id %d outside [0, %d) of %s's tokenizer: wrong --token-field or "
                                 "a source tokenised with another tokenizer" % (doc, lo if lo < 0 else hi, vocab,
                                                                                fam.tokenizer_id))
            if any(keep(windows[j], doc, j * seq_len) for j in range(n_full)):
                break
            if n_docs % 100000 == 0:
                print("  %d rows, %d sequences kept (%.0f s)" % (n_docs, st["kept"], time.perf_counter() - t0),
                      flush=True)
    else:
        # ---------------------------------------------------------------- text: tokenise, pack with EOS, cut
        carry = np.zeros(0, dtype=np.int64)                    # < seq_len tokens left over from earlier documents
        carry_doc = -1                                         # document of carry[0]
        sep = np.asarray([] if args.no_doc_sep else [eos], dtype=np.int64)

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
                first_doc = carry_doc if len(carry) else doc   # carry < seq_len, so only window 0 starts in it
                if n_full == 0:
                    carry, carry_doc = stream, first_doc
                    continue
                windows = stream[:n_full * seq_len].reshape(n_full, seq_len)
                carry, carry_doc = stream[n_full * seq_len:], doc
                for j in range(n_full):
                    if keep(windows[j], first_doc if j == 0 else doc):
                        done = True
                        break
                if done:
                    break
            if done:
                break
            if n_docs and n_docs % (args.tokenize_batch * 50) == 0:
                print("  %d docs, %d sequences kept (%.0f s)" % (n_docs, st["kept"], time.perf_counter() - t0),
                      flush=True)
    writer.flush()
    n_windows, n_dup, n_kept = st["windows"], st["dup"], st["kept"]

    if args.token_field:
        corpus_input = dict(input="token_ids", token_field=args.token_field, document_unit="source row")
        fmt_sep = "as packed by the source (pre-tokenised rows; not re-tokenised)"
        fmt_windows = ("each source row cut alone into consecutive, non-overlapping windows; the row's last "
                       "len %% %d tokens dropped; no window spans two rows" % seq_len)
        extra_counts = dict(rows_read=n_docs, row_tokens_read=n_doc_tokens,
                            rows_shorter_than_window=row_stats["rows_shorter_than_window"],
                            row_tail_tokens_dropped=row_stats["row_tail_tokens_dropped"])
        # a short histogram (first 16 distinct lengths): apollo's rows should all be 1,024 tokens
        corpus_input["row_length_histogram"] = {str(k): v for k, v in sorted(row_stats["row_lengths"].items())}
    else:
        corpus_input = dict(input="text", text_field=args.text_field, doc_mode=args.doc_mode)
        fmt_sep = None if args.no_doc_sep else "eos"
        fmt_windows = "consecutive, non-overlapping"
        extra_counts = {}
    manifest = dict(
        schema="public_corpus_manifest_v1",
        created=_dt.datetime.now().isoformat(timespec="seconds"),
        family=fam.name,
        corpus=dict(name=args.source_name or (args.hf_dataset or ",".join(args.input)), licence=args.licence,
                    source=source, **corpus_input),
        tokenizer=dict(id=fam.tokenizer_id, vocab_size=vocab, eos_token_id=eos, lead_token_id=lead),
        format=dict(seq_len=seq_len, prepend_bos=False, lead_token="tokenizer BOS/EOS id, dropped by the loader "
                    "(skip_first_token=True)", separator=-1, dtype="int64", doc_separator=fmt_sep,
                    windows=fmt_windows, sequence_ids="1-based, positional over the sorted shard list",
                    provenance="provenance/shard_<N>.docs.npy (source document / row index)"
                    + (", provenance/shard_<N>.offsets.npy (token offset in the row)" if args.token_field else "")),
        counts=dict(documents_read=n_docs, document_tokens_read=n_doc_tokens, windows_cut=n_windows,
                    duplicate_windows_dropped=n_dup,
                    duplicate_window_rate=(n_dup / n_windows) if n_windows else 0.0,
                    sequences=n_kept, tokens=n_kept * seq_len,
                    eos_tokens_in_sequences=st["eos_tokens"], sequences_with_eos=st["eos_windows"],
                    **extra_counts),
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
