"""069 prepare_corpus.py (DAN-37 / B6): pre-tokenised sources (--token-field) and Pile-style *.jsonl.zst.

CPU only, no network: a tiny synthetic token-id dataset (parquet / jsonl / jsonl.zst) and a stub tokenizer stand in
for apollo-research/Skylion007-openwebtext-tokenizer-gpt2 (rows = 1,024-token packed chunks, field input_ids) and
monology/pile-uncopyrighted (rows {text, meta}). The shards are read back with src/data/loader.py, the reader every
consumer uses.
"""
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from config import config
from data.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SEQ = 8
EOS = 99
VOCAB = 100


def _module():
    spec = importlib.util.spec_from_file_location(
        "prepare_corpus_069", ROOT / "experiments" / "069-public-data-pass" / "prepare_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubTokenizer:
    """GPT-2-like: EOS, no BOS; a 'document' is whitespace-separated integer ids."""
    eos_token_id = EOS
    bos_token_id = None
    model_max_length = 1024

    def __len__(self):
        return VOCAB

    def __call__(self, batch, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [[int(w) for w in t.split()] for t in batch]}


@pytest.fixture
def pc(monkeypatch):
    mod = _module()
    monkeypatch.setattr(mod, "_load_tokenizer", lambda _tid: StubTokenizer())
    return mod


# rows of a packed, pre-tokenised source (lengths 20, 5, 19, 16)
ROWS = [
    list(range(1, 21)),                                   # windows 1..8, 9..16; tail 4
    [5, 6, 7, 8, 9],                                      # shorter than a window
    list(range(1, 9)) + list(range(50, 58)) + [EOS, 1, 2],  # 1..8 repeats row 0's first window; 50..57; tail 3
    list(range(60, 68)) + list(range(70, 78)),            # two windows, no tail
]
KEPT = [list(range(1, 9)), list(range(9, 17)), list(range(50, 58)), list(range(60, 68)), list(range(70, 78))]
KEPT_DOCS = [0, 0, 2, 3, 3]
KEPT_OFFSETS = [0, 8, 8, 0, 8]


def _write_parquet(path, rows, field="input_ids"):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({field: pa.array(rows, type=pa.list_(pa.int32()))})
    pq.write_table(table, path, row_group_size=2)                 # two row groups: the reader must walk both
    return path


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def _read_back(out_dir, monkeypatch):
    shards = sorted(p.name for p in Path(out_dir).glob("*.npy"))
    monkeypatch.setattr(config.data, "dataset_path", str(out_dir))
    monkeypatch.setattr(config.data, "n_shards", len(shards))
    loader = DataLoader(device=torch.device("cpu"))
    n = sum(loader._shard_sequence_counts)
    return [loader.get_sequence(i).tolist() for i in range(1, n + 1)], shards


def _run(pc, tmp_path, inputs, *extra):
    out = tmp_path / "out"
    argv = ["--family", "gpt2-small", "--out", str(out), "--input", *[str(p) for p in inputs],
            "--seq-len", str(SEQ), "--seqs-per-shard", "2", "--max-sequences", "1000", *extra]
    return pc.main(argv), out


def _provenance(out, stem):
    return np.concatenate([np.load(p) for p in sorted((out / "provenance").glob("shard_*.%s.npy" % stem),
                                                      key=lambda p: int(p.name.split("_")[1].split(".")[0]))])


def test_token_field_parquet_recut_dedupe_provenance(pc, tmp_path, monkeypatch):
    src = _write_parquet(tmp_path / "train-00000-of-00001.parquet", ROWS)
    man, out = _run(pc, tmp_path, [src], "--token-field", "input_ids")

    seqs, shards = _read_back(out, monkeypatch)
    assert seqs == KEPT                                           # the loader's sequences = the re-cut windows
    assert shards == ["shard_0.npy", "shard_1.npy", "shard_2.npy"]  # provenance/ is not listed as shards
    flat = np.load(out / "shard_0.npy")
    assert flat.tolist() == [EOS] + KEPT[0] + [-1] + [EOS] + KEPT[1] + [-1]  # lead (EOS: no BOS) + window + sep
    assert _provenance(out, "docs").tolist() == KEPT_DOCS
    assert _provenance(out, "offsets").tolist() == KEPT_OFFSETS

    c = man["counts"]
    assert (c["rows_read"], c["documents_read"], c["row_tokens_read"]) == (4, 4, 60)
    assert (c["windows_cut"], c["duplicate_windows_dropped"], c["sequences"]) == (6, 1, 5)
    assert c["duplicate_window_rate"] == pytest.approx(1 / 6)
    assert (c["rows_shorter_than_window"], c["row_tail_tokens_dropped"]) == (1, 4 + 5 + 3)
    assert (c["eos_tokens_in_sequences"], c["sequences_with_eos"]) == (0, 0)  # row 2's EOS sits in its dropped tail
    assert man["corpus"]["input"] == "token_ids" and man["corpus"]["token_field"] == "input_ids"
    assert man["corpus"]["row_length_histogram"] == {"5": 1, "16": 1, "19": 1, "20": 1}
    assert "text_field" not in man["corpus"]
    assert man["format"]["seq_len"] == SEQ and man["format"]["prepend_bos"] is False
    assert [s["n_sequences"] for s in man["shards"]] == [2, 2, 1]
    on_disk = json.loads((out / "corpus_manifest.json").read_text())
    assert on_disk["counts"] == c


def test_token_field_jsonl_matches_parquet(pc, tmp_path, monkeypatch):
    jl = _write_jsonl(tmp_path / "rows.jsonl", [{"input_ids": r, "meta": {"i": i}} for i, r in enumerate(ROWS)])
    man, out = _run(pc, tmp_path, [jl], "--token-field", "input_ids")
    assert _read_back(out, monkeypatch)[0] == KEPT
    assert _provenance(out, "offsets").tolist() == KEPT_OFFSETS


def test_token_field_jsonl_zst_two_frames(pc, tmp_path, monkeypatch):
    zstandard = pytest.importorskip("zstandard")
    zpath = tmp_path / "zst" / "00.jsonl.zst"
    zpath.parent.mkdir()
    lines = [json.dumps({"input_ids": r}) + "\n" for r in ROWS]
    cctx = zstandard.ZstdCompressor()
    with open(zpath, "wb") as fh:                                 # two concatenated frames
        fh.write(cctx.compress("".join(lines[:2]).encode()))
        fh.write(cctx.compress("".join(lines[2:]).encode()))
    man, out = _run(pc, tmp_path, [zpath.parent], "--token-field", "input_ids")
    assert _read_back(out, monkeypatch)[0] == KEPT
    assert man["counts"]["rows_read"] == 4


def test_token_field_budget_stops_exactly(pc, tmp_path, monkeypatch):
    src = _write_parquet(tmp_path / "a.parquet", ROWS)
    man, out = _run(pc, tmp_path, [src], "--token-field", "input_ids", "--max-sequences", "3")
    assert _read_back(out, monkeypatch)[0] == KEPT[:3]
    assert man["counts"]["sequences"] == 3 and man["counts"]["rows_read"] == 3   # row 3 never read


def test_token_field_no_dedupe_keeps_repeats(pc, tmp_path, monkeypatch):
    src = _write_parquet(tmp_path / "a.parquet", ROWS)
    man, out = _run(pc, tmp_path, [src], "--token-field", "input_ids", "--no-dedupe")
    seqs = _read_back(out, monkeypatch)[0]
    assert len(seqs) == 6 and seqs[2] == KEPT[0] and man["dedupe"] is False


def test_token_field_rejects_ids_outside_the_vocabulary(pc, tmp_path):
    src = _write_parquet(tmp_path / "a.parquet", [list(range(1, 9)), [1, 2, 3, 4, 5, 6, 7, VOCAB]])
    with pytest.raises(SystemExit, match="outside"):
        _run(pc, tmp_path, [src], "--token-field", "input_ids")


def test_token_field_rejects_text_only_options(pc, tmp_path):
    src = _write_parquet(tmp_path / "a.parquet", ROWS)
    with pytest.raises(SystemExit):
        _run(pc, tmp_path, [src], "--token-field", "input_ids", "--doc-mode", "wikitext")
    with pytest.raises(SystemExit):
        _run(pc, tmp_path, [src], "--token-field", "input_ids", "--no-doc-sep")


def test_hf_source_records_snapshot_revision(pc, tmp_path, monkeypatch):
    src = _write_parquet(tmp_path / "train-00000-of-00073.parquet", ROWS)
    seen = {}

    def fake_hf_files(repo, patterns, revision=None):
        seen.update(repo=repo, patterns=patterns, revision=revision)
        return [str(src)], "f02886b54795e8acabceb637ca119f9ae8f19d3f"
    monkeypatch.setattr(pc, "_hf_files", fake_hf_files)
    out = tmp_path / "out"
    man = pc.main(["--family", "gpt2-small", "--out", str(out), "--hf-dataset", "org/tok-set",
                   "--hf-pattern", "data/train-0000[0-6]-of-00073.parquet", "--hf-revision", "f02886b5",
                   "--token-field", "input_ids", "--seq-len", str(SEQ), "--max-sequences", "100"])
    assert seen == dict(repo="org/tok-set", patterns=["data/train-0000[0-6]-of-00073.parquet"], revision="f02886b5")
    s = man["corpus"]["source"]
    assert s["revision"] == "f02886b54795e8acabceb637ca119f9ae8f19d3f" and s["revision_requested"] == "f02886b5"
    assert man["counts"]["sequences"] == 5


# ----------------------------------------------------------------------------------------------- text path
DOCS = ["1 2 3 4 5", "6 7 8 9 10 11 12 13 14 15 16", "17 18", "20 21 22 23 24 25 26 27 28 29"]


def _expected_text_windows(docs):
    stream, owner = [], []
    for i, d in enumerate(docs):
        if i:
            stream.append(EOS)
            owner.append(i - 1)                                   # the separator belongs to the earlier carry
        ids = [int(w) for w in d.split()]
        stream += ids
        owner += [i] * len(ids)
    n = len(stream) // SEQ
    return [stream[j * SEQ:(j + 1) * SEQ] for j in range(n)], [owner[j * SEQ] for j in range(n)]


def test_text_path_packing_unchanged(pc, tmp_path, monkeypatch):
    src = _write_jsonl(tmp_path / "docs.jsonl", [{"text": d} for d in DOCS])
    man, out = _run(pc, tmp_path, [src])
    want, owners = _expected_text_windows(DOCS)
    seqs = _read_back(out, monkeypatch)[0]
    assert seqs == want
    assert _provenance(out, "docs").tolist() == owners
    assert not list((out / "provenance").glob("*.offsets.npy"))   # offsets are a token-field file only
    assert man["corpus"]["input"] == "text" and man["corpus"]["text_field"] == "text"
    assert man["format"]["doc_separator"] == "eos"
    assert man["counts"]["eos_tokens_in_sequences"] == sum(w.count(EOS) for w in want)


def test_pile_style_jsonl_zst_multiframe_text(pc, tmp_path, monkeypatch):
    zstandard = pytest.importorskip("zstandard")
    path = tmp_path / "00.jsonl.zst"
    recs = [json.dumps({"text": d, "meta": {"pile_set_name": "Pile-CC"}}) + "\n" for d in DOCS]
    cctx = zstandard.ZstdCompressor()
    buf = io.BytesIO()
    for r in recs:                                                # one frame per document
        buf.write(cctx.compress(r.encode()))
    path.write_bytes(buf.getvalue())
    man, out = _run(pc, tmp_path, [path])
    assert man["counts"]["documents_read"] == len(DOCS)          # every frame was read, not just the first
    assert _read_back(out, monkeypatch)[0] == _expected_text_windows(DOCS)[0]
