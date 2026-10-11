"""Shared helpers for 070: cached wikitext windows (smoke-test text only).

Windows are built the way 033 built them: wikitext-103 train parquet, non-empty
lines tokenised without special tokens, concatenated, cut into SEQ-token
windows (no BOS: both public SAE families were trained with prepend_bos off).
"""
from __future__ import annotations

import torch


def wikitext_windows(tokenizer, n: int, seq_len: int = 64, skip: int = 0) -> torch.Tensor:
    """[n, seq_len] token windows; ``skip`` windows are dropped first."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                           filename="wikitext-103-raw-v1/train-00000-of-00002.parquet")
    texts = pq.read_table(path, columns=["text"])["text"].to_pylist()
    buf, out = [], []
    for t in texts:
        if not t.strip():
            continue
        buf.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
        while len(buf) >= seq_len:
            out.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(out) >= n + skip:
                return torch.tensor(out[skip:], dtype=torch.long)
    return torch.tensor(out[skip:], dtype=torch.long)


def load_tokenizer(family_name: str):
    from transformers import AutoTokenizer
    from model.families import FAMILIES
    return AutoTokenizer.from_pretrained(FAMILIES[family_name].tokenizer_id)
