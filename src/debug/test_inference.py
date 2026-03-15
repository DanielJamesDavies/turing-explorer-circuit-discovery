"""
Standalone inference test for TuringLLM.

Loads the first 4 sequences from the first data shard, runs 64 tokens of
autoregressive generation, and prints the decoded output.

Run from the repo root:
    python -m debug.test_inference
    # or: python src/debug/test_inference.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# Load configuration and model
# ---------------------------------------------------------------------------

from config import config
from model.inference import Inference
from model.tokenizer import Tokenizer
from hardware import detect_devices


def load_first_sequences(n: int = 4, context_len: int = 64) -> tuple[list[list[int]], int]:
    """
    Reads the first shard, splits on -1 separators, and returns the first n
    valid sequences (each truncated to context_len tokens).
    Mirrors the DataLoader's skip_first_token=True convention.
    """
    data_path = str(config.data.dataset_path)
    shard_files = sorted(
        [f for f in os.listdir(data_path) if f.endswith(".npy")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    if not shard_files:
        raise FileNotFoundError(f"No .npy shards found in {data_path}")

    shard_path = os.path.join(data_path, shard_files[0])
    print(f"Loading shard: {shard_path}")
    shard = np.load(shard_path)

    # Split on -1 separators (same logic as DataLoader)
    sep_positions = np.where(shard == -1)[0]
    seg_starts = np.concatenate([[0], sep_positions + 1])
    seg_ends   = np.concatenate([sep_positions, [len(shard)]])

    sequences: list[list[int]] = []
    for start, end in zip(seg_starts, seg_ends):
        seq = shard[start:end]
        seq = seq[seq != -1]   # strip any stray separators
        seq = seq[1:]           # skip_first_token=True (matches pipeline convention)
        if len(seq) > 1:
            sequences.append(seq[:context_len].tolist())
        if len(sequences) == n:
            break

    if not sequences:
        raise RuntimeError("No valid sequences found in the first shard")

    max_len = max(len(s) for s in sequences)
    return sequences, max_len


def pad_sequences(sequences: list[list[int]], pad_token: int) -> torch.Tensor:
    """Right-pads sequences to the same length and returns a LongTensor."""
    max_len = max(len(s) for s in sequences)
    padded = [s + [pad_token] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long)


def main():
    devices = detect_devices()
    device  = devices[0]

    print(f"Device: {device}")
    print()

    # -----------------------------------------------------------------------
    # Load tokenizer and model
    # -----------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = Tokenizer()
    pad_token = tokenizer.get_pad_token() or 0

    print("Loading TuringLLM...")
    model = Inference(device=device, compile=False)  # skip compile for quick test
    print()

    # -----------------------------------------------------------------------
    # Load sequences
    # -----------------------------------------------------------------------
    CONTEXT_LEN = 64
    N_GENERATE  = 64

    sequences, _ = load_first_sequences(n=4, context_len=CONTEXT_LEN)
    print(f"Loaded {len(sequences)} sequences (context up to {CONTEXT_LEN} tokens each)")
    print()

    tokens = pad_sequences(sequences, pad_token).to(device)
    context_len = tokens.shape[1]

    # -----------------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------------
    print(f"Generating {N_GENERATE} tokens per sequence...")
    print("=" * 72)

    with torch.no_grad():
        output_tokens, _, _ = model.forward(
            tokens,
            num_gen=N_GENERATE,
            tokenize_final=True,
            activations_callback=None,
            return_activations=False,
        )

    # output_tokens: [B, context_len + N_GENERATE]
    generated = output_tokens[:, context_len:]  # strip the input context

    # -----------------------------------------------------------------------
    # Decode and print
    # -----------------------------------------------------------------------
    for i, (ctx_ids, gen_ids) in enumerate(zip(sequences, generated.tolist())):
        context_text   = tokenizer.decode(ctx_ids)
        generated_text = tokenizer.decode(gen_ids)

        print(f"\n── Sequence {i + 1} ──")
        print(f"[CONTEXT]   {context_text!r}")
        print(f"[GENERATED] {generated_text!r}")
        print()

    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
