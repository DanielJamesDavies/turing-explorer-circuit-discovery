"""Shared loaders and maths for 064-prediction-circuits (DAN-134).

Everything here is CPU-friendly and reads the 062 explorer bundle read-only.

Conventions
  gid   = (layer * 3 + kind_idx) * 40960 + latent, 0-based, kinds attn, mlp, resid
  key   = research key "5.mlp.2277" (0-based)
  label = explorer label "L6 · MLP · 2278" (1-based layer and latent)

Final-norm treatment (the only non-linearity between a site and the logits on the direct path):
  norm_f(x) = g * x / rms(x),  rms(x) = sqrt(mean(x^2) + eps)
  Its Jacobian at the real final residual x (dimension n) applied to a direction d is
      J d = g * (d - x (x.d) / (n rms^2)) / rms
  Static per-target effects use a TYPICAL scale (J ~ g / rms_typ, rms_typ = median final-residual rms over
  corpus positions >= 1); inference-time DLA uses the exact J at each position.
  DLA is to the LOG-PROB of a token, first order: d logp(t) = a * (u_t - sum_j p_j u_j) . (J d),
  which is invariant to adding a constant to every logit (a raw-logit DLA is not).
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # config.yaml paths are relative to the repo root

BUNDLE = ROOT / "outputs" / "explorer_bundle" / "2026-09-27_062-full"
LOGIT_CTX = ROOT / "outputs" / "logit_ctx.pt"
RESULTS = HERE / "results"
KINDS = ("attn", "mlp", "resid")
N_LAT = 40960
N_LAYERS = 12
N_REAL_VOCAB = 32064     # Phi-3 tokenizer ids; the model's vocab is padded to 50,304
BOS = 1
SAE_WINDOW = 64          # SAEs / bundle statistics cover positions 0..63
RMS_TYP_FILE = RESULTS / "rms_typ.json"


# ----------------------------------------------------------------------------- identifiers
def gid_of(layer: int, kind: str, latent: int) -> int:
    return (layer * 3 + KINDS.index(kind)) * N_LAT + latent


def split_gid(gid: int) -> tuple[int, str, int]:
    comp, latent = divmod(int(gid), N_LAT)
    layer, k = divmod(comp, 3)
    return layer, KINDS[k], latent


def key_of(gid: int) -> str:
    l, k, i = split_gid(gid)
    return f"{l}.{k}.{i}"


def label_of(gid: int) -> str:
    l, k, i = split_gid(gid)
    return f"L{l + 1} · {k.upper()} · {i + 1}"


def full_label(gid: int) -> str:
    return f"{label_of(gid)}  ({key_of(gid)})"


def parse_target(s: str) -> int:
    """'5.mlp.2277' (research, 0-based) or 'L6.mlp.2278' / 'L6 MLP 2278' / 'L6 · MLP · 2278' (1-based) -> gid."""
    t = s.replace("·", " ").replace(".", " ").replace(",", " ").split()
    if len(t) != 3:
        raise ValueError(f"cannot parse target {s!r}")
    one_based = t[0].upper().startswith("L")
    layer = int(t[0][1:] if one_based else t[0])
    kind = t[1].lower()
    latent = int(t[2])
    if one_based:
        layer, latent = layer - 1, latent - 1
    if kind not in KINDS or not (0 <= layer < N_LAYERS) or not (0 <= latent < N_LAT):
        raise ValueError(f"bad target {s!r}")
    return gid_of(layer, kind, latent)


# ----------------------------------------------------------------------------- model + SAEs
def load_model():
    from model.inference import Inference
    inf = Inference(torch.device("cpu"), compile=False)
    return inf.model


def load_tokenizer():
    from model.tokenizer import Tokenizer
    return Tokenizer()


def load_bank():
    from sae.bank import SAEBank
    return SAEBank(device=torch.device("cpu"), load_decoders=True, compile=False)


def unembed(model):
    """(W_U [V, d] float32, g [d], eps)."""
    return (model.lm_head.weight.detach().float(), model.transformer.norm_f.scale.detach().float(),
            float(model.transformer.norm_f.eps))


def decoder_dirs(bank, gids) -> torch.Tensor:
    """Decoder columns (unit norm) of the given gids -> [n, d_model]."""
    gids = np.asarray(gids, dtype=np.int64)
    out = torch.empty(len(gids), 1024)
    by_site: dict = {}
    for j, g in enumerate(gids):
        l, k, i = split_gid(g)
        by_site.setdefault((l, k), []).append((j, i))
    with torch.no_grad():
        for (l, k), items in by_site.items():
            W = bank.saes[k][l].decoder.weight  # [d_model, d_sae]
            js = torch.tensor([j for j, _ in items])
            ii = torch.tensor([i for _, i in items])
            out[js] = W[:, ii].T.float()
    return out


@torch.no_grad()
def forward_capture(model, ids):
    """ids list[int] -> logits [T, V] f32, acts [L, 3, T, d] f32 (attn, mlp, resid as the SAEs were trained)."""
    from model.hooks import capture_activations
    x = torch.tensor([ids], dtype=torch.long)
    with capture_activations(model) as a:
        logits, _ = model(x, return_all_logits=True)
    acts = a.tensor[0]  # [L, K, T, N]
    return logits[0].float(), acts.float()


@torch.no_grad()
def encode_all(bank, acts):
    """acts [L, 3, T, d] -> vals [36, T, k], idx [36, T, k] (research encode; values sorted desc)."""
    L, K, T, _ = acts.shape
    vals = torch.empty(L * K, T, bank.k)
    idx = torch.empty(L * K, T, bank.k, dtype=torch.long)
    for l in range(L):
        for k, kind in enumerate(KINDS):
            v, i = bank.encode(acts[l, k], kind, l)
            v, o = v.float().sort(dim=-1, descending=True)
            vals[l * 3 + k], idx[l * 3 + k] = v, torch.gather(i, -1, o)
    return vals, idx


def active_at(vals, idx, t):
    """(gids [m], acts [m]) of latents active (> 0) at position t over all 36 components."""
    v = vals[:, t, :]
    keep = v > 0
    offs = (torch.arange(v.shape[0]) * N_LAT)[:, None]
    return (idx[:, t, :] + offs)[keep].numpy(), v[keep].numpy()


def rms_of(x, eps):
    return torch.sqrt(x.pow(2).mean(-1) + eps)


def logprob_dla(x_final, probs, W_U, g, eps, token, dirs, acts):
    """First-order effect on log p(token) at one position of each (direction, activation), exact norm Jacobian.

    x_final [d] final residual (pre norm_f); probs [V]; dirs [m, d]; acts [m] -> [m] (nats)."""
    n = x_final.shape[0]
    r = float(rms_of(x_final, eps))
    u = W_U[token] - probs @ W_U                           # [d]
    v = u * g / r
    proj = (v @ x_final) / (n * r * r)                     # scalar
    dd = dirs @ v - proj * (dirs @ x_final)               # [m]
    return torch.as_tensor(acts, dtype=torch.float32) * dd


def load_rms_typ() -> float:
    import json
    return float(json.loads(RMS_TYP_FILE.read_text())["rms_typ"])


# ----------------------------------------------------------------------------- bundle
def connect():
    return sqlite3.connect(f"file:{BUNDLE / 'explorer.sqlite'}?mode=ro", uri=True, check_same_thread=False)


def load_circuits(con):
    import pandas as pd
    df = pd.read_sql_query(
        "SELECT cid, key, seed_gid, layer, kind, latent, n_nodes, free0_tk, freeM_topk_tk, freeN_topk_tk, "
        "phi_sup_blind_tk, amp_any, near_threshold, pass FROM circuit ORDER BY cid", con)
    peaks = dict(con.execute("SELECT gid, MAX(peak) FROM target_ctx WHERE pool = 'strong' GROUP BY gid").fetchall())
    df["peak"] = [peaks.get(int(g)) or np.nan for g in df.seed_gid]
    if df.peak.isna().any():
        top_val = np.load(BUNDLE / "arrays" / "top_val.npy", mmap_mode="r")
        m = df.peak.isna()
        df.loc[m, "peak"] = [float(top_val[g, 0]) for g in df.seed_gid[m]]
    df["pass"] = df["pass"].astype(bool)
    return df


class MemberIndex:
    """Member support, as the explorer's CircuitIndex: sum |attr| of active members / sum |attr| of all members."""

    def __init__(self, con, circuits):
        rows = con.execute("SELECT cid, gid, attribution FROM member ORDER BY cid, gid").fetchall()
        arr = np.array(rows, dtype=np.float64)
        m_cid = arr[:, 0].astype(np.int64)
        self.gids = arr[:, 1].astype(np.int64)
        self.w = np.abs(np.nan_to_num(arr[:, 2]))
        cids = circuits.cid.to_numpy()
        pos = np.searchsorted(cids, m_cid)
        counts = np.bincount(pos, minlength=len(cids))
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.total = np.array([self.w[a:b].sum() for a, b in zip(self.offsets[:-1], self.offsets[1:])])
        self.scratch = np.zeros(36 * N_LAT, dtype=bool)

    def support(self, rows, active_gids):
        self.scratch[active_gids] = True
        try:
            out = np.zeros(len(rows))
            for j, r in enumerate(rows):
                a, b = self.offsets[r], self.offsets[r + 1]
                if self.total[r] > 0:
                    out[j] = self.w[a:b][self.scratch[self.gids[a:b]]].sum() / self.total[r]
        finally:
            self.scratch[active_gids] = False
        return np.clip(out, 0, 1)


class TargetTable:
    """gid -> circuit row lookup (one circuit per target in the primary run)."""

    def __init__(self, circuits):
        self.c = circuits
        self.row_of = {int(g): r for r, g in enumerate(circuits.seed_gid)}
        self.is_target = np.zeros(36 * N_LAT, dtype=bool)
        self.is_target[circuits.seed_gid.to_numpy()] = True


def fired_circuits(tt: TargetTable, mi: MemberIndex, gids, acts):
    """Rows of circuits whose target is active, with target activation, member support and explorer score."""
    f = tt.is_target[gids]
    tg, ta = gids[f], acts[f]
    rows = np.array([tt.row_of[int(g)] for g in tg], dtype=np.int64)
    sup = mi.support(rows, gids) if len(rows) else np.zeros(0)
    peak = tt.c.peak.to_numpy()[rows] if len(rows) else np.zeros(0)
    rel = np.where(peak > 0, ta / np.where(peak > 0, peak, 1), 0.0)
    return rows, tg, ta, sup, rel * sup


# ----------------------------------------------------------------------------- logit_ctx
def load_logit_ctx():
    d = torch.load(LOGIT_CTX, map_location="cpu", weights_only=False)
    return d["top_tokens"].numpy(), d["top_probs"].numpy(), d["latent_counts"].numpy()


def distinct_next_tokens(top_tokens, top_probs, gid, n=32):
    """Distinct empirical next tokens of a latent (by max prob), dropping empty slots."""
    comp, lat = divmod(int(gid), N_LAT)
    toks, probs = top_tokens[comp, lat], top_probs[comp, lat]
    seen, out = set(), []
    for t, p in zip(toks.tolist(), probs.tolist()):
        if p <= 0 or t in seen:
            continue
        seen.add(t)
        out.append((int(t), float(p)))
        if len(out) >= n:
            break
    return out


# ----------------------------------------------------------------------------- text
class Dec:
    def __init__(self, tok):
        self.tok = tok
        self.cache: dict = {}

    def __call__(self, t: int) -> str:
        s = self.cache.get(t)
        if s is None:
            # decode with a leading anchor so SentencePiece keeps the leading space
            s = self.tok.tokenizer.convert_ids_to_tokens(int(t)) if int(t) < N_REAL_VOCAB else f"<{t}>"
            s = (s or "").replace("▁", " ").replace("\n", "\\n")
            if s.startswith("<0x") and s.endswith(">"):
                try:
                    s = repr(bytes([int(s[3:-1], 16)]).decode("utf-8", "replace"))[1:-1]
                except ValueError:
                    pass
            self.cache[t] = s
        return s

    def text(self, ids) -> str:
        """Concatenated token pieces (keeps each token's leading space, unlike decode of a slice)."""
        return "".join(self(int(i)) for i in ids)
