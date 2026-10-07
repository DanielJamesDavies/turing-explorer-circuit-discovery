"""Compact text report of one circuit, built from the explorer bundle's reading data (no GPU).

This is the input the describer sends to Claude. It carries what the 063 case studies found decisive for reading a
circuit (dev-notes/2026-09-28-reading-circuits-for-explorer.md), trimmed to fit a small prompt:
  - the target: scores, peak tokens + consistency, distinct windows with the peak marked, one contrast snippet
    (where the target is silent), and its direct logit effect;
  - the top members by contribution share at the target's anchor, hubs dropped, each with its specificity
    (strongest activation on the target's contexts vs the contrast contexts), own peak tokens, logits and one context.

Sources: <bundle>/reading.sqlite (converter stage 8) + explorer.sqlite (circuit scores) + tokens/tokens.npy.
Keys are research keys (0-based layer.kind.latent).

    python experiments/065-circuit-describer/report.py 10.resid.17597 [--members 15] [--stats]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

BUNDLE = ROOT / "outputs" / "explorer_bundle" / "2026-09-27_062-full"
RESULTS = HERE / "results"
KINDS = ("attn", "mlp", "resid")
N_LAT = 40960
N_REAL_VOCAB = 32064
DISCRIMINATIVE_RATIO = 3.0      # pos_max / neg_max at or above this: the member mostly switches off on the contrast
RATIO_EPS = 1e-3


# ----------------------------------------------------------------------------- identifiers
def gid_of(key: str) -> int:
    layer, kind, latent = key.split(".")
    return (int(layer) * 3 + KINDS.index(kind)) * N_LAT + int(latent)


def key_of(gid: int) -> str:
    comp, latent = divmod(int(gid), N_LAT)
    layer, k = divmod(comp, 3)
    return f"{layer}.{KINDS[k]}.{latent}"


# ----------------------------------------------------------------------------- data access
class Bundle:
    def __init__(self, bundle: Path = BUNDLE):
        self.reading = sqlite3.connect(f"file:{bundle / 'reading.sqlite'}?mode=ro", uri=True)
        self.explorer = sqlite3.connect(f"file:{bundle / 'explorer.sqlite'}?mode=ro", uri=True)
        self.tokens = np.load(bundle / "tokens" / "tokens.npy", mmap_mode="r")
        self._tok = None
        self._piece: dict[int, str] = {}

    # ---- text
    def piece(self, t: int) -> str:
        """One token as text, keeping its leading space."""
        s = self._piece.get(t)
        if s is None:
            if self._tok is None:
                cwd = os.getcwd()
                os.chdir(ROOT)  # the tokenizer's config paths are relative to the repo root
                try:
                    from model.tokenizer import Tokenizer
                    self._tok = Tokenizer()
                finally:
                    os.chdir(cwd)
            s = self._tok.tokenizer.convert_ids_to_tokens(int(t)) if int(t) < N_REAL_VOCAB else f"<{t}>"
            s = (s or "").replace("▁", " ").replace("\n", "↵")
            if s.startswith("<0x") and s.endswith(">"):
                try:
                    s = bytes([int(s[3:-1], 16)]).decode("utf-8", "replace").replace("\n", "↵")
                except ValueError:
                    pass
            self._piece[t] = s
        return s

    def window(self, seq_id: int, peak_pos: int, before: int, after: int) -> str:
        """Text around a peak, the peak token in [[ ]]; BOS shown as ¶ (a document start)."""
        row = self.tokens[int(seq_id) - 1]
        lo, hi = max(0, peak_pos - before), min(len(row), peak_pos + after + 1)
        out = []
        for i in range(lo, hi):
            t = int(row[i])
            s = "¶" if t == 1 else ("" if t == 0 else self.piece(t))
            out.append(f"[[{s}]]" if i == peak_pos else s)
        return ("…" if lo > 0 else "") + "".join(out) + ("…" if hi < len(row) else "")

    def span(self, seq_id: int, lo: int, hi: int) -> str:
        row = self.tokens[int(seq_id) - 1]
        return "…" + "".join("¶" if int(t) == 1 else self.piece(int(t)) for t in row[lo:hi] if int(t) != 0) + "…"

    def tokens_list(self, pairs, n: int) -> str:
        return " ".join(repr(self.piece(int(t))) for t, _ in pairs[:n])

    # ---- rows
    def circuit(self, key: str):
        gid = gid_of(key)
        row = self.explorer.execute(
            "SELECT cid, pass, n, free0_tk, freeM_topk_tk, freeN_topk_tk, phi_sup_blind_tk, amp_any, near_threshold "
            "FROM circuit WHERE seed_gid = ?", (gid,)).fetchone()
        if row is None:
            raise KeyError(f"no circuit for {key}")
        names = ("cid", "pass", "n", "Z", "A", "C", "necessity", "amplifier", "near_threshold")
        return gid, dict(zip(names, row))

    def latent(self, gid: int) -> dict | None:
        r = self.reading.execute(
            "SELECT n_ctx, top_token, consistency, peaks, ctx, logit_up, logit_down FROM reading_latent WHERE gid = ?",
            (gid,)).fetchone()
        if r is None:
            return None
        return {"n_ctx": r[0], "top_token": r[1], "consistency": r[2], "peaks": json.loads(r[3] or "[]"),
                "ctx": json.loads(r[4] or "[]"), "up": json.loads(r[5] or "[]"), "down": json.loads(r[6] or "[]")}

    def is_hub(self, gid: int) -> bool:
        r = self.reading.execute("SELECT is_hub FROM reading_hub WHERE gid = ?", (gid,)).fetchone()
        return bool(r and r[0])


# ----------------------------------------------------------------------------- report
def merged_consistency(b: Bundle, peaks, total: int) -> float:
    """Share of peaks on the most common token once case, a plural 's' and prefix variants are merged
    (' festival' / ' Festival' / ' festiv'; ' Gen' / ' gen'). Used to decide the one_token trigger."""
    groups: list[list] = []   # [root, count]
    for t, c in sorted(peaks, key=lambda tc: len(b.piece(int(tc[0])).strip())):
        k = b.piece(int(t)).strip().lower()
        if len(k) > 3 and k.endswith("s"):
            k = k[:-1]
        for g in groups:
            root = g[0]
            if k == root or (min(len(k), len(root)) >= 4 and (k.startswith(root) or root.startswith(k))):
                g[1] += c
                break
        else:
            groups.append([k, c])
    return max((c for _, c in groups), default=0) / total if total else 0.0


def peaks_text(b: Bundle, peaks, total: int, n: int = 4) -> str:
    return ", ".join(f"{b.piece(int(t))!r}×{c}" for t, c in peaks[:n]) + (f" (of {total})" if total else "")


def distinct_windows(b: Bundle, refs, n: int, before: int, after: int) -> list[str]:
    """Up to n windows [(seq_id, pos)] with distinct text and distinct two tokens before the peak, so the list varies."""
    out, seen_text, seen_prev = [], set(), set()
    for s, p in refs:
        text = b.window(s, p, before, after)
        prev = tuple(int(t) for t in b.tokens[s - 1][max(0, p - 2):p])
        if text in seen_text or prev in seen_prev:
            continue
        seen_text.add(text)
        seen_prev.add(prev)
        out.append(text)
        if len(out) == n:
            break
    return out


def build_report(b: Bundle, key: str, n_members: int = 15, n_windows: int = 5, n_logits: int = 6,
                 member_logits: int = 4, n_mid: int = 2, n_contrast: int = 4) -> tuple[str, dict]:
    """-> (report text, stats). Stats say what was included, for sizing the prompt."""
    gid, c = b.circuit(key)
    rc = b.reading.execute("SELECT n_members, top25_share FROM reading_circuit WHERE cid = ?", (c["cid"],)).fetchone()
    if rc is None:
        raise KeyError(f"{key}: no reading data")
    tgt = b.reading.execute("SELECT n_strong, consistency, peaks, windows, contrast FROM reading_target WHERE gid = ?",
                            (gid,)).fetchone()
    t_lat = b.latent(gid)
    n_strong, consistency, peaks, windows, contrast = tgt[0], tgt[1], json.loads(tgt[2]), json.loads(tgt[3]), json.loads(tgt[4] or "null")
    merged = max(consistency, merged_consistency(b, peaks, n_strong))

    flags = ["pass" if c["pass"] else "fail"]
    if c["amplifier"]:
        flags.append("also lifts related latents")
    if c["near_threshold"]:
        flags.append("near TopK threshold")
    # Failing circuits can lack a read (e.g. a vacuous metric or no member share): show "n/a" instead of failing.
    num = lambda v, spec: "n/a" if v is None else format(v, spec)
    lines = [
        f"CIRCUIT {key} ({', '.join(flags)}; {rc[0]} members; top 25 carry {num(rc[1], '.0%')}; "
        f"faithfulness Z {num(c['Z'], '.2f')} A {num(c['A'], '.2f')} C {num(c['C'], '.2f')})",
        "",
        "TARGET",
        f"peak tokens: {peaks_text(b, peaks, n_strong)}; consistency {consistency:.0%}"
        + (f" ({merged:.0%} with case / plural / prefix variants merged)" if merged > consistency + 0.005 else "")
        + ("; trigger: one_token" if merged >= 0.8 else ""),
        "windows (peak in [[ ]], ¶ = document start):",
    ]
    lines += [f"- {b.window(s, p, 12, 3)}" for s, p, _ in windows[:n_windows]]

    # Mid-band contexts: where the target fires only moderately, i.e. the edges of what it responds to.
    mid = b.explorer.execute("SELECT seq_id, arg, peak FROM target_ctx WHERE gid = ? AND pool = 'mid' AND arg IS NOT NULL "
                             "ORDER BY rank", (gid,)).fetchall()
    top_peak = windows[0][2] if windows else None
    mid_lines = distinct_windows(b, [(s, a) for s, a, _ in mid], n_mid, 12, 3)
    if mid_lines:
        rel = f" (about {mid[0][2] / top_peak:.0%} of the strongest)" if top_peak else ""
        lines.append(f"weaker contexts{rel}:")
        lines += [f"- {w}" for w in mid_lines]

    # Contrast contexts (the target was verified silent on the whole context). Where they contain the target's own
    # peak token, show that token: the same token, but the target stays silent (a natural minimal pair).
    top_token = peaks[0][0] if peaks else None
    neg = [s for (s,) in b.explorer.execute("SELECT seq_id FROM target_ctx WHERE gid = ? AND pool = 'neg' ORDER BY rank",
                                            (gid,))]
    same_token = []
    for s in neg:
        hits = np.nonzero(np.asarray(b.tokens[s - 1]) == top_token)[0] if top_token is not None else []
        if len(hits):
            same_token.append((s, int(hits[0])))
    # Tightest pairs first: the same token with the same left neighbour as in the strong windows (e.g. 's after an
    # apostrophe), where the target still stays silent.
    strong_prev = {int(b.tokens[s - 1][p - 1]) for s, p, _ in windows if p > 0}
    same_token.sort(key=lambda sp: sp[1] == 0 or int(b.tokens[sp[0] - 1][sp[1] - 1]) not in strong_prev)
    silent_lines = distinct_windows(b, same_token, n_contrast, 12, 3)
    if silent_lines:
        lines.append(f"contrast, target SILENT on its own peak token {b.piece(int(top_token))!r} here "
                     f"({len(same_token)} of {len(neg)} contrast contexts contain it):")
        lines += [f"- {w}" for w in silent_lines]
    elif contrast:
        lines.append(f"contrast, target silent: {b.span(contrast[0], contrast[1], contrast[2])}")
    if t_lat:
        lines.append(f"logits + {b.tokens_list(t_lat['up'], n_logits)} | − {b.tokens_list(t_lat['down'], n_logits)}")

    # Members by share, hubs dropped. ratio = specificity: >= 3 discriminative (switches off on the contrast).
    rows = b.reading.execute(
        "SELECT rank, gid, alpha, fire, pos_max, neg_max, share FROM reading_member WHERE cid = ? ORDER BY rank",
        (c["cid"],)).fetchall()
    hubs = [r for r in rows if b.is_hub(r[1])]
    kept = [r for r in rows if not b.is_hub(r[1])]
    disc_share = sum(r[6] for r in kept if r[4] / max(r[5], RATIO_EPS) >= DISCRIMINATIVE_RATIO)
    lines += [
        "",
        f"MEMBERS: top {min(n_members, len(kept))} of {len(kept)} by share at the target's peak token "
        f"({len(hubs)} generic hubs dropped, {sum(r[6] for r in hubs):.0%} of share; "
        f"discriminative members carry {disc_share:.0%}).",
        "Columns: share | alpha | specificity = strongest act on target contexts / on contrast (>=3 = switches off "
        "on contrast) | own peak tokens | own logits + | one own context",
    ]
    for rank, mgid, alpha, fire, pos, neg, share in kept[:n_members]:
        lat = b.latent(mgid) or {"peaks": [], "n_ctx": 0, "up": [], "ctx": []}
        ratio = pos / max(neg, RATIO_EPS)
        spec = ">100" if ratio > 100 else f"{ratio:.1f}"
        ctx = lat["ctx"][0] if lat["ctx"] else None
        own = b.window(ctx[0], ctx[1], 8, 2) if ctx else "-"
        lines.append(
            f"{key_of(mgid)} | {share:.1%} | a{alpha:.2f} | spec {spec} | "
            f"{peaks_text(b, lat['peaks'], 0, 3)} | + {b.tokens_list(lat['up'], member_logits)} | {own}")
    text = "\n".join(lines)
    stats = {"chars": len(text), "members_shown": min(n_members, len(kept)), "hubs": len(hubs),
             "merged_consistency": merged}
    return text, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="+", help="research keys, e.g. 10.resid.17597")
    ap.add_argument("--members", type=int, default=15)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--stats", action="store_true", help="print size stats after each report")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    b = Bundle()
    for key in a.keys:
        text, stats = build_report(b, key, n_members=a.members, n_windows=a.windows)
        print(text)
        if a.stats:
            print(f"\n[{stats}]")
        print()


if __name__ == "__main__":
    main()
