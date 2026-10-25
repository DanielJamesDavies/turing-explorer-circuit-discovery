"""Worst concrete cases from the full audit, decoded (085): out/audit_all.csv + out/examples_all.jsonl -> out/worst_examples.md.

  PYTHONPATH=src .venv/bin/python experiments/085-heldout-audit/examples.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import audit  # noqa: E402
from peek import tok  # noqa: E402


def rows_of(rec, g):
    """Global row index (audit numbering: strongest, then mid, then contrast) -> (pool, tokens, anchor, peak, id)."""
    s, m = rec["strong"], rec.get("mid")
    nS, nM = int(s["pos"].shape[0]), (int(m["pos"].shape[0]) if m is not None else 0)
    if g < nS:
        return "strong", s["pos"][g], int(s["arg"][g]), float(s["peak"][g]), s["ids"][g], g
    if g < nS + nM:
        j = g - nS
        return "mid", m["pos"][j], int(m["arg"][j]), float(m["peak"][j]), m["ids"][j], j
    j = g - nS - nM
    return "contrast", rec["neg"][j], None, None, (rec["neg_ids"] or [None] * 99)[j], j


def main():
    T = tok()
    d = pd.read_csv(HERE / "out" / "audit_all.csv")
    d = d[d["skip"].isna()]
    ex = pd.DataFrame([json.loads(l) for l in open(HERE / "out" / "examples_all.jsonl")])
    L = ["# 085 worst examples (full audit)\n"]
    picks = []
    full = d[d.n_S_ho == 16]
    picks.append(("16 held-out strongest rows, none survives R2 (most exact hits)",
                  full[full.Sho_clean_R2 == 0].sort_values("Sho_exact_TRAIN", ascending=False).head(4).key.tolist()))
    picks.append(("most exact duplicates of FIT rows in the held-out strongest set",
                  d.sort_values("Sho_exact_FIT", ascending=False).head(3).key.tolist()))
    picks.append(("position-0 latents (every strongest anchor at 0: the scored input is ONE token)",
                  d[(d.anchor0_S_ho == d.n_S_ho) & (d.anchor0_S_tr == d.n_S_tr)].head(3).key.tolist()))
    picks.append(("anchor-prefix duplicates that are NOT exact (later tokens differ, the read is identical)",
                  d.assign(x=d.Sho_apfx_TRAIN - d.Sho_exact_TRAIN).sort_values("x", ascending=False).head(3).key.tolist()))
    picks.append(("near-duplicates that are neither exact nor anchor-prefix (shifted boilerplate)",
                  d.assign(x=d.Sho_clean_R1 - d.Sho_clean_R2).sort_values("x", ascending=False).head(3).key.tolist()))
    picks.append(("strongest-pool rows where the target is silent on re-forward (peak <= 0, anchor defaults to 0)",
                  d.sort_values("peak_le0_S", ascending=False).head(3).key.tolist()))
    for title, keys in picks:
        L.append("## %s\n" % title)
        for k in keys:
            r = d[d.key == k].iloc[0]
            L.append("- **%s** (arm %s): held-out strongest %d; exact dup of FIT %d, of training side %d; anchor-prefix %d; "
                     "near %d; R0/R1/R2/R3 clean %d/%d/%d/%d; anchors at 0 in held %d; peak<=0 rows %d" % (
                         k, r.arm, r.n_S_ho, r.Sho_exact_FIT, r.Sho_exact_TRAIN, r.Sho_apfx_TRAIN, r.Sho_near_TRAIN,
                         r.Sho_clean_R0, r.Sho_clean_R1, r.Sho_clean_R2, r.Sho_clean_R3, r.anchor0_S_ho, r.peak_le0_S))
        k = keys[0] if keys else None
        if k is None:
            continue
        rec = torch.load(audit.CTX / ("%s.pt" % k), weights_only=False)[k]
        e = ex[(ex.key == k) & (ex.set == "Sho")].sort_values(["exact", "apfx", "lcr"], ascending=False)
        if "non-exact" in title or "NOT exact" in title:
            e = e[~e.exact]
        if "neither" in title:
            e = e[~e.exact & ~e.apfx]
        for _, x in e.head(2).iterrows():
            a, b = rows_of(rec, int(x.row)), rows_of(rec, int(x.other))
            L.append("\n  `%s` held-out %s row %d (id %s, anchor %s, peak %.3g) vs %s row %d (id %s, roles %s): exact %s, "
                     "anchor-prefix %s, LCR %d, 8-gram containment %.2f" % (
                         k, a[0], a[5], a[4], a[2], a[3], b[0], b[5], b[4], "/".join(x.other_role), x.exact, x.apfx,
                         x.lcr, x.c8))
            for lab, t in (("held ", a), ("other", b)):
                txt = T.decode(t[1].tolist()).replace("\n", " ")
                L.append("    - %s: %s" % (lab, txt[:220]))
        if "position-0" in title:
            s = rec["strong"]
            firsts = sorted(set(T.decode([int(t)]) for t in s["pos"][:, 0].tolist()))
            L.append("\n  first tokens across the 64 strongest: %s" % firsts[:12])
        L.append("")
    (HERE / "out" / "worst_examples.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
