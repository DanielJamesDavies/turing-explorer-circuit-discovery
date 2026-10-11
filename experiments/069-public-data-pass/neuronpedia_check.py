"""Neuronpedia as an independent check on the GPT-2 pipeline (DAN-37), offline: cached records only.

Reads the Neuronpedia feature records 053 cached (~/neuronpedia_cache, or --np-cache; nothing is fetched) and, per
latent:
  1. re-encodes Neuronpedia's own activating contexts through our GPT-2 adapter + bank (the site and dictionary the
     pipeline mines with) and compares the per-token activations with Neuronpedia's stored values: if the pipeline's
     site / normalisation / encode were wrong, these would disagree. Scored twice: as displayed (no BOS, the SAEs'
     training convention and ours) and with a BOS prepended (in case Neuronpedia's forward pass used one);
  2. counts Neuronpedia's duplicate contexts (same token strings), the issue's "3-33 % duplicates";
  3. when --out holds our stores, how many contexts we mined for the same latent (Neuronpedia tops out at ~45).
     A peak-vs-peak comparison of mined contexts is only meaningful on a corpus like Neuronpedia's (OpenWebText),
     so it belongs to the real run, not the 069 smoke corpus.

  PYTHONPATH=src .venv/bin/python experiments/069-public-data-pass/neuronpedia_check.py \
      --data data_public/gpt2-small --out outputs_public/gpt2-small --skip-missing-saes
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

_KIND = {"att": "attn", "res_mid": "resid_mid", "mlp": "mlp", "res_post": "resid"}
_NAME = re.compile(r"gpt2-small_(\d+)-(att|res_mid|mlp|res_post)_32k-oai_(\d+)\.json$")


def to_ids(tok, tokens):
    """Neuronpedia display strings -> GPT-2 ids (053/neuronpedia.py: space -> 'Ġ', newline -> 'Ċ')."""
    out = []
    for t in tokens:
        v = t.replace(" ", "Ġ").replace("\n", "Ċ").replace("\t", "ĉ")
        i = tok.convert_tokens_to_ids(v)
        if i is None or i == tok.unk_token_id:
            ids = tok(t, add_special_tokens=False).input_ids
            i = ids[0] if ids else tok.eos_token_id
        out.append(int(i))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Neuronpedia cross-check (cached records only)")
    ap.add_argument("--data", required=True, help="GPT-2 shard directory (for configure / our stores)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--np-cache", default=os.path.expanduser("~/neuronpedia_cache"))
    ap.add_argument("--skip-missing-saes", action="store_true")
    args = ap.parse_args(argv)
    common.configure("gpt2-small", args.data, args.out, skip_missing_saes=args.skip_missing_saes)
    import torch
    from transformers import AutoTokenizer
    from model.inference import Inference
    from sae.bank import SAEBank
    from sae.dense import target_latent_activations

    files = sorted(f for f in glob.glob(os.path.join(args.np_cache, "*.json")) if _NAME.search(f))
    if not files:
        raise SystemExit("no cached Neuronpedia GPT-2 32k records under %s" % args.np_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("gpt2")
    inference = Inference(device=device)
    bank = SAEBank(device=device)
    have_store = os.path.exists(os.path.join(args.out, "top_ctx.pt"))
    if have_store:
        from store.context import top_ctx
        top_ctx.load(os.path.join(args.out, "top_ctx.pt"))

    rows = []
    for f in files:
        layer, src, idx = _NAME.search(f).groups()
        layer, idx, kind = int(layer), int(idx), _KIND[src]
        if not bank.has_site(kind, layer):
            rows.append(dict(target="%d.%s.%d" % (layer, kind, idx), skipped="missing SAE"))
            continue
        rec = json.load(open(f))
        acts = [a for a in rec.get("activations") or [] if a.get("tokens") and a.get("values")]
        if not acts:
            rows.append(dict(target="%d.%s.%d" % (layer, kind, idx), skipped="no activating contexts in record"))
            continue
        toks = [tuple(a["tokens"]) for a in acts]
        n_dup = len(toks) - len(set(toks))
        kind_idx = list(bank.kinds).index(kind)
        row = dict(target="%d.%s.%d" % (layer, kind, idx), np_contexts=len(acts), np_duplicates=n_dup,
                   np_duplicate_rate=round(n_dup / max(1, len(acts)), 4), np_frac_nonzero=rec.get("frac_nonzero"),
                   np_context_len=sorted({len(a["tokens"]) for a in acts}))
        # Neuronpedia's display tokens may omit a BOS its own forward pass used: score both conventions.
        for variant, bos in (("no_bos", False), ("bos", True)):
            diffs, peaks_np, peaks_ours = [], [], []
            for a in acts:
                body = to_ids(tok, a["tokens"])
                ids = torch.tensor([([tok.bos_token_id] if bos else []) + body], device=device)
                got = []

                def hook(layer_idx, activations):
                    if layer_idx == layer:
                        ta, ti = bank.encode(activations[kind_idx], kind, layer_idx)
                        got.append(target_latent_activations(ta, ti, idx).float()[0].cpu())

                with torch.no_grad():
                    inference.forward(ids, activations_callback=hook, return_activations=False,
                                      tokenize_final=False)
                ours = got[0][1:] if bos else got[0]
                theirs = torch.tensor(a["values"], dtype=torch.float32)
                n_cmp = min(int(ours.shape[0]), int(theirs.shape[0]))
                diffs.append(float((ours[:n_cmp] - theirs[:n_cmp]).abs().max()))
                peaks_np.append(float(theirs[:n_cmp].max()))
                peaks_ours.append(float(ours[:n_cmp].max()))
            pn, po = torch.tensor(peaks_np), torch.tensor(peaks_ours)
            row[variant] = dict(max_abs_diff=round(max(diffs), 4),
                                peak_rel_err_median=round(float(((po - pn).abs() / pn.clamp(min=1e-6)).median()), 4))
        row["np_peak_max"] = round(float(pn.max()), 4)
        if have_store:
            comp = layer * len(bank.kinds) + kind_idx
            row["our_stored_contexts"] = int((top_ctx.ctx_seq_idx[comp, idx] > 0).sum())
        rows.append(row)
        print(row, flush=True)

    ok = [r for r in rows if "no_bos" in r]
    report = dict(np_cache=args.np_cache, n_records=len(rows), n_checked=len(ok),
                  max_abs_diff_no_bos=max(r["no_bos"]["max_abs_diff"] for r in ok) if ok else None,
                  max_abs_diff_bos=max(r["bos"]["max_abs_diff"] for r in ok) if ok else None,
                  peak_rel_err_median_no_bos=sorted(r["no_bos"]["peak_rel_err_median"] for r in ok)[len(ok) // 2]
                  if ok else None,
                  peak_rel_err_median_bos=sorted(r["bos"]["peak_rel_err_median"] for r in ok)[len(ok) // 2]
                  if ok else None,
                  np_duplicate_rate_mean=round(sum(r["np_duplicate_rate"] for r in ok) / max(1, len(ok)), 4),
                  np_duplicate_rate_range=[min(r["np_duplicate_rate"] for r in ok), max(r["np_duplicate_rate"] for r in ok)]
                  if ok else None, rows=rows)
    path = os.path.join(args.out, "neuronpedia_check.json")
    common.write_json(path, report)
    print({k: v for k, v in report.items() if k != "rows"})
    print("report:", path)


if __name__ == "__main__":
    main()
