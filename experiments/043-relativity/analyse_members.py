"""OPEN member analysis for a fitted circuit: no expected members, no
label shortcuts. For each member (all of them, or top-N by alpha):

  * its own top_ctx sequences rendered as FULL ACTIVATION PROFILES
    ([[tok=v]] >= 50% of peak, [tok=v] >= 15%) -- the firing shape is
    data: a single hot token reads as a token detector, a ramp across a
    clause as a context accumulator, and secondary firing tokens are
    often the interpretive key;
  * a profile-shape statistic: peakiness = mean over its contexts of
    (peak activation / sum of activations in the sequence). ~1.0 =
    one-token detector, small = distributed accumulator;
  * its echo correlation with the seed (from the same streaming stats
    the penalty uses), so the analysis can ask "what did the echo
    penalty keep, and what does that look like".

Writes a markdown report; classification happens by READING the
report, after discovery, not by matching a hoped-for member list.

  MEMFILE=know_members.jsonl COMP=29 LAT=3736 ARM=know400 TOPM=25 \
    PYTHONPATH=src python .../analyse_members.py
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer
from pipeline.component_index import component_idx, split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/"
                "Runs/20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
COMP = int(os.environ["COMP"])
LAT = int(os.environ["LAT"])
ARM = os.environ.get("ARM", "know400")
MEMFILE = os.environ.get("MEMFILE", "know_members.jsonl")
TOPM = int(os.environ.get("TOPM", 25))
TOPK = int(os.environ.get("TOPK", 3))
NSEQ = int(os.environ.get("NSEQ", 24))
BATCH = 16


def render_profile(tok, seq, prof):
    peak = float(max(float(prof.max()), 1e-9))
    parts = []
    for i, t in enumerate(seq):
        if t < 0:
            continue
        w = tok.decode([int(t)])
        v = float(prof[i]) if i < prof.shape[0] else 0.0
        if v >= 0.5 * peak:
            parts.append("[[%s=%.0f]]" % (w, v))
        elif v >= 0.15 * peak:
            parts.append("[%s=%.0f]" % (w, v))
        else:
            parts.append(w)
    return "".join(parts).replace(chr(10), " ")


def main():
    rec = None
    for line in open(HERE / MEMFILE):
        r = json.loads(line)
        if (r["comp_idx"], r["latent"], r["arm"]) == (COMP, LAT, ARM):
            rec = r
    assert rec, "no %s members for %d/%d in %s" % (ARM, COMP, LAT, MEMFILE)

    load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
    from store.context import top_ctx
    tok = Tokenizer()
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=False,
                   compile=should_compile())
    n_kinds = len(bank.kinds)
    loader = DataLoader(device=device, pin_memory=is_fast_memory())

    members = []
    for site, d in rec["alphas"].items():
        lyr, knd = site.split("/")
        for i, a in d.items():
            members.append((float(a), int(lyr), knd, int(i)))
    members.sort(reverse=True)
    shown = members[:TOPM]

    targets = [(component_idx(l, bank.kinds.index(k), n_kinds), i)
               for _a, l, k, i in shown]
    own_ids, all_ids = {}, set()
    for c, lat in targets:
        ids = [int(x) for x in top_ctx.ctx_seq_idx[c, lat].tolist()
               if x > 0][:NSEQ]
        own_ids[(c, lat)] = set(ids)
        all_ids |= set(ids)
    all_ids = sorted(all_ids)
    seqs, order = {}, []
    for b_ids, b_toks in loader.get_batches_by_ids(all_ids):
        ids_l = b_ids.tolist() if torch.is_tensor(b_ids) else list(b_ids)
        for sid, row in zip(ids_l, b_toks.tolist()):
            seqs[int(sid)] = row
            order.append(int(sid))

    want = {}
    for c, lat in targets:
        want.setdefault(c, []).append(lat)
    best = {}
    for s0 in range(0, len(order), BATCH):
        chunk = order[s0:s0 + BATCH]
        toks = torch.tensor([[max(t, 0) for t in seqs[s]] for s in chunk],
                            dtype=torch.long, device=device)

        def cb(layer_idx, activations):
            with torch.no_grad():
                for ki, knd in enumerate(bank.kinds):
                    c = component_idx(layer_idx, ki, n_kinds)
                    if c not in want:
                        continue
                    ta, ti = bank.encode(activations[ki], knd, layer_idx)
                    for lat in want[c]:
                        hit = (ti == lat)
                        if not bool(hit.any()):
                            continue
                        vals = torch.where(hit, ta.float(),
                                           torch.zeros_like(ta.float())
                                           ).amax(-1)
                        for b, sid in enumerate(chunk):
                            if sid not in own_ids[(c, lat)]:
                                continue
                            v = float(vals[b].max())
                            if v > 0:
                                best.setdefault((c, lat), []).append(
                                    (v, sid, vals[b].cpu()))
        inference.forward(toks, num_gen=1, tokenize_final=False,
                          activations_callback=cb, return_activations=False)

    out = ["# Member analysis: c%d/%d, arm %s (n=%d, top %d by alpha)"
           % (COMP, LAT, ARM, len(members), len(shown)), "",
           "Profiles over each member's OWN top contexts. peakiness ~1 =",
           "one-token detector; small = distributed context accumulator.",
           "No member was sought; classify by reading.", ""]
    for a, lyr, knd, i in shown:
        c = component_idx(lyr, bank.kinds.index(knd), n_kinds)
        hits = sorted(best.get((c, i), []), key=lambda t: -t[0])[:TOPK]
        pk = None
        if hits:
            pks = [float(h[0]) / max(float(h[2].sum()), 1e-9) for h in hits]
            pk = sum(pks) / len(pks)
        out.append("## alpha %.2f | L%d %s latent %d | peakiness %s"
                   % (a, lyr, knd, i,
                      ("%.2f" % pk) if pk is not None else "n/a"))
        for v, sid, prof in hits:
            out.append("  peak %.1f | %s"
                       % (v, render_profile(tok, seqs[sid], prof)[:400]))
        if not hits:
            out.append("  (silent in its own top contexts)")
        out.append("")
    rpt = HERE / ("members_c%d_%d_%s.md" % (COMP, LAT, ARM))
    rpt.write_text("\n".join(out), encoding="utf-8", newline="")
    print("\n".join(out[:80]))
    print("\n-> %s" % rpt.name)


if __name__ == "__main__":
    main()
