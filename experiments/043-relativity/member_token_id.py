"""TOKEN-IDENTITY FILTER: split a circuit's members into token-machinery
vs everything else, by projecting each member's decoder direction onto
the unembedding (logit-lens style) and checking overlap with the seed's
own surface tokens.

This is a READING discipline, not a validity judgement: a member flagged
as token-machinery is still causally real; the flag says its content is
the seed's surface form rather than knowledge about the concept. Used
at selection/reporting time only -- nothing here touches the fit.

  MEMFILE=know_members.jsonl COMP=29 LAT=3736 ARM=know400 \
    STEM="relativ,Einstein" PYTHONPATH=src python .../member_token_id.py
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from hardware import detect_devices, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer
from sae.bank import SAEBank

HERE = Path(__file__).parent
COMP = int(os.environ["COMP"])
LAT = int(os.environ["LAT"])
ARM = os.environ.get("ARM", "know400")
MEMFILE = os.environ.get("MEMFILE", "know_members.jsonl")
STEMS = [x.strip() for x in os.environ.get("STEM", "").split(",") if x.strip()]
TOPT = int(os.environ.get("TOPT", 8))


def main():
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=True,
                   compile=should_compile())
    tok = Tokenizer()
    W_U = inference.model.lm_head.weight.detach()          # [vocab, d_model]

    stem_ids = set()
    for stem in STEMS:
        for form in (stem, " " + stem, stem.lower(), " " + stem.lower(),
                     stem.capitalize(), " " + stem.capitalize()):
            for i in tok.encode(form):
                if i > 2 and i != 29871:
                    stem_ids.add(i)
    print("stem token ids (%s): %s" % (",".join(STEMS), sorted(stem_ids)))

    rec = None
    for line in open(HERE / MEMFILE):
        r = json.loads(line)
        if (r["comp_idx"], r["latent"], r["arm"]) == (COMP, LAT, ARM):
            rec = r
    assert rec, "no %s members for %d/%d in %s" % (ARM, COMP, LAT, MEMFILE)

    rows = []
    for site, d in rec["alphas"].items():
        lyr, knd = site.split("/")
        lyr = int(lyr)
        sae = bank.saes[knd][lyr]
        W_dec = sae.decoder.weight.detach()                # [d_model, d_sae]
        for i, a in d.items():
            v = W_dec[:, int(i)].to(W_U.device, W_U.dtype)
            logits = W_U @ v
            top = logits.topk(TOPT)
            toks = [int(t) for t in top.indices.tolist()]
            hit = bool(stem_ids & set(toks))
            rows.append({
                "site": site, "latent": int(i), "alpha": float(a),
                "token_machinery": hit,
                "top_tokens": [tok.decode([t]) for t in toks]})
    n_tm = sum(r["token_machinery"] for r in rows)
    print("\nSEED c%d/%d arm %s: %d members | token-machinery %d (%.0f%%)"
          % (COMP, LAT, ARM, len(rows), n_tm, 100 * n_tm / max(len(rows), 1)))
    rows.sort(key=lambda r: -r["alpha"])
    print("\n%-9s %-7s %-6s %-4s %s"
          % ("site", "latent", "alpha", "tm?", "top unembedding tokens"))
    for r in rows[:30]:
        print("%-9s %-7d %-6.2f %-4s %s"
              % (r["site"], r["latent"], r["alpha"],
                 "TM" if r["token_machinery"] else "",
                 " | ".join(t.strip() or "_" for t in r["top_tokens"][:6])))
    out = HERE / ("token_id_c%d_%d_%s.jsonl" % (COMP, LAT, ARM))
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("\n-> %s" % out.name)


if __name__ == "__main__":
    main()
