"""Task datasets for the two classic circuits TuringLLM demonstrably has
(probe_tasks.py, 2026-09-06), in the behaviour_clusters.pt schema with
RIGHT-padded windows and per-row anchors (left-padding corrupts the
model; the runner reads each prompt at its own last token):

  gt_clusters.pt     greater-than (Hanna et al. 2023): "The {noun} lasted
                     from the year {c}{yy} to the year {c}" -> next digit.
                     target = the model's own argmax digit, kept only when
                     it is > tens(yy) (the behaviour the circuit explains).
  agree_clusters.pt  subject-verb agreement with an opposite-number
                     distractor in a PP (SFC's within-PP task): target =
                     the correct verb token, kept when logit diff > 0.

  N=256 PYTHONPATH=src python experiments/047-known-circuits/make_tasks.py
"""
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from hardware import detect_devices, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer

HERE = Path(__file__).parent
SEQ = 64
N = int(os.environ.get("N", 256))
rng = random.Random(0)
device = detect_devices()[0]
inference = Inference(device=device, compile=should_compile())
tok = Tokenizer()
inference.disable_compile()


def last_logits(seq):
    with torch.no_grad():
        out = inference.forward(torch.tensor([seq], dtype=torch.long,
                                             device=device), all_logits=True,
                                grad_enabled=False, return_activations=False,
                                tokenize_final=False)
    lg = out[1] if isinstance(out, (tuple, list)) else out
    return lg[0, -1].float()


def tok_after(prefix, word):
    a, b = tok.encode(prefix), tok.encode(prefix + word)
    return b[len(a)] if (b[:len(a)] == a and len(b) == len(a) + 1) else None


def save(name, rows, extra):
    wins = [r["toks"] + [0] * (SEQ - len(r["toks"])) for r in rows]
    torch.save({"windows": wins, "targets": [r["target"] for r in rows],
                "anchors": [len(r["toks"]) - 1 for r in rows],
                "probs": [r["p"] for r in rows],
                "prompts": [r["prompt"] for r in rows],
                "assign": torch.zeros(len(rows), dtype=torch.long),
                "coherence": torch.ones(1), "sizes": torch.tensor([len(rows)]),
                "anchor": SEQ - 2, **extra}, HERE / name)
    print("-> %s (%d prompts)" % (name, len(rows)))


# ---------------- greater-than ----------------
digit_ids = [tok_after("The war lasted from the year 17", str(d)) for d in range(10)]
nouns = ["war", "empire", "dynasty", "reign", "siege", "famine", "festival",
         "project", "expedition", "strike", "drought", "occupation"]
gt_rows, gt_seen = [], set()
while len(gt_rows) < N:
    c = rng.choice(["16", "17", "18", "19"])
    yy = rng.randint(11, 88)
    prompt = "The %s lasted from the year %s%02d to the year %s" % (
        rng.choice(nouns), c, yy, c)
    if prompt in gt_seen:
        continue
    gt_seen.add(prompt)
    t = tok.encode(prompt)
    lg = last_logits(t)
    p = torch.softmax(lg, -1)
    pd = p[digit_ids]
    d = int(pd.argmax())
    tens = yy // 10
    if d <= tens or len(t) > SEQ - 1:
        continue
    gt_rows.append({"prompt": prompt, "toks": t, "target": digit_ids[d],
                    "p": float(pd[d]), "tens": tens,
                    "gt_mass": float(pd[tens + 1:].sum())})
print("greater-than kept %d | mean p(argmax digit) %.3f | mean P(>tens) %.3f"
      % (len(gt_rows), sum(r["p"] for r in gt_rows) / len(gt_rows),
         sum(r["gt_mass"] for r in gt_rows) / len(gt_rows)))
save("gt_clusters.pt", gt_rows,
     {"tens": [r["tens"] for r in gt_rows], "digit_ids": digit_ids})

# ---------------- agreement ----------------
pairs = [("key", "keys"), ("book", "books"), ("car", "cars"), ("teacher", "teachers"),
         ("dog", "dogs"), ("report", "reports"), ("student", "students"),
         ("plan", "plans"), ("river", "rivers"), ("idea", "ideas"),
         ("bridge", "bridges"), ("letter", "letters"), ("farmer", "farmers")]
pps = ["on the {n}", "near the {n}", "behind the {n}", "of the {n}",
       "beside the {n}", "from the {n}", "under the {n}"]
verbs = [("is", "are"), ("was", "were"), ("has", "have")]
ag_rows, ag_seen = [], set()
tries = 0
while len(ag_rows) < N and tries < 20 * N:
    tries += 1
    subj_sg, subj_pl = rng.choice(pairs)
    dist_sg, dist_pl = rng.choice(pairs)
    if subj_sg == dist_sg:
        continue
    plural = rng.random() < 0.5
    subj = subj_pl if plural else subj_sg
    dist = dist_sg if plural else dist_pl
    prompt = "The %s %s" % (subj, rng.choice(pps).format(n=dist))
    v_sg, v_pl = rng.choice(verbs)
    key = (prompt, v_sg)
    if key in ag_seen:
        continue
    ag_seen.add(key)
    correct, wrong = (v_pl, v_sg) if plural else (v_sg, v_pl)
    tc, tw = tok_after(prompt, " " + correct), tok_after(prompt, " " + wrong)
    if tc is None or tw is None:
        continue
    t = tok.encode(prompt)
    lg = last_logits(t)
    ld = float(lg[tc] - lg[tw])
    if ld <= 0:
        continue
    ag_rows.append({"prompt": prompt, "toks": t, "target": tc, "wrong": tw,
                    "p": float(torch.softmax(lg, -1)[tc]), "ld": ld,
                    "plural": plural})
print("agreement kept %d | mean p(correct verb) %.3f | mean logit diff %.2f"
      % (len(ag_rows), sum(r["p"] for r in ag_rows) / len(ag_rows),
         sum(r["ld"] for r in ag_rows) / len(ag_rows)))
save("agree_clusters.pt", ag_rows,
     {"wrong_tokens": [r["wrong"] for r in ag_rows],
      "plural": [r["plural"] for r in ag_rows]})
