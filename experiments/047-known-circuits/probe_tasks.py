"""TASK-COMPETENCE PROBE for well-known circuits on TuringLLM, unpadded,
one prompt per forward: which classic behaviours does the model have?

  induction   : random 24-token sequence repeated; accuracy of argmax on
                the second copy (positions 25..47 predicting the token
                that followed the same token in the first copy)
  greater-than: "The {noun} lasted from the year {c}{yy} to the year {c}"
                -> P(next digit > tens(yy)) - P(next digit < tens(yy))
                over the digit tokens (single-digit tokenisation)
  agreement   : "The {subj} {pp}" -> logit(correct verb) - logit(wrong)
                for singular/plural subjects with a distractor noun of
                the opposite number inside the PP (SFC's within-PP task)

  PYTHONPATH=src python experiments/047-known-circuits/probe_tasks.py
"""
import random
import sys

import torch

sys.path.insert(0, "src")
from hardware import detect_devices, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer

rng = random.Random(0)
device = detect_devices()[0]
inference = Inference(device=device, compile=should_compile())
tok = Tokenizer()
inference.disable_compile()


def last_logits(seq):
    """(T, V) logits for every position (all_logits forward)."""
    with torch.no_grad():
        out = inference.forward(torch.tensor([seq], dtype=torch.long,
                                             device=device),
                                all_logits=True, grad_enabled=False,
                                return_activations=False, tokenize_final=False)
    lg = out[1] if isinstance(out, (tuple, list)) else out
    return lg[0].float()


def tok_after(prefix, word):
    a, b = tok.encode(prefix), tok.encode(prefix + word)
    return b[len(a)] if (b[:len(a)] == a and len(b) == len(a) + 1) else None


# ---------------- induction ----------------
vocab_pool = [t for t in range(2000, 12000)]
acc_n = acc_hit = 0
for _ in range(32):
    seq = rng.sample(vocab_pool, 24)
    full = seq + seq
    lg = last_logits(full)                     # (T, V)
    for pos in range(24, 47):                  # predict full[pos+1]
        acc_n += 1
        acc_hit += int(int(lg[pos].argmax()) == full[pos + 1])
print("induction: argmax accuracy on repeated copy %d/%d (%.0f%%)"
      % (acc_hit, acc_n, 100 * acc_hit / acc_n))

# ---------------- greater-than ----------------
digits = {d: tok_after("The war lasted from the year 17", str(d)) for d in range(10)}
digit_ids = [digits[d] for d in range(10)]
assert all(x is not None for x in digit_ids), digits
nouns = ["war", "empire", "dynasty", "reign", "siege", "famine", "festival",
         "project", "expedition", "strike"]
gt_scores, gt_n = [], 0
for _ in range(120):
    c = rng.choice(["16", "17", "18", "19"])
    yy = rng.randint(11, 88)
    noun = rng.choice(nouns)
    prompt = "The %s lasted from the year %s%02d to the year %s" % (noun, c, yy, c)
    lg = last_logits(tok.encode(prompt))[-1]
    p = torch.softmax(lg, -1)[digit_ids]
    tens = yy // 10
    gt = float(p[tens + 1:].sum()); lt = float(p[:tens].sum())
    gt_scores.append(gt - lt); gt_n += 1
    if _ < 2:
        print("   e.g. %r -> P(d>%d)=%.3f P(d<%d)=%.3f" % (prompt, tens, gt, tens, lt))
print("greater-than: mean [P(next digit > tens) - P(< tens)] = %.3f | positive on %d/%d"
      % (sum(gt_scores) / gt_n, sum(1 for x in gt_scores if x > 0), gt_n))

# ---------------- subject-verb agreement (within-PP distractor) ----------------
pairs = [("key", "keys"), ("book", "books"), ("car", "cars"), ("teacher", "teachers"),
         ("dog", "dogs"), ("report", "reports"), ("student", "students"),
         ("plan", "plans"), ("river", "rivers"), ("idea", "ideas")]
pps = ["on the {n}", "near the {n}", "behind the {n}", "of the {n}", "beside the {n}"]
verbs = [("is", "are"), ("was", "were"), ("has", "have")]
ag_scores, ag_correct = [], 0
for _ in range(160):
    subj_sg, subj_pl = rng.choice(pairs)
    dist_sg, dist_pl = rng.choice(pairs)
    plural = rng.random() < 0.5
    subj = subj_pl if plural else subj_sg
    dist = dist_sg if plural else dist_pl        # opposite number distractor
    prompt = "The %s %s" % (subj, rng.choice(pps).format(n=dist))
    v_sg, v_pl = rng.choice(verbs)
    correct, wrong = (v_pl, v_sg) if plural else (v_sg, v_pl)
    tc, tw = tok_after(prompt, " " + correct), tok_after(prompt, " " + wrong)
    if tc is None or tw is None:
        continue
    lg = last_logits(tok.encode(prompt))[-1]
    ld = float(lg[tc] - lg[tw])
    ag_scores.append(ld); ag_correct += int(ld > 0)
    if len(ag_scores) <= 2:
        print("   e.g. %r -> %s vs %s: ld %.2f" % (prompt, correct, wrong, ld))
print("agreement: mean logit diff (correct - wrong) %.3f | correct on %d/%d (%.0f%%)"
      % (sum(ag_scores) / len(ag_scores), ag_correct, len(ag_scores),
         100 * ag_correct / len(ag_scores)))
