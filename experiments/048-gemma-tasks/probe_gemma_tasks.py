"""Task-competence probe on Gemma-2-2B (unsloth mirror), same four
classic behaviours as 047/probe_tasks.py on TuringLLM, unpadded:
IOI (ABBA/BABA), induction (repeated natural sentence), greater-than,
subject-verb agreement with a PP distractor.

  PYTHONPATH=src python experiments/048-gemma-tasks/probe_gemma_tasks.py
"""
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b")
rng = random.Random(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(dev).eval()


def enc(text):
    return tok(text, return_tensors="pt")["input_ids"][0].tolist()   # includes BOS


def logits_all(ids):
    with torch.no_grad():
        return model(torch.tensor([ids], device=dev)).logits[0].float()


def tok_after(prefix, word):
    a, b = enc(prefix), enc(prefix + word)
    return b[len(a)] if (b[:len(a)] == a and len(b) == len(a) + 1) else None


# ---------------- IOI ----------------
NAMES = ["Mary", "John", "Tom", "James", "Dan", "Martin", "Amy", "Joseph",
         "Jim", "Peter", "Paul", "Bob", "Alice", "Sarah", "Emma", "David",
         "Lucy", "Anna", "Kate", "Mark", "Steve", "Laura", "Jack", "Sam"]
PLACES = ["store", "school", "park", "office", "hospital", "garden", "station"]
OBJECTS = ["drink", "book", "ring", "bone", "necklace", "snack", "letter"]
TEMPLATES = ["Then, {X} and {Y} went to the {place}. {S} gave a {obj} to",
             "When {X} and {Y} got a {obj} at the {place}, {S} decided to give it to",
             "After {X} and {Y} went to the {place}, {S} gave a {obj} to"]
lds, correct, n = [], 0, 0
while n < 120:
    io, s = rng.sample(NAMES, 2)
    abba = rng.random() < 0.5
    x, y = (io, s) if abba else (s, io)
    prompt = rng.choice(TEMPLATES).format(X=x, Y=y, S=s, place=rng.choice(PLACES),
                                          obj=rng.choice(OBJECTS))
    ti, ts = tok_after(prompt, " " + io), tok_after(prompt, " " + s)
    if ti is None or ts is None:
        continue
    lg = logits_all(enc(prompt))[-1]
    ld = float(lg[ti] - lg[ts]); lds.append(ld)
    correct += int(int(lg.argmax()) == ti); n += 1
print("IOI: mean logit diff IO-S %.2f | logit_diff>0 %d/%d | argmax==IO %d/%d (%.0f%%)"
      % (sum(lds) / n, sum(1 for x in lds if x > 0), n, correct, n, 100 * correct / n))

# ---------------- induction ----------------
sents = ["The committee reviewed the proposal carefully before voting on the final budget for the coming year.",
         "Photosynthesis converts light energy into chemical energy stored in glucose molecules within plant cells.",
         "Early navigators used the positions of stars to estimate latitude while crossing open ocean.",
         "The museum acquired several paintings from a private collector in Vienna last spring."]
tot = h1 = h2 = 0
for s_ in sents:
    t = enc(s_)[1:]          # drop BOS, then repeat
    full = [enc("")[0]] + t + t
    L = len(t); lg = logits_all(full)
    h1 += sum(int(int(lg[p].argmax()) == full[p + 1]) for p in range(1, L))
    h2 += sum(int(int(lg[p].argmax()) == full[p + 1]) for p in range(L + 1, 2 * L))
    tot += L - 1
print("induction: 1st copy %.0f%% -> 2nd copy %.0f%%" % (100 * h1 / tot, 100 * h2 / tot))

# ---------------- greater-than ----------------
gt, gn = [], 0
for _ in range(80):
    c = rng.choice(["16", "17", "18", "19"]); yy = rng.randint(11, 88)
    prompt = "The %s lasted from the year %s%02d to the year %s" % (
        rng.choice(["war", "empire", "dynasty", "siege", "festival"]), c, yy, c)
    ids = [tok_after(prompt, str(d)) for d in range(10)]
    if any(i is None for i in ids):
        continue
    p = torch.softmax(logits_all(enc(prompt))[-1], -1)[ids]
    tens = yy // 10
    gt.append(float(p[tens + 1:].sum() - p[:tens].sum())); gn += 1
print("greater-than: mean P(>tens)-P(<tens) %.3f | positive %d/%d" % (sum(gt) / gn, sum(1 for x in gt if x > 0), gn))

# ---------------- agreement ----------------
pairs = [("key", "keys"), ("book", "books"), ("car", "cars"), ("teacher", "teachers"),
         ("dog", "dogs"), ("report", "reports"), ("student", "students"), ("plan", "plans")]
pps = ["on the {n}", "near the {n}", "behind the {n}", "of the {n}"]
verbs = [("is", "are"), ("was", "were"), ("has", "have")]
ag, ac = [], 0
for _ in range(120):
    ss, sp = rng.choice(pairs); ds, dp = rng.choice(pairs)
    plural = rng.random() < 0.5
    prompt = "The %s %s" % (sp if plural else ss, rng.choice(pps).format(n=ds if plural else dp))
    vs, vp = rng.choice(verbs)
    cor, wr = (vp, vs) if plural else (vs, vp)
    tc, tw = tok_after(prompt, " " + cor), tok_after(prompt, " " + wr)
    if tc is None or tw is None:
        continue
    lg = logits_all(enc(prompt))[-1]
    ld = float(lg[tc] - lg[tw]); ag.append(ld); ac += int(ld > 0)
print("agreement: mean logit diff %.2f | correct %d/%d (%.0f%%)" % (sum(ag) / len(ag), ac, len(ag), 100 * ac / len(ag)))
