"""WHICH CIRCUITS ARE WORTH FITTING ON GEMMA-2-2B? A competence probe over
candidate tasks, each as a contrastive next-token decision (target vs a
plausible wrong answer), so the winners can go straight into the 048
tri-amp fitter (which needs exactly that pair).

Per task: n prompts kept (both answers single-token), argmax accuracy,
mean logit diff target-contrast, fraction with a positive margin, and
mean p(target). A task is FITTABLE when accuracy is high and the margin
is comfortably positive — those are the behaviours the model actually
has, and only those can have a circuit.

  N=60 python experiments/050-gemma-circuits/probe_candidates.py
"""
import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL_ID = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b")
N = int(os.environ.get("N", 60))
rng = random.Random(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(dev).eval()


def enc(text):
    return tok(text, return_tensors="pt")["input_ids"][0].tolist()


def tok_after(prefix, word):
    """Single next token id for `word` after `prefix`, or None if it splits."""
    a, b = enc(prefix), enc(prefix + word)
    return b[len(a)] if (b[:len(a)] == a and len(b) == len(a) + 1) else None


def last_logits(ids):
    with torch.no_grad():
        return model(torch.tensor([ids], device=dev)).logits[0, -1].float()


# ------------------------------------------------------------------ tasks
CAPITALS = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
            ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"),
            ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
            ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Denmark", "Copenhagen"), ("Ireland", "Dublin"),
            ("Cuba", "Havana"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("India", "Delhi")]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
VERBS = [("walk", "walked"), ("play", "played"), ("watch", "watched"), ("call", "called"),
         ("open", "opened"), ("close", "closed"), ("paint", "painted"), ("visit", "visited"),
         ("clean", "cleaned"), ("cook", "cooked"), ("start", "started"), ("finish", "finished")]
NOUNS = [("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("car", "cars"), ("tree", "trees"),
         ("house", "houses"), ("bird", "birds"), ("chair", "chairs"), ("river", "rivers")]
OPPOSITES = [("hot", "cold"), ("big", "small"), ("fast", "slow"), ("light", "dark"),
             ("happy", "sad"), ("rich", "poor"), ("young", "old"), ("hard", "soft"),
             ("open", "closed"), ("full", "empty"), ("high", "low"), ("wet", "dry")]
FEM = ["Mary", "Alice", "Sarah", "Emma", "Laura", "Anna", "Kate", "Lucy", "Amy", "Julia"]
MASC = ["John", "Tom", "James", "Peter", "Paul", "David", "Mark", "Steve", "Jack", "Robert"]
COMMON = ["book", "chair", "river", "window", "garden", "letter", "picture", "market",
          "doctor", "engine", "forest", "island", "candle", "basket", "monkey", "pencil"]


def t_capital():
    for country, city in rng.sample(CAPITALS, min(N, len(CAPITALS))):
        other = rng.choice([c for _, c in CAPITALS if c != city])
        yield "Q: What is the capital of %s?\nA: The capital of %s is" % (country, country), " " + city, " " + other


def t_country():
    for country, city in rng.sample(CAPITALS, min(N, len(CAPITALS))):
        other = rng.choice([c for c, _ in CAPITALS if c != country])
        yield "%s is the capital city of" % city, " " + country, " " + other


def t_month():
    for _ in range(N):
        i = rng.randrange(len(MONTHS) - 3)
        yield ", ".join(MONTHS[i:i + 3]) + ",", " " + MONTHS[i + 3], " " + MONTHS[i]


def t_day():
    for _ in range(N):
        i = rng.randrange(len(DAYS) - 3)
        yield ", ".join(DAYS[i:i + 3]) + ",", " " + DAYS[i + 3], " " + DAYS[i]


def t_number():
    for _ in range(N):
        i = rng.randint(1, 40)
        yield "%d, %d, %d, %d," % (i, i + 1, i + 2, i + 3), " %d" % (i + 4), " %d" % (i + 3)


def t_add():
    """single-digit sums with a single-token answer; contrast = off-by-one"""
    seen = set()
    for _ in range(N * 4):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        yield "%d + %d =" % (a, b), " %d" % (a + b), " %d" % (a + b + 1)


def t_add_words():
    seen = set()
    for _ in range(N * 4):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        yield "Q: What is %d plus %d?\nA:" % (a, b), " %d" % (a + b), " %d" % (a + b + 1)


def t_past():
    for _ in range(N):
        v, p = rng.choice(VERBS)
        yield "Today I %s. Yesterday I" % v, " " + p, " " + v


def t_plural():
    for _ in range(N):
        s, p = rng.choice(NOUNS)
        yield "I have one %s. You have two" % s, " " + p, " " + s


def t_opposite():
    for _ in range(N):
        a, b = rng.choice(OPPOSITES)
        yield "The opposite of %s is" % a, " " + b, " " + a


def t_pronoun():
    for _ in range(N):
        fem = rng.random() < 0.5
        name = rng.choice(FEM if fem else MASC)
        yield "%s went to the shop because" % name, " she" if fem else " he", " he" if fem else " she"


def t_induction():
    """repeated token sequence: the second copy must predict the continuation"""
    for _ in range(N):
        seq = rng.sample(COMMON, 6)
        text = " ".join(seq) + " " + " ".join(seq[:3])
        yield text, " " + seq[3], " " + rng.choice([w for w in COMMON if w not in seq])


def t_acronym():
    orgs = [("National Aeronautics and Space Administration", "NASA"),
            ("World Health Organization", "WHO"), ("United Nations", "UN"),
            ("Federal Bureau of Investigation", "FBI"), ("European Union", "EU"),
            ("International Monetary Fund", "IMF"), ("World Trade Organization", "WTO"),
            ("Central Intelligence Agency", "CIA"), ("National Basketball Association", "NBA"),
            ("British Broadcasting Corporation", "BBC")]
    for full, ac in orgs:
        yield "The %s (" % full, ac[0], rng.choice([c for c in "XYZQK"])


def t_bracket():
    for _ in range(N):
        v = rng.choice(["x", "n", "k", "value", "total"])
        yield "def f(%s):\n    return (%s + 1" % (v, v), ")", ":"


def t_greater():
    for _ in range(N):
        c = rng.choice(["16", "17", "18", "19"]); yy = rng.randint(11, 88)
        tens = yy // 10
        yield ("The war lasted from the year %s%02d to the year %s" % (c, yy, c),
               "%d" % min(tens + 1, 9), "%d" % max(tens - 1, 0))


TASKS = {"capital": t_capital, "country_of_capital": t_country, "month_succ": t_month,
         "day_succ": t_day, "number_succ": t_number, "addition_digits": t_add,
         "addition_words": t_add_words, "past_tense": t_past, "plural": t_plural,
         "opposite": t_opposite, "pronoun": t_pronoun, "induction": t_induction,
         "acronym": t_acronym, "code_bracket": t_bracket, "greater_than": t_greater}

report = {}
print("task                  n   acc   margin   pos%%  p(target)  example")
for name, gen in TASKS.items():
    rows, split = [], 0
    for prompt, target, contrast in gen():
        if len(rows) >= N:
            break
        ti, tc = tok_after(prompt, target), tok_after(prompt, contrast)
        if ti is None or tc is None or ti == tc:
            split += 1
            continue
        rows.append((prompt, ti, tc, target, contrast))
    if not rows:
        print("%-20s  --  (no single-token prompts; %d split)" % (name, split)); continue
    acc = margin = pos = ptar = 0.0
    for prompt, ti, tc, _, _ in rows:
        lg = last_logits(enc(prompt))
        d = float(lg[ti] - lg[tc])
        acc += int(lg.argmax() == ti); margin += d; pos += int(d > 0)
        ptar += float(torch.softmax(lg, -1)[ti])
    m = len(rows)
    report[name] = dict(n=m, acc=acc / m, margin=margin / m, pos=pos / m, p_target=ptar / m,
                        example=rows[0][0][-60:], target=rows[0][3], contrast=rows[0][4])
    print("%-20s %3d  %.2f  %+6.2f   %.2f     %.3f    %r -> %r vs %r"
          % (name, m, acc / m, margin / m, pos / m, ptar / m, rows[0][0][-45:].replace("\n", "\\n"),
             rows[0][3], rows[0][4]), flush=True)

json.dump(report, open(HERE / "candidate_competence.json", "w"), indent=1)
best = sorted([k for k, v in report.items() if v["acc"] >= 0.7 and v["margin"] > 1.0],
              key=lambda k: -report[k]["margin"])
print("\nFITTABLE (acc >= 0.70 and margin > 1.0), best first: %s" % ", ".join(best))
print("->", HERE / "candidate_competence.json")
