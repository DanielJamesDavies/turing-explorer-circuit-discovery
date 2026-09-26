"""DATASETS for the Gemma-2-2B behaviours that probe_candidates.py showed
the model actually has. Writes <task>_rows.json in this folder, in the
shape 048/fit_gemma_task.py consumes via ROWS=<path>:
  [{prompt, target(token id), contrast(token id), meta{...}}]

Every row is validated: both answers must be a single next token after
the prompt, and they must differ. Number tasks put the space INSIDE the
prompt ("3 + 5 = ") because Gemma tokenises digits separately, which is
why the first probe rejected every arithmetic row.

  N=200 python experiments/050-gemma-circuits/make_datasets.py
"""
import json
import os
import random
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).parent
MODEL_ID = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b")
N = int(os.environ.get("N", 200))
rng = random.Random(0)
tok = AutoTokenizer.from_pretrained(MODEL_ID)


def enc(t):
    return tok(t, return_tensors="pt")["input_ids"][0].tolist()


def tok_after(prefix, word):
    a, b = enc(prefix), enc(prefix + word)
    return b[len(a)] if (b[:len(a)] == a and len(b) == len(a) + 1) else None


CAPITALS = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
            ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"),
            ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
            ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Denmark", "Copenhagen"), ("Ireland", "Dublin"),
            ("Cuba", "Havana"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("India", "Delhi"),
            ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Chile", "Santiago"),
            ("Hungary", "Budapest"), ("Finland", "Helsinki"), ("Belgium", "Brussels"), ("Ukraine", "Kyiv"),
            ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Morocco", "Rabat"), ("Ghana", "Accra")]
CAP_TPL = ["Q: What is the capital of {c}?\nA: The capital of {c} is",
           "The capital city of {c} is",
           "{c}'s capital city is called",
           "When people visit the capital of {c}, they arrive in"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FEM = ["Mary", "Alice", "Sarah", "Emma", "Laura", "Anna", "Kate", "Lucy", "Amy", "Julia",
       "Helen", "Clara", "Nina", "Rosa", "Ellen"]
MASC = ["John", "Tom", "James", "Peter", "Paul", "David", "Mark", "Steve", "Jack", "Robert",
        "Henry", "Frank", "Oscar", "Victor", "Simon"]
PRON_TPL = ["{name} went to the shop because", "After the meeting, {name} said that",
            "{name} finished the work quickly because", "When the phone rang, {name} answered because"]
OPPOSITES = [("hot", "cold"), ("big", "small"), ("fast", "slow"), ("light", "dark"),
             ("happy", "sad"), ("rich", "poor"), ("young", "old"), ("hard", "soft"),
             ("open", "closed"), ("full", "empty"), ("high", "low"), ("wet", "dry"),
             ("strong", "weak"), ("early", "late"), ("clean", "dirty"), ("loud", "quiet"),
             ("near", "far"), ("heavy", "light"), ("sharp", "blunt"), ("smooth", "rough")]


def add(rows, prompt, target_s, contrast_s, meta, seen):
    if prompt in seen:
        return
    ti, tc = tok_after(prompt, target_s), tok_after(prompt, contrast_s)
    if ti is None or tc is None or ti == tc:
        return
    seen.add(prompt)
    rows.append({"prompt": prompt, "target": ti, "contrast": tc, "meta": meta})


def capital():
    rows, seen = [], set()
    for _ in range(N * 6):
        if len(rows) >= N:
            break
        country, city = rng.choice(CAPITALS)
        other = rng.choice([c for _, c in CAPITALS if c != city])
        add(rows, rng.choice(CAP_TPL).format(c=country), " " + city, " " + other,
            {"country": country, "city": city, "wrong_city": other}, seen)
    return rows


def country_of_capital():
    rows, seen = [], set()
    tpl = ["{city} is the capital city of", "The city of {city} is the capital of",
           "Travellers landing in {city} have arrived in"]
    for _ in range(N * 6):
        if len(rows) >= N:
            break
        country, city = rng.choice(CAPITALS)
        other = rng.choice([c for c, _ in CAPITALS if c != country])
        add(rows, rng.choice(tpl).format(city=city), " " + country, " " + other,
            {"country": country, "city": city}, seen)
    return rows


def _succ(items, name):
    rows, seen = [], set()
    for _ in range(N * 8):
        if len(rows) >= N:
            break
        k = rng.choice([2, 3, 4])
        i = rng.randrange(len(items) - k)
        prompt = ", ".join(items[i:i + k]) + ","
        add(rows, prompt, " " + items[i + k], " " + items[i],
            {"kind": name, "prev": items[i + k - 1], "next": items[i + k], "run": k}, seen)
    return rows


def month_succ():
    return _succ(MONTHS, "month")


def day_succ():
    return _succ(DAYS, "day")


def number_succ():
    """digits are separate tokens in Gemma: keep the space in the prompt"""
    rows, seen = [], set()
    for _ in range(N * 8):
        if len(rows) >= N:
            break
        i = rng.randint(1, 40)
        prompt = "%d, %d, %d, %d, " % (i, i + 1, i + 2, i + 3)
        add(rows, prompt, str(i + 4), str(i + 3), {"start": i, "next": i + 4}, seen)
    return rows


def addition():
    """single-digit sums, answer 2-9 (single digit) or the tens digit of 10-18"""
    rows, seen = [], set()
    for a in range(1, 10):
        for b in range(1, 10):
            s = a + b
            for tpl in ("%d + %d = ", "Q: What is %d plus %d?\nA: "):
                if len(rows) >= N:
                    break
                prompt = tpl % (a, b)
                wrong = s + 1 if s < 18 else s - 1
                add(rows, prompt, str(s)[0], str(wrong)[0],
                    {"a": a, "b": b, "sum": s, "carry": s >= 10}, seen)
    return rows


def pronoun():
    rows, seen = [], set()
    for _ in range(N * 6):
        if len(rows) >= N:
            break
        fem = rng.random() < 0.5
        name = rng.choice(FEM if fem else MASC)
        add(rows, rng.choice(PRON_TPL).format(name=name), " she" if fem else " he",
            " he" if fem else " she", {"name": name, "female": fem}, seen)
    return rows


def opposite():
    rows, seen = [], set()
    tpl = ["The opposite of {a} is", "Q: What is the opposite of {a}?\nA: The opposite of {a} is",
           "Not {a}, but the very opposite:"]
    for _ in range(N * 8):
        if len(rows) >= N:
            break
        a, b = rng.choice(OPPOSITES)
        if rng.random() < 0.5:
            a, b = b, a
        add(rows, rng.choice(tpl).format(a=a), " " + b, " " + a, {"word": a, "opposite": b}, seen)
    return rows


_WORDS = ["book", "chair", "river", "window", "garden", "letter", "picture", "market",
          "doctor", "engine", "forest", "island", "candle", "basket", "monkey", "pencil",
          "bottle", "carpet", "hammer", "lantern", "meadow", "napkin", "orchard", "table",
          "bridge", "castle", "farmer", "guitar", "helmet", "jacket", "kitchen", "ladder",
          "mirror", "needle", "office", "palace", "rabbit", "saddle", "ticket", "valley"]
# Induction roles are POSITIONAL, so every word must be exactly one token
# (with its leading space): a word that splits breaks the match between the
# two copies and the analysis cannot label ans_first / prev_first at all
# (measured 2026-09-09 — the first induction run labelled neither).
COMMON = [w for w in _WORDS if len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1]
print("induction vocabulary: %d of %d words are single-token" % (len(COMMON), len(_WORDS)))


def induction():
    """A B C D E F  A B C -> D. The canonical in-context copying task:
    the answer is in the prompt, so no world knowledge is involved."""
    rows, seen = [], set()
    for _ in range(N * 8):
        if len(rows) >= N:
            break
        seq = rng.sample(COMMON, 6)
        k = rng.choice([2, 3, 4])
        prompt = " ".join(seq) + " " + " ".join(seq[:k])
        wrong = rng.choice([w for w in COMMON if w not in seq])
        add(rows, prompt, " " + seq[k], " " + wrong,
            {"seq": seq, "k": k, "answer": seq[k], "prev": seq[k - 1]}, seen)
    return rows


def code_bracket():
    """close the open paren rather than start a block: needs the bracket
    state carried from '(' to the prediction position."""
    rows, seen = [], set()
    fns = ["f", "g", "step", "solve", "count", "total", "check", "score"]
    vs = ["x", "n", "k", "m", "value", "size"]
    for _ in range(N * 8):
        if len(rows) >= N:
            break
        v = rng.choice(vs)
        prompt = "def %s(%s):\n    return (%s %s %d" % (
            rng.choice(fns), v, v, rng.choice(["+", "-", "*"]), rng.randint(1, 20))
        add(rows, prompt, ")", ":", {"var": v}, seen)
    return rows


TASKS = {"capital": capital, "country_of_capital": country_of_capital, "month_succ": month_succ,
         "day_succ": day_succ, "number_succ": number_succ, "addition": addition,
         "pronoun": pronoun, "opposite": opposite, "induction": induction,
         "code_bracket": code_bracket}

for name, fn in TASKS.items():
    rows = fn()
    p = HERE / ("%s_rows.json" % name)
    json.dump(rows, open(p, "w"), indent=1)
    print("%-20s %4d rows -> %s | e.g. %r -> %r vs %r" % (
        name, len(rows), p.name, rows[0]["prompt"][-50:].replace("\n", "\\n"),
        tok.decode([rows[0]["target"]]), tok.decode([rows[0]["contrast"]])) if rows else
        "%-20s NO ROWS" % name)
