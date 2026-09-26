"""STAGE 0a — IS GEMMA 3 270M A BETTER SUBSTRATE THAN TURINGLLM?

Same contrastive-decision battery run on three backends so the answer is a
side-by-side, not an impression:
  BACKEND=turing   TuringLLM (12L x 1024, our model)        -> the baseline
  BACKEND=hf       any HF causal LM, default unsloth/gemma-3-270m (12L x 1024)
                   (HF_MODEL=unsloth/gemma-2-2b for the upper reference)

Each task yields (prompt, target, contrast). A prompt is kept only when both
answers are a SINGLE next token under that backend's tokenizer, so n can differ
by model. Reported per task: n, argmax accuracy, fraction with a positive
target-contrast margin, mean margin. Accuracy and pos% are the comparable
columns across tokenizers; the margin is only comparable within a model.

  BACKEND=turing PYTHONPATH=src python experiments/052-gemma3-270m/capability_probe.py
  BACKEND=hf     PYTHONPATH=src python experiments/052-gemma3-270m/capability_probe.py
  BACKEND=hf HF_MODEL=unsloth/gemma-2-2b ...
"""
import json
import os
import random
from pathlib import Path

import torch

HERE = Path(__file__).parent
BACKEND = os.environ.get("BACKEND", "hf")
HF_MODEL = os.environ.get("HF_MODEL", "unsloth/gemma-3-270m")
N = int(os.environ.get("N", 60))
rng = random.Random(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if BACKEND == "turing":
    from hardware import detect_devices, should_compile
    from model.inference import Inference
    from model.tokenizer import Tokenizer
    _devices = detect_devices()
    _inf = Inference(device=_devices[0], compile=should_compile())
    _inf.disable_compile()
    _tok = Tokenizer()
    NAME = "TuringLLM"

    def enc(text):
        return list(_tok.encode(text))

    def last_logits(ids):
        tk = torch.tensor([ids], dtype=torch.long, device=_devices[0])
        with torch.no_grad():
            res = _inf.forward(tk, all_logits=True, grad_enabled=False,
                               return_activations=False, tokenize_final=False)
        lg = res[1] if isinstance(res, (tuple, list)) else res
        return lg[0, -1].float()
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(HF_MODEL)
    _model = AutoModelForCausalLM.from_pretrained(HF_MODEL, dtype=torch.bfloat16).to(dev).eval()
    NAME = HF_MODEL

    def enc(text):
        return _tok(text, return_tensors="pt")["input_ids"][0].tolist()

    def last_logits(ids):
        with torch.no_grad():
            return _model(torch.tensor([ids], device=dev)).logits[0, -1].float()


def tok_after(prefix, word):
    """single next-token id for `word` after `prefix`, or None if it splits."""
    a, b = enc(prefix), enc(prefix + word)
    return b[len(a)] if (b[:len(a)] == a and len(b) == len(a) + 1) else None


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
# city -> (country, language) for the two-hop task; the country is never written
TWOHOP = [("Munich", "Germany", "German"), ("Hamburg", "Germany", "German"), ("Lyon", "France", "French"),
          ("Marseille", "France", "French"), ("Milan", "Italy", "Italian"), ("Naples", "Italy", "Italian"),
          ("Barcelona", "Spain", "Spanish"), ("Seville", "Spain", "Spanish"), ("Osaka", "Japan", "Japanese"),
          ("Kyoto", "Japan", "Japanese"), ("Shanghai", "China", "Chinese"), ("Porto", "Portugal", "Portuguese"),
          ("Krakow", "Poland", "Polish"), ("Bergen", "Norway", "Norwegian"), ("Gothenburg", "Sweden", "Swedish")]


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
    seen = set()
    for _ in range(N * 4):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        yield "%d + %d =" % (a, b), " %d" % (a + b), " %d" % (a + b + 1)


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
    for _ in range(N):
        seq = rng.sample(COMMON, 6)
        text = " ".join(seq) + " " + " ".join(seq[:3])
        yield text, " " + seq[3], " " + rng.choice([w for w in COMMON if w not in seq])


def t_ioi():
    """indirect object identification: the repeated name is the subject, answer the other"""
    places = ["store", "park", "office", "station", "garden", "market"]
    objs = ["a drink", "a book", "a letter", "a gift", "a key"]
    for _ in range(N):
        a, b = rng.sample(FEM + MASC, 2)
        yield ("When %s and %s went to the %s, %s gave %s to" % (a, b, rng.choice(places), b, rng.choice(objs)),
               " " + a, " " + b)


def t_agreement():
    """subject-verb agreement across an attractor of the OPPOSITE number"""
    pairs = [("key", "keys"), ("book", "books"), ("letter", "letters"), ("student", "students"),
             ("report", "reports"), ("picture", "pictures")]
    attractors = [("cabinet", "cabinets"), ("teacher", "teachers"), ("table", "tables"), ("river", "rivers")]
    for _ in range(N):
        (s, sp), (x, xp) = rng.choice(pairs), rng.choice(attractors)
        if rng.random() < 0.5:
            yield "The %s near the %s" % (s, xp), " is", " are"
        else:
            yield "The %s near the %s" % (sp, x), " are", " is"


def t_twohop():
    """city -> (unwritten) country -> language"""
    langs = sorted({l for _, _, l in TWOHOP})
    for city, _, lang in TWOHOP:
        other = rng.choice([l for l in langs if l != lang])
        yield "The main language spoken in the city of %s is" % city, " " + lang, " " + other


def t_list_close():
    items = ["apples", "pears", "plums", "grapes", "melons", "lemons", "cherries", "peaches"]
    for _ in range(N):
        seq = rng.sample(items, 4)
        yield "We bought %s, %s, %s," % tuple(seq[:3]), " and", " or"


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


TASKS = {"ioi": t_ioi, "induction": t_induction, "greater_than": t_greater, "agreement_attractor": t_agreement,
         "twohop_language": t_twohop, "capital": t_capital, "country_of_capital": t_country,
         "month_succ": t_month, "day_succ": t_day, "number_succ": t_number, "addition_digits": t_add,
         "past_tense": t_past, "plural": t_plural, "opposite": t_opposite, "pronoun": t_pronoun,
         "list_close": t_list_close, "acronym": t_acronym, "code_bracket": t_bracket}

report = {}
print("MODEL: %s" % NAME)
print("%-20s %4s %6s %6s %8s   example" % ("task", "n", "acc", "pos%", "margin"))
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
        report[name] = dict(n=0, split=split)
        print("%-20s   --  (no single-token prompts; %d split)" % (name, split), flush=True)
        continue
    acc = margin = pos = 0.0
    for prompt, ti, tc, _, _ in rows:
        lg = last_logits(enc(prompt))
        d = float(lg[ti] - lg[tc])
        acc += int(lg.argmax() == ti); margin += d; pos += int(d > 0)
    m = len(rows)
    report[name] = dict(n=m, split=split, acc=acc / m, pos=pos / m, margin=margin / m,
                        example=rows[0][0][-70:], target=rows[0][3], contrast=rows[0][4])
    print("%-20s %4d %6.2f %6.2f %+8.2f   %r -> %r vs %r"
          % (name, m, acc / m, pos / m, margin / m, rows[0][0][-40:].replace("\n", "\\n"),
             rows[0][3], rows[0][4]), flush=True)

tag = "turing" if BACKEND == "turing" else HF_MODEL.split("/")[-1]
json.dump(dict(model=NAME, tasks=report), open(HERE / ("capability_%s.json" % tag), "w"), indent=1)
print("-> capability_%s.json" % tag)
