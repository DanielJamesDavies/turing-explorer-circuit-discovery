"""CIRCUIT LAB CLIENT for the case-study hunt. No GPU, no torch: plain Python on any machine that sees the repo.

GPU commands go to lab_server.py through lab_queue/ (the server must be running); catalogue commands read
results_lab/catalogue.jsonl directly.

  python experiments/063-case-studies/lab.py report 9.resid.7640
  python experiments/063-case-studies/lab.py probe 9.resid.7640,8.resid.12223 --text "It is not X, but Y." --text "..."
  python experiments/063-case-studies/lab.py ingredients 8.attn.29991 --text "A weak acid partially dissociates." [--at dissoc]
  python experiments/063-case-studies/lab.py top --text "The capital of France is Paris" --at Paris [--layers 6,7,8] [--n 5]
  python experiments/063-case-studies/lab.py contexts 7.resid.35144 [--n 8]
  python experiments/063-case-studies/lab.py find "\\d+ ?[+=×x] ?\\d" [--subset 3] [--field windows|logits|peaks|tags] [--limit 30]
  python experiments/063-case-studies/lab.py show 9.resid.7640         (the catalogue entry)

--text may be repeated; --at is a token index or a substring of the token to read (default: the target's peak for
ingredients, the last token for top). Activations are post-Top-K (0 = not in its site's Top-K at that token).
"""
import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).parent
QUEUE = HERE / "lab_queue"
CATALOGUE = HERE / "results_lab" / "catalogue.jsonl"


def ask(cmd, args, timeout=900):
    QUEUE.mkdir(exist_ok=True)
    rid = "%d-%s" % (int(time.time() * 1000), uuid.uuid4().hex[:8])
    req = QUEUE / ("%s.req.json" % rid)
    tmp = QUEUE / ("%s.req.tmp" % rid)
    tmp.write_text(json.dumps({"cmd": cmd, "args": args}), encoding="utf-8")
    tmp.replace(req)
    resp = QUEUE / ("%s.resp.md" % rid)
    t0 = time.time()
    while not resp.exists():
        if time.time() - t0 > timeout:
            req.unlink(missing_ok=True)
            sys.exit("timed out after %ds: is lab_server.py running? (queue: %d waiting)"
                     % (timeout, len(list(QUEUE.glob("*.req.json")))))
        time.sleep(0.4)
    time.sleep(0.1)
    body = resp.read_text(encoding="utf-8")
    resp.unlink(missing_ok=True)
    return body


def catalogue():
    return [json.loads(l) for l in open(CATALOGUE, encoding="utf-8")]


def entry(r):
    head = "## %s | n %d | Z %.2f A %.2f C %.2f | nec %.2f | ind %.2f | %s | cons %.0f%% | %s | %s" % (
        r["key"], r["n"], r["Z"], r["A"], r["C"], r["nec"], r["induce"], "AMP" if r["amp"] else "-",
        100 * r["consistency"], ", ".join("%r×%d" % tuple(p) for p in r["peaks"][:3]),
        " ".join("%s:%.1f" % kv for kv in r["tags"].items()) or "no tags")
    return "\n".join([head] + ["- …%s" % w for w in r["windows"]]
                     + ["- logits + %s | − %s" % (r["logit_up"], r["logit_down"])])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("report"); s.add_argument("key")
    s = sp.add_parser("probe"); s.add_argument("latents"); s.add_argument("--text", action="append", required=True)
    s = sp.add_parser("ingredients"); s.add_argument("key"); s.add_argument("--text", action="append", required=True)
    s.add_argument("--at"); s.add_argument("--n", type=int, default=15)
    s = sp.add_parser("top"); s.add_argument("--text", required=True); s.add_argument("--at")
    s.add_argument("--layers"); s.add_argument("--n", type=int, default=5)
    s = sp.add_parser("contexts"); s.add_argument("latent"); s.add_argument("--n", type=int, default=6)
    s = sp.add_parser("find"); s.add_argument("pattern"); s.add_argument("--subset", type=int)
    s.add_argument("--field", default="windows", choices=["windows", "logits", "peaks", "tags", "all"])
    s.add_argument("--limit", type=int, default=30); s.add_argument("--nsubsets", type=int, default=10)
    s = sp.add_parser("show"); s.add_argument("key")
    a = ap.parse_args()

    if a.cmd == "find":
        rx = re.compile(a.pattern, re.I)
        rows = catalogue()
        if a.subset is not None:
            rows = rows[a.subset::a.nsubsets]
        hits = []
        for r in rows:
            fields = {"windows": r["windows"], "logits": r["logit_up"] + r["logit_down"],
                      "peaks": [p[0] for p in r["peaks"]], "tags": list(r["tags"])}
            text = [t for f, v in fields.items() if a.field in (f, "all") for t in v]
            n = sum(len(rx.findall(t)) for t in text)
            if n:
                hits.append((n, r))
        hits.sort(key=lambda h: -h[0])
        print("%d matches for /%s/ in %s%s" % (len(hits), a.pattern, a.field,
                                              "" if a.subset is None else " (subset %d)" % a.subset))
        for n, r in hits[:a.limit]:
            print("\n[%d hits] %s" % (n, entry(r)))
        return
    if a.cmd == "show":
        for r in catalogue():
            if r["key"] == a.key:
                print(entry(r)); return
        sys.exit("not in the catalogue: %s" % a.key)
    if a.cmd == "report":
        print(ask("report", {"key": a.key}))
    elif a.cmd == "probe":
        print(ask("probe", {"latents": a.latents.split(","), "texts": a.text}))
    elif a.cmd == "ingredients":
        print(ask("ingredients", {"key": a.key, "texts": a.text, "at": a.at, "n": a.n}))
    elif a.cmd == "top":
        layers = [int(x) for x in a.layers.split(",")] if a.layers else None
        print(ask("top", {"text": a.text, "at": a.at, "layers": layers, "n": a.n}))
    elif a.cmd == "contexts":
        print(ask("contexts", {"latent": a.latent, "n": a.n}))


if __name__ == "__main__":
    main()
