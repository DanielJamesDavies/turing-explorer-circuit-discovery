"""CIRCUIT LAB SERVER: one GPU process that holds the model and SAE bank and answers circuit-hunt requests.

Agents never load the model themselves (a 16 GB card cannot hold one bank per agent). They call lab.py, which drops a
JSON request into lab_queue/ and waits for the markdown answer this server writes next to it. Requests are served
oldest first; while the queue is empty the server generates the missing inspect_circuit.py reports for the catalogue
(results_lab/catalogue.jsonl), so every target eventually has one.

Commands (see lab.py for the client side):
  report KEY                       the inspect_circuit.py report (generated on demand if missing)
  probe LATENTS TEXTS              post-Top-K activations of each latent at every token of each text
  ingredients KEY TEXTS [AT]       which circuit members are active at the target's peak (or at token AT) on each text,
                                   by contribution alpha x activation x decoder norm, and which top members are silent
  top TEXT [AT] [LAYERS] [N]       the strongest latents at one token, per site (discovery)
  contexts LATENT [N]              a latent's own strongest stored contexts, peak marked, plus its logit effect

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/063-case-studies/lab_server.py
"""
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("OUT", str(HERE.parent / "062-h100-protocol-v1" / "out_full" / "out"))
os.environ.setdefault("RESULTS", str(HERE / "results_full_deep"))
os.environ.setdefault("N_MEMBERS", "25")
import inspect_circuit as I  # noqa: E402

QUEUE = HERE / "lab_queue"
CATALOGUE = HERE / "results_lab" / "catalogue.jsonl"


def fmt_tokens(toks, acts, dec):
    """'tok tok[3.1] tok' with the activation printed on every token where it is nonzero (tokens space-separated)."""
    return " ".join(dec([t]).strip() + ("[%.1f]" % a if a > 0 else "") for t, a in zip(toks, acts))


class Lab:
    def __init__(self):
        self.d = I.scores_table()
        self.B = I.Browser()
        self.members_cache = {}

    # ------------------------------------------------------------------------------------------ helpers
    def encode(self, text):
        return self.B.tok.encode(text)

    def circuit(self, key):
        if key not in self.members_cache:
            c = torch.load(I.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
            mem = []
            for nd in c.nodes.values():
                md = nd.metadata
                if md.get("role") == "seed":
                    continue
                f = md["feature_id"]
                mem.append((int(f.layer), str(f.kind), int(f.index), float(md.get("amplitude", 1.0))))
            self.members_cache[key] = mem
        return self.members_cache[key]

    def at_index(self, toks, at, fallback):
        """Token index for AT: an int index, or the first token whose text contains AT, else the fallback."""
        if at is None or at == "":
            return fallback
        if isinstance(at, int) or str(at).lstrip("-").isdigit():
            return int(at) % len(toks)
        for j, t in enumerate(toks):
            if str(at) in self.B.dec([t]):
                return j
        return fallback

    # ------------------------------------------------------------------------------------------ commands
    def report(self, key):
        path = Path(os.environ["RESULTS"]) / "reports" / ("%s.md" % key)
        if not path.exists():
            txt, cdf = self.B.report(key, self.d.loc[key] if key in self.d.index else None)
            path.write_text(txt, encoding="utf-8")
            cdf.to_csv(path.parent.parent / "members" / ("%s.csv" % key), index=False)
        return path.read_text(encoding="utf-8")

    def probe(self, latents, texts):
        nodes = [I.parse(x) for x in latents]
        out = ["# probe: %s" % ", ".join(latents), ""]
        for text in texts:
            toks = self.encode(text)
            A = self.B.acts(nodes, torch.tensor([toks]))[0]                # [T, n]
            out.append("**%s**" % text)
            for j, name in enumerate(latents):
                a = A[:, j].tolist()
                m = max(a)
                out.append("- %s max %.2f%s: %s" % (name, m, " on %r" % self.B.dec([toks[a.index(m)]]) if m > 0 else "",
                                                   fmt_tokens(toks, a, self.B.dec)))
            out.append("")
        return "\n".join(out)

    def ingredients(self, key, texts, at=None, n_show=15):
        tl, tk, ti = I.parse(key)
        mem = self.circuit(key)
        nodes = [(l, k, i) for l, k, i, _ in mem] + [(tl, tk, ti)]
        csv = Path(os.environ["RESULTS"]) / "members" / ("%s.csv" % key)
        top25 = pd.read_csv(csv).node.head(25).tolist() if csv.exists() else []
        out = ["# ingredients of %s (%d members) on custom text" % (key, len(mem)),
               "At the chosen token: each ACTIVE member's contribution alpha x activation x decoder norm and its share; "
               "then the report's top-25 members that are SILENT there. Target activation shown per token.", ""]
        for text in texts:
            toks = self.encode(text)
            A = self.B.acts(nodes, torch.tensor([toks]))[0]                # [T, n + 1]
            tgt = A[:, -1].tolist()
            p = self.at_index(toks, at, int(torch.tensor(tgt).argmax()))
            out.append("**%s**" % text)
            out.append("- target per token: %s" % fmt_tokens(toks, tgt, self.B.dec))
            out.append("- at token %d %r: target %.2f" % (p, self.B.dec([toks[p]]), tgt[p]))
            rows = []
            for j, (l, k, i, a) in enumerate(mem):
                v = float(A[p, j])
                if v > 0:
                    rows.append(("%d.%s.%d" % (l, k, i), a, v, abs(a) * v * self.B.dec_norm(l, k, i)))
            tot = sum(r[3] for r in rows) or 1.0
            rows.sort(key=lambda r: -r[3])
            out.append("- %d of %d members active here; top %d:" % (len(rows), len(mem), min(n_show, len(rows))))
            for name, a, v, c in rows[:n_show]:
                out.append("  - %s act %.2f α %.2f share %.1f%%%s" % (name, v, a, 100 * c / tot,
                                                                    " (top-25)" if name in top25 else ""))
            active = {r[0] for r in rows}
            silent = [m for m in top25 if m not in active]
            out.append("- top-25 members SILENT here (%d): %s" % (len(silent), ", ".join(silent) or "none"))
            out.append("")
        return "\n".join(out)

    def top(self, text, at=None, layers=None, n=5):
        B, G = self.B, self.B.G
        toks = self.encode(text)
        p = self.at_index(toks, at, len(toks) - 1)
        want = set(layers) if layers else set(range(100))
        k2i = {k: n_ for n_, k in enumerate(B.KINDS)}
        found = []

        def hook(layer_idx, activations):
            if layer_idx not in want:
                return
            for kd in B.KINDS:
                ta, ti = B.bank.encode(activations[k2i[kd]], kd, layer_idx)
                v, ix = ta[0, p].float().cpu(), ti[0, p].long().cpu()
                order = v.argsort(descending=True)[:n]
                found.append((layer_idx, kd, [(int(ix[o]), float(v[o])) for o in order if v[o] > 0]))

        B.inference.disable_compile()
        try:
            with torch.no_grad():
                B.inference.forward(torch.tensor([toks], device=G["device"]), activations_callback=hook,
                                    return_activations=False, tokenize_final=False)
        finally:
            B.inference.enable_compile()
        out = ["# top latents at token %d %r of: %s" % (p, B.dec([toks[p]]), text), ""]
        for l, k, lst in sorted(found):
            out.append("- L%d %s: %s" % (l, k, ", ".join("%d.%s.%d (%.1f)" % (l, k, i, v) for i, v in lst)))
        return "\n".join(out)

    def contexts(self, latent, n=6):
        l, k, i = I.parse(latent)
        I.N_CTX = n
        t = self.B.own_contexts(l, k, i)
        if t is None:
            return "%s: no stored contexts" % latent
        a = self.B.acts([(l, k, i)], t)[:, :, 0]
        pk = a.argmax(1)
        toks = Counter(self.B.dec([t[b, int(pk[b])]]) for b in range(t.shape[0]))
        out = ["# %s own contexts" % latent,
               "peaks: %s" % ", ".join("%r×%d" % pc for pc in toks.most_common(5))]
        seen = set()
        for b in range(t.shape[0]):                               # the corpus repeats some passages: show each once
            w = self.B.window(t[b].tolist(), int(pk[b]), before=20, after=5)
            if w not in seen:
                seen.add(w)
                out.append("- (%.1f) …%s" % (float(a[b].max()), w))
        if k != "attn":
            up, down = self.B.logits(l, k, i, n=8)
            out.append("logits + %s | − %s" % (up, down))
        return "\n".join(out)

    def handle(self, req):
        c, a = req["cmd"], req.get("args", {})
        if c == "report":
            return self.report(a["key"])
        if c == "probe":
            return self.probe(a["latents"], a["texts"])
        if c == "ingredients":
            return self.ingredients(a["key"], a["texts"], a.get("at"), int(a.get("n", 15)))
        if c == "top":
            return self.top(a["text"], a.get("at"), a.get("layers"), int(a.get("n", 5)))
        if c == "contexts":
            return self.contexts(a["latent"], int(a.get("n", 6)))
        raise ValueError("unknown command %r" % c)


def backlog():
    if not CATALOGUE.exists():
        return []
    have = {p.stem for p in (Path(os.environ["RESULTS"]) / "reports").glob("*.md")}
    keys = [json.loads(l)["key"] for l in open(CATALOGUE, encoding="utf-8")]
    return [k for k in keys if k not in have]


def main():
    QUEUE.mkdir(exist_ok=True)
    for sub in ("reports", "members"):
        (Path(os.environ["RESULTS"]) / sub).mkdir(parents=True, exist_ok=True)
    lab = Lab()
    todo = backlog()
    print("lab ready; %d reports in the idle backlog" % len(todo), flush=True)
    served = 0
    while True:
        reqs = sorted(QUEUE.glob("*.req.json"), key=lambda p: p.stat().st_mtime)
        if reqs:
            p = reqs[0]
            try:
                req = json.loads(p.read_text(encoding="utf-8"))
                t0 = time.time()
                try:
                    body = lab.handle(req)
                except Exception:  # noqa: BLE001
                    body = "ERROR\n```\n%s```" % traceback.format_exc()
                tmp = p.with_name(p.name.replace(".req.json", ".resp.tmp"))
                tmp.write_text(body, encoding="utf-8")
                tmp.replace(p.with_name(p.name.replace(".req.json", ".resp.md")))
                served += 1
                print("served %-11s %-40s %.1fs (total %d)" % (req.get("cmd"), str(req.get("args", {}))[:40],
                                                              time.time() - t0, served), flush=True)
            finally:
                p.unlink(missing_ok=True)
            continue
        if todo:                                                   # idle: fill in one missing report
            key = todo.pop(0)
            try:
                lab.report(key)
            except Exception as e:  # noqa: BLE001
                print("  backlog %s FAILED: %s" % (key, e), flush=True)
            if len(todo) % 100 == 0:
                print("  backlog: %d reports left" % len(todo), flush=True)
            continue
        time.sleep(0.3)


if __name__ == "__main__":
    main()
