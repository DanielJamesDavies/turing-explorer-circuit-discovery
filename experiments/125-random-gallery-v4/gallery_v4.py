"""DAN-219 gallery decision: the 072 random-circuit gallery, redrawn from the PRODUCTION run (WCM v4, TuringLLM 8k,
2026-11-07/08 pod) instead of protocol v1.

Same design as 072 (Daniel, 2026-10-12), production data throughout:
  - draw: one target per depth band x site kind (L0-4 / L5-7 / L8-11 x attn / mlp / resid), uniformly from the
    NON-VACUOUS production circuits of that cell (held-out strongest row, merge.py v2 vacuous rule), any outcome,
    random.Random(SEED).choice over the sorted keys of each cell in 072's order (SEED 20261012, as 072);
  - scores: the production eval row (held-out strongest): Z / A / C, necessity, induce, size; pass = merge.py rule
    (floor 0.75); amplifier / near-threshold = wcm_specificity.target_specificity on the production spec rows (as 111);
  - members: contribution share at the target's anchor (|alpha| x mean activation at the anchor x decoder norm,
    063 / 110 member_table) on the production circuit and the fresh (deduplicated) records; generic hubs excluded;
  - hubs: the explorer converter's HUB_RULE (stage_reading.py) with its membership count taken over the 7,786
    production circuits (> 50%) and its specificity part (median pos_max / neg_max <= 1.5 over >= 10 circuits) taken
    from the v1 bundle's reading_hub, the only place that median was measured; a candidate with no v1 median is kept
    (is_hub NULL, as the converter);
  - target labels: the 065 circuit describer (prompt v5, describe.SYSTEM_PROMPT + schema_for), on a report in
    065 report.build_report's format built from the production circuit and fresh records (build_report itself reads
    the v1 bundle);
  - member labels: 072's latent-only call, verbatim (072 gallery.LATENT_SYSTEM_PROMPT / LATENT_SCHEMA /
    build_prompt / latent_report). A member's own evidence (peak tokens, strongest stored contexts, logit effect) is a
    property of the SAE latent, not of the run, so it is read from the v1 bundle's reading_latent when present, else
    computed from the stored top contexts in the same format.

Stages (env STAGE, default all; each caches into results/):
  draw     (CPU, seconds)   results/draw.json
  hubs     (CPU, minutes)   results/member_freq.json (membership count of every latent over the production circuits)
  members  (model; CPU or GPU) results/members/<key>.csv, results/latents.json (computed latent readings),
                            results/target_reports/<key>.txt
  labels   (claude CLI)     results/target_labels.jsonl, results/member_labels.jsonl
  write                     results/gallery.json, gallery.md, gallery_table.tex

  PYTHONPATH=src python -X utf8 experiments/125-random-gallery-v4/gallery_v4.py
  env: SEED (20261012)  WORKERS (4; at most 8)  DRY=1 (print one target and one member prompt, call nothing)
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments"
# 065 first, so `report` / `describe` resolve to the describer's modules (110 also has a report.py).
sys.path.insert(0, str(EXP / "065-circuit-describer"))
sys.path.insert(0, str(ROOT / "src"))
import describe as D  # noqa: E402
import report as R65  # noqa: E402

sys.path.append(str(EXP / "110-case-studies-v4"))
import lib110 as L  # noqa: E402  (production paths; also puts 109 / 062 / 063 on sys.path)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G72 = _load("gallery072", EXP / "072-random-gallery" / "gallery.py")   # 072's prompt, schema, tex, num

_find_claude_065 = D.find_claude


def _find_claude():
    """065 find_claude, plus the desktop app's newer CLI location (%LOCALAPPDATA%/Claude-Data/claude_*/claude-code/
    <version>/<hash>/claude.exe, 2026-11), which 065 does not search. Highest version wins."""
    import glob
    import re
    exe = _find_claude_065()
    if exe:
        return exe
    local = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    cands = glob.glob(os.path.join(local, "Claude-Data", "claude_*", "claude-code", "*", "*", "claude.exe"))
    cands += glob.glob(os.path.join(local, "Claude-Data", "claude_*", "claude-code", "*", "claude.exe"))
    ver = lambda p: [int(n) for n in re.findall(r"\d+", next(s for s in Path(p).parts if re.fullmatch(r"[\d.]+", s)))]
    return max(cands, key=ver) if cands else None


D.find_claude = _find_claude

SEED = int(os.environ.get("SEED", 20261012))
WORKERS = min(int(os.environ.get("WORKERS", 4)), D.MAX_WORKERS)
N_MEMBERS = 3
BANDS = G72.BANDS
KINDS = R65.KINDS
RESULTS = HERE / "results"
DRAW = RESULTS / "draw.json"
FREQ = RESULTS / "member_freq.json"
MEMBERS = RESULTS / "members"
LATENTS = RESULTS / "latents.json"
TREPORTS = RESULTS / "target_reports"
TLABELS = RESULTS / "target_labels.jsonl"
MLABELS = RESULTS / "member_labels.jsonl"
HUB_MIN_FRAC, HUB_MAX_RATIO, HUB_MIN_COVERED = 0.5, 1.5, 10      # stage_reading.py HUB_RULE
N_CTX_OWN = 6                                                      # stage_reading.py N_CTX
N_LOGIT = 10                                                       # stage_reading.py N_LOGIT


def band_of(layer: int) -> str:
    return G72.band_of(int(layer))


# ----------------------------------------------------------------------------- draw + scores
def production_table():
    """Production held-out-strongest rows (110 lib: 109 reader, merge.py pass rule + v2 vacuous) + amplifier /
    near-threshold flags from the production spec rows (111's per_target_spec = wcm_specificity)."""
    d = L.scores("strong")
    from eval.wcm_specificity import amplifier_flag, target_specificity
    import lib109 as L9
    sp = L9.frame(L9.read_rows([L.OUT_V4 / "main"], "spec"), keys=("seed", "pi"))
    for c in ("target_faith_pre", "sibling_faith_median", "control_faith_median", "switched_on_circuit",
              "switched_on_empty", "in_topk_circuit", "in_topk_clean", "rank_clean", "K"):
        if c in sp:
            sp[c] = sp[c].astype(float)
    flags = {}
    for seed, g in sp.groupby("seed"):
        t = target_specificity(g.to_dict("records"))
        flags[seed] = (bool(t.get("amp_any")), bool(t.get("near_threshold")))
    d["amp_any"] = [flags.get(s, (False, False))[0] for s in d.index]
    d["near_threshold"] = [flags.get(s, (False, False))[1] for s in d.index]
    return d


def draw(d) -> list[str]:
    cells: dict[tuple[str, str], list[str]] = {}
    for key, vac in zip(d.index, d["vacuous"]):
        if bool(vac):
            continue
        cells.setdefault((band_of(int(key.split(".")[0])), key.split(".")[1]), []).append(key)
    rng = random.Random(SEED)
    keys = [rng.choice(sorted(cells[(band, kind)])) for band, _ in BANDS for kind in KINDS]
    sizes = {f"{b}/{k}": len(v) for (b, k), v in sorted(cells.items())}
    return keys, sizes


def stage_draw():
    d = production_table()
    keys, sizes = draw(d)
    nv = d[~d["vacuous"].astype(bool)]
    run = {"n_scored": int(len(nv)), "n_pass": int(nv["passes"].sum()),
           "pass_rate": float(nv["passes"].mean()), "amp_rate": float(nv["amp_any"].mean()),
           "near_threshold_rate": float(nv["near_threshold"].mean())}
    rows = []
    for k in keys:
        r = d.loc[k]
        num = lambda c: None if r[c] is None or (isinstance(r[c], float) and np.isnan(r[c])) else float(r[c])
        rows.append({"key": k, "band": band_of(int(k.split(".")[0])), "kind": k.split(".")[1], "n": int(r["n"]),
                     "Z": num("free0_tk"), "A": num("freeM_topk_tk"), "C": num("freeN_topk_tk"),
                     "necessity": num("phi_sup_blind_tk"), "induce": num("phi_cf_alpha_blind_tk"),
                     "pass": bool(r["passes"]), "amplifier": bool(r["amp_any"]),
                     "near_threshold": bool(r["near_threshold"]), "worst_dev": num("worst_dev")})
    RESULTS.mkdir(parents=True, exist_ok=True)
    DRAW.write_text(json.dumps({"seed": SEED, "population": "production TuringLLM 8k, held-out strongest, "
                                "non-vacuous (merge.py v2)", "cell_sizes": sizes, "run": run, "rows": rows},
                               indent=1), encoding="utf-8")
    print(f"seed {SEED}: " + ", ".join(f"{r['key']} ({'pass' if r['pass'] else 'fail'})" for r in rows))
    print(f"run: {run}")


# ----------------------------------------------------------------------------- hubs
def stage_hubs():
    if FREQ.exists():
        return
    import torch
    files = sorted((L.OUT_V4 / "main" / "circuits").glob("*.pt"))
    cnt: Counter = Counter()
    t0 = time.time()
    for n, p in enumerate(files):
        c = torch.load(p, weights_only=False)
        cnt.update("%d.%s.%d" % (l, k, i) for l, k, i, _ in L.members(c))
        if n % 500 == 0:
            print(f"  hubs {n}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    FREQ.write_text(json.dumps({"n_circuits": len(files), "count": dict(cnt.most_common())}), encoding="utf-8")


def hub_set(b: R65.Bundle) -> tuple[set[str], dict]:
    f = json.loads(FREQ.read_text(encoding="utf-8"))
    n = f["n_circuits"]
    cand = {k: c for k, c in f["count"].items() if c > HUB_MIN_FRAC * n}
    hubs, info = set(), {}
    for k, c in cand.items():
        r = b.reading.execute("SELECT n_covered, med_ratio, is_hub FROM reading_hub WHERE gid = ?",
                              (R65.gid_of(k),)).fetchone()
        ok = r is not None and r[0] is not None and r[0] >= HUB_MIN_COVERED and r[1] is not None \
            and r[1] <= HUB_MAX_RATIO
        info[k] = {"frac": c / n, "v1_med_ratio": None if r is None else r[1], "hub": bool(ok)}
        if ok:
            hubs.add(k)
    return hubs, info


# ----------------------------------------------------------------------------- text helpers (065 Bundle format)
def win(b, row, p, before, after) -> str:
    """065 Bundle.window on a token row (not a bundle seq id): peak in [[ ]], BOS as ¶, pad dropped."""
    row = [int(t) for t in row]
    lo, hi = max(0, p - before), min(len(row), p + after + 1)
    out = []
    for i in range(lo, hi):
        t = row[i]
        s = "¶" if t == 1 else ("" if t == 0 else b.piece(t))
        out.append(f"[[{s}]]" if i == p else s)
    return ("…" if lo > 0 else "") + "".join(out) + ("…" if hi < len(row) else "")


def distinct_rows(b, refs, n, before, after) -> list[str]:
    """065 distinct_windows on (token row, pos) pairs."""
    out, seen_text, seen_prev = [], set(), set()
    for row, p in refs:
        text = win(b, row, p, before, after)
        prev = tuple(int(t) for t in row[max(0, p - 2):p])
        if text in seen_text or prev in seen_prev:
            continue
        seen_text.add(text)
        seen_prev.add(prev)
        out.append(text)
        if len(out) == n:
            break
    return out


def span(b, row, lo, hi) -> str:
    return "…" + "".join("¶" if int(t) == 1 else b.piece(int(t)) for t in row[lo:hi] if int(t) != 0) + "…"


# ----------------------------------------------------------------------------- latent readings
class Latents:
    """A latent's own reading: from the v1 bundle's reading_latent (windows as bundle rows) or computed here from its
    stored top contexts (windows as token rows). Same fields: peaks [(tok, n)], consistency, ctx [(row, pos)], up,
    down [(tok, value)]."""

    def __init__(self, b):
        self.b = b
        self.computed = json.loads(LATENTS.read_text(encoding="utf-8")) if LATENTS.exists() else {}

    def get(self, key):
        lat = self.b.latent(R65.gid_of(key))
        if lat is not None:
            return {"src": "v1_bundle", "peaks": lat["peaks"], "consistency": lat["consistency"],
                    "ctx": [(np.asarray(self.b.tokens[s - 1]), p) for s, p, _ in lat["ctx"]],
                    "up": lat["up"], "down": lat["down"]}
        c = self.computed.get(key)
        if c is None:
            return None
        return {"src": "computed", "peaks": c["peaks"], "consistency": c["consistency"],
                "ctx": [(np.asarray(r), p) for r, p in c["ctx"]], "up": c["up"], "down": c["down"]}


def compute_latent(B, key):
    """A reading_latent-shaped record from the latent's stored top contexts (063 Browser.own_contexts + acts)."""
    import torch
    l, k, i = L.parse(key)
    t = B.own_contexts(l, k, i)
    up, down = logits_pairs(B, l, k, i)
    if t is None:
        return {"peaks": [], "consistency": 0.0, "ctx": [], "up": up, "down": down}
    a = B.acts([(l, k, i)], t)[:, :, 0]
    pk = a.argmax(1)
    order = torch.argsort(a.max(1).values, descending=True).tolist()
    toks = Counter(int(t[r, int(pk[r])]) for r in range(t.shape[0]))
    peaks = [[tok, n] for tok, n in toks.most_common()]
    return {"peaks": peaks, "consistency": peaks[0][1] / t.shape[0],
            "ctx": [[t[r].tolist(), int(pk[r])] for r in order], "up": up, "down": down}


def logits_pairs(B, l, k, i):
    wd = B.bank.saes[k][l].decoder.weight.detach()[:, i].float().to(B.W_U.device)
    lg = ((wd * B.g_norm.to(wd.device)) @ B.W_U.T)[:R65.N_REAL_VOCAB].cpu()
    top, bot = lg.topk(N_LOGIT), (-lg).topk(N_LOGIT)
    return ([[int(t), float(v)] for t, v in zip(top.indices, top.values)],
            [[int(t), -float(v)] for t, v in zip(bot.indices, bot.values)])


# ----------------------------------------------------------------------------- members (model)
def stage_members():
    rows = json.loads(DRAW.read_text(encoding="utf-8"))["rows"]
    todo = [r["key"] for r in rows if not (MEMBERS / f"{r['key']}.csv").exists()]
    b = R65.Bundle()
    hubs, _ = hub_set(b)
    MEMBERS.mkdir(parents=True, exist_ok=True)
    TREPORTS.mkdir(parents=True, exist_ok=True)
    lats = Latents(b)
    need_lat = [r["key"] for r in rows]                       # targets' own logits / readings if not in v1
    if todo or any(lats.get(k) is None for k in need_lat):
        r110 = _load("report110", EXP / "110-case-studies-v4" / "report.py")    # sets the production env
        import inspect_circuit as IC
        B = IC.Browser()
        t0 = time.time()
        for key in todo:
            cdf = r110.member_table(B, key, L.load_circuit(key), L.record(key))
            cdf["hub"] = cdf.node.isin(hubs)
            cdf.to_csv(MEMBERS / f"{key}.csv", index=False)
            print(f"  members {key}: {len(cdf)} ({cdf.hub.sum()} hubs) {time.time() - t0:.0f}s", flush=True)
        # Latent readings missing from the v1 bundle: the targets, every member shown in a target report (15) and
        # the top-3 gallery members.
        import pandas as pd
        want = list(need_lat)
        for key in [r["key"] for r in rows]:
            cdf = pd.read_csv(MEMBERS / f"{key}.csv")
            want += cdf[~cdf.hub].node.head(15).tolist()
        comp = lats.computed
        for k in dict.fromkeys(want):
            if lats.get(k) is None:
                comp[k] = compute_latent(B, k)
                print(f"  latent {k} computed", flush=True)
        LATENTS.write_text(json.dumps(comp), encoding="utf-8")
        print(f"members stage done {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- target report (065 format)
def target_report(b, lats, row, n_members=15, n_windows=5, n_logits=6, member_logits=4, n_mid=2, n_contrast=4):
    """065 report.build_report, line for line, on production data: the production circuit's scores and member
    table, the fresh record's strong / mid / contrast contexts. Returns (text, merged_consistency)."""
    import pandas as pd
    key = row["key"]
    rec = L.record(key)
    cdf = pd.read_csv(MEMBERS / f"{key}.csv")
    st = rec["strong"]
    pos, arg, peak = st["pos"], st["arg"].long(), st["peak"].float()
    order = sorted(range(pos.shape[0]), key=lambda j: -float(peak[j]))
    n_strong = pos.shape[0]
    toks = Counter(int(pos[j, int(arg[j])]) for j in range(n_strong))
    peaks = toks.most_common()
    consistency = peaks[0][1] / n_strong
    merged = max(consistency, R65.merged_consistency(b, peaks, n_strong))
    top25 = cdf.share.clip(lower=0).head(25).sum()

    flags = ["pass" if row["pass"] else "fail"]
    if row["amplifier"]:
        flags.append("also lifts related latents")
    if row["near_threshold"]:
        flags.append("near TopK threshold")
    num = lambda v, spec: "n/a" if v is None else format(v, spec)
    lines = [
        f"CIRCUIT {key} ({', '.join(flags)}; {len(cdf)} members; top 25 carry {num(top25, '.0%')}; "
        f"faithfulness Z {num(row['Z'], '.2f')} A {num(row['A'], '.2f')} C {num(row['C'], '.2f')})",
        "",
        "TARGET",
        f"peak tokens: {R65.peaks_text(b, peaks, n_strong)}; consistency {consistency:.0%}"
        + (f" ({merged:.0%} with case / plural / prefix variants merged)" if merged > consistency + 0.005 else "")
        + ("; trigger: one_token" if merged >= 0.8 else ""),
        "windows (peak in [[ ]], ¶ = document start):",
    ]
    # reading_target windows = distinct windows of the strongest contexts (stage_reading N_WIN, WIN = 12, 3)
    lines += [f"- {w}" for w in distinct_rows(b, [(pos[j].tolist(), int(arg[j])) for j in order], n_windows, 12, 3)]

    mid = rec.get("mid")
    if mid is not None and mid["pos"].shape[0]:
        mo = sorted(range(mid["pos"].shape[0]), key=lambda j: -float(mid["peak"][j]))
        mid_lines = distinct_rows(b, [(mid["pos"][j].tolist(), int(mid["arg"][j])) for j in mo], n_mid, 12, 3)
        if mid_lines:
            rel = f" (about {float(mid['peak'][mo[0]]) / float(peak[order[0]]):.0%} of the strongest)"
            lines.append(f"weaker contexts{rel}:")
            lines += [f"- {w}" for w in mid_lines]

    top_token = peaks[0][0]
    neg = rec["neg"]
    strong_prev = {int(pos[j, int(arg[j]) - 1]) for j in range(n_strong) if int(arg[j]) > 0}
    same_token = []
    for s in range(neg.shape[0]):
        hits = np.nonzero(neg[s].numpy() == top_token)[0]
        if len(hits):
            same_token.append((neg[s].tolist(), int(hits[0])))
    same_token.sort(key=lambda sp: sp[1] == 0 or int(sp[0][sp[1] - 1]) not in strong_prev)
    silent_lines = distinct_rows(b, same_token, n_contrast, 12, 3)
    if silent_lines:
        lines.append(f"contrast, target SILENT on its own peak token {b.piece(int(top_token))!r} here "
                     f"({len(same_token)} of {neg.shape[0]} contrast contexts contain it):")
        lines += [f"- {w}" for w in silent_lines]
    else:
        lines.append(f"contrast, target silent: {span(b, neg[0].tolist(), 20, 44)}")   # CONTRAST_SPAN
    t_lat = lats.get(key)
    if t_lat:
        lines.append(f"logits + {b.tokens_list(t_lat['up'], n_logits)} | − {b.tokens_list(t_lat['down'], n_logits)}")

    hubs = cdf[cdf.hub]
    kept = cdf[~cdf.hub]
    ratio = kept.pos_max / kept.neg_max.clip(lower=R65.RATIO_EPS)
    disc_share = kept.share[ratio >= R65.DISCRIMINATIVE_RATIO].sum()
    lines += [
        "",
        f"MEMBERS: top {min(n_members, len(kept))} of {len(kept)} by share at the target's peak token "
        f"({len(hubs)} generic hubs dropped, {hubs.share.sum():.0%} of share; "
        f"discriminative members carry {disc_share:.0%}).",
        "Columns: share | alpha | specificity = strongest act on target contexts / on contrast (>=3 = switches off "
        "on contrast) | own peak tokens | own logits + | one own context",
    ]
    for m in kept.head(n_members).itertuples():
        lat = lats.get(m.node) or {"peaks": [], "up": [], "ctx": []}
        rt = m.pos_max / max(m.neg_max, R65.RATIO_EPS)
        spec = ">100" if rt > 100 else f"{rt:.1f}"
        own = win(b, lat["ctx"][0][0], lat["ctx"][0][1], 8, 2) if lat["ctx"] else "-"
        lines.append(f"{m.node} | {m.share:.1%} | a{m.alpha:.2f} | spec {spec} | "
                     f"{R65.peaks_text(b, lat['peaks'], 0, 3)} | + {b.tokens_list(lat['up'], member_logits)} | {own}")
    return "\n".join(lines), merged


# ----------------------------------------------------------------------------- member report (072 latent_report)
def latent_report(b, lats, key) -> str:
    """072 gallery.latent_report, line for line, on Latents records."""
    lat = lats.get(key)
    if lat is None:
        return f"LATENT {key}: no reading data"
    total = sum(c for _, c in lat["peaks"])
    merged = R65.merged_consistency(b, lat["peaks"], total) if total else 0.0
    lines = [f"LATENT {key}",
             f"peak tokens: {R65.peaks_text(b, lat['peaks'], total)}; consistency {lat['consistency']:.0%}"
             + (f" ({merged:.0%} with case / plural / prefix variants merged)" if merged > lat["consistency"] + 0.005
                else ""),
             "windows from its strongest contexts (peak in [[ ]], ¶ = document start):"]
    lines += [f"- {w}" for w in distinct_rows(b, lat["ctx"], 6, 12, 3)]
    lines.append(f"logits + {b.tokens_list(lat['up'], 6)} | − {b.tokens_list(lat['down'], 6)}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- labels
def gallery_rows(b):
    import pandas as pd
    rows = json.loads(DRAW.read_text(encoding="utf-8"))["rows"]
    for r in rows:
        cdf = pd.read_csv(MEMBERS / f"{r['key']}.csv")
        kept = cdf[~cdf.hub]
        r["hubs_dropped"] = int(cdf.hub.sum())
        r["members"] = [{"key": m.node, "alpha": float(m.alpha), "share": float(m.share),
                         "specificity": float(m.pos_max / max(m.neg_max, 1e-3))} for m in kept.head(N_MEMBERS).itertuples()]
    return rows


def _done(path):
    return {r["key"]: r for r in D.read_records(path) if r.get("result")} if path.exists() else {}


def stage_labels():
    b = R65.Bundle()
    lats = Latents(b)
    rows = gallery_rows(b)
    TREPORTS.mkdir(parents=True, exist_ok=True)
    reports = {}
    for r in rows:
        text, merged = target_report(b, lats, r)
        (TREPORTS / f"{r['key']}.txt").write_text(text, encoding="utf-8")
        reports[r["key"]] = (text, merged)
    member_keys = list(dict.fromkeys(m["key"] for r in rows for m in r["members"]))
    if os.environ.get("DRY") == "1":
        k = rows[0]["key"]
        print("=== TARGET PROMPT ===\n" + D.build_prompt(k, reports[k][0]))
        print("=== MEMBER PROMPT ===\n" + G72.build_prompt(member_keys[0], latent_report(b, lats, member_keys[0])))
        return
    # describe.ask_claude reads SYSTEM_PROMPT at call time: targets first (v5 describer), then the 072 latent prompt.
    sys_v5 = D.SYSTEM_PROMPT
    done = _done(TLABELS)
    todo = [r["key"] for r in rows if r["key"] not in done]
    print(f"target labels: {len(done)} cached, {len(todo)} to ask", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        fut = {}
        for k in todo:
            one_token = reports[k][1] >= D.ONE_TOKEN_MIN
            fut[pool.submit(D.ask_claude, D.build_prompt(k, reports[k][0]), "sonnet", 180, D.schema_for(one_token))] \
                = (k, one_token)
        for f in as_completed(fut):
            k, one_token = fut[f]
            payload, meta = f.result()
            rec = {"key": k, "model": "sonnet", "prompt_version": D.PROMPT_VERSION, "source": "production v4",
                   "time": time.strftime("%Y-%m-%d %H:%M:%S"), "report_chars": len(reports[k][0]),
                   "merged_consistency": round(reports[k][1], 3), "trigger_computed": one_token, **meta,
                   "result": payload}
            with TLABELS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  {k}: {payload['label'] if payload else 'ERROR ' + str(meta.get('error'))}", flush=True)
    D.SYSTEM_PROMPT = G72.LATENT_SYSTEM_PROMPT
    try:
        done = _done(MLABELS)
        todo = [k for k in member_keys if k not in done]
        print(f"member labels: {len(done)} cached, {len(todo)} to ask", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            fut = {pool.submit(G72.ask_latent, G72.build_prompt(k, latent_report(b, lats, k))): k for k in todo}
            for f in as_completed(fut):
                k = fut[f]
                payload, meta = f.result()
                rec = {"key": k, "model": "sonnet", "prompt_version": G72.LATENT_PROMPT_VERSION,
                       "reading": (lats.get(k) or {}).get("src"), **meta, "result": payload}
                with MLABELS.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  {k}: {payload['label'] if payload else 'ERROR ' + str(meta.get('error'))}", flush=True)
    finally:
        D.SYSTEM_PROMPT = sys_v5


# ----------------------------------------------------------------------------- outputs (072 format)
def stage_write():
    b = R65.Bundle()
    rows = gallery_rows(b)
    tl, ml = _done(TLABELS), _done(MLABELS)
    for r in rows:
        res = (tl.get(r["key"]) or {}).get("result") or {}
        r["label"], r["description"] = res.get("label"), res.get("description")
        r["trigger"], r["condition"] = res.get("trigger"), res.get("condition")
        for m in r["members"]:
            mr = (ml.get(m["key"]) or {}).get("result") or {}
            m["label"], m["description"] = mr.get("label"), mr.get("description")
    meta = json.loads(DRAW.read_text(encoding="utf-8"))
    _, hinfo = hub_set(b)
    (RESULTS / "gallery.json").write_text(json.dumps({"seed": SEED, "population": meta["population"],
                                                      "run": meta["run"], "hubs": hinfo, "rows": rows},
                                                     indent=1, ensure_ascii=False), encoding="utf-8")
    import re
    num = G72.num
    # 072's tex, plus one typography fix it lacked: an opening single quote at a word start ("'by' after
    # 'replaced'") becomes a TeX open quote. Apostrophes inside words are untouched.
    tex = lambda s: re.sub(r"(^|[\s(/])'(?=[^\s'])", lambda m: m.group(1) + "`", G72.tex(s))
    run = meta["run"]
    md = [f"# Random gallery, production run (seed {SEED})", "",
          f"Run: {run['n_pass']}/{run['n_scored']} = {run['pass_rate']:.1%} pass (floor 0.75); "
          f"amplifier {run['amp_rate']:.0%}; near-threshold {run['near_threshold_rate']:.0%}. "
          f"Gallery: {sum(r['pass'] for r in rows)}/9 pass.", ""]
    for r in rows:
        flags = ["PASS" if r["pass"] else "fail"] + (["amplifier"] if r["amplifier"] else []) + \
                (["near-threshold"] if r["near_threshold"] else [])
        md += [f"## {r['key']} ({r['band']}, {r['kind']}) — {r['label']}",
               f"{', '.join(flags)}; {r['n']} latents; Z {num(r['Z'])} A {num(r['A'])} C {num(r['C'])}; "
               f"necessity {num(r['necessity'])}; induce {num(r['induce'])}; {r['hubs_dropped']} hubs dropped; "
               f"trigger {r['trigger']}, condition {r['condition']}",
               f"> {r['description']}", ""]
        md += [f"- {m['key']} (share {m['share']:.1%}, α {m['alpha']:.2f}, spec {m['specificity']:.1f}): "
               f"**{m['label']}** — {m['description']}" for m in r["members"]]
        md.append("")
    (RESULTS / "gallery.md").write_text("\n".join(md), encoding="utf-8")
    lines = []
    for r in rows:      # 072 format: no rules (main.tex puts a \midrule after every third row)
        mem = r"\newline ".join(f"{m['key']}: {tex(m['label'])}" for m in r["members"])
        lines.append(f"{r['key']} & {tex(r['label'])} & {r['n']} & {num(r['Z'])} / {num(r['A'])} / {num(r['C'])} & "
                     f"{num(r['necessity'])} & {num(r['induce'])} & {'pass' if r['pass'] else 'fail'}"
                     f"{r'$^{a}$' if r['amplifier'] else ''}{r'$^{t}$' if r['near_threshold'] else ''} & {mem} \\\\")
    (RESULTS / "gallery_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS / 'gallery.md'} and gallery_table.tex")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    stage = os.environ.get("STAGE", "all")
    for name, fn in (("draw", stage_draw), ("hubs", stage_hubs), ("members", stage_members),
                     ("labels", stage_labels), ("write", stage_write)):
        if stage in ("all", name):
            if name == "draw" and DRAW.exists() and stage == "all":
                continue
            fn()


if __name__ == "__main__":
    main()
