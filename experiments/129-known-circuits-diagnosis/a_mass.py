"""129 A (CPU, JSON only): re-read 117 / 120 structure rows (out/wcm/dan215/structure.jsonl, read-only).

Per (target, frame): where the attribution mass sits (site kind x role x layer band), how the pre-stated head-class
rule treats the FULL MODEL itself (positive control scored as if it were a circuit, against the same circuit's null),
and looser readings of the head checks (1x base, any position, rank of the best published head).

  python -I experiments/129-known-circuits-diagnosis/a_mass.py  -> out/a_mass.json, out/a_mass.md
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SRC = {"117": EXP / "117-known-circuits-gpt2" / "out" / "wcm" / "dan215" / "structure.jsonl",
       "120": EXP / "120-known-circuits-errors-zeroed" / "out" / "wcm" / "dan215" / "structure.jsonl"}
N_HEADS = 12

def _hs(*xs):
    return [tuple(int(v) for v in x.split(".")) for x in xs]

# copied from 117 lib117.CLASSES (not imported: lib117 adds src paths and torch-only modules)
CLASSES = {
    "name_mover": dict(heads=_hs("9.6", "9.9", "10.0"), role="end"),
    "s_inhibition": dict(heads=_hs("7.3", "7.9", "8.6", "8.10"), role="end"),
    "induction": dict(heads=_hs("5.5", "5.8", "5.9", "6.9"), role="s2"),
    "duplicate_token": dict(heads=_hs("0.1", "0.10", "3.0"), role="s2"),
    "previous_token": dict(heads=_hs("2.2", "4.11"), role="s1_next"),
    "gt_heads": dict(heads=_hs("5.1", "5.5", "6.1", "6.9", "7.10", "8.8", "8.11", "9.1"), role="end"),
}
EXPECT = {
    "IOI-NM": ["s_inhibition", "induction", "duplicate_token", "previous_token", "name_mover"],
    "IOI-SI": ["induction", "duplicate_token", "previous_token"],
    "IOI-IND": ["duplicate_token", "previous_token", "induction"],
    "GT-MLP": ["gt_heads"],
}


def parse(k):
    l, kd, i = k.split(".")
    return int(l), kd, int(i)


def in_scope(heads, layer, kind):
    return [(l, h) for (l, h) in heads if (l < layer or (kind == "mlp" and l == layer))]


def base_rate(chk, layer, kind):
    hs = in_scope(CLASSES[chk]["heads"], layer, kind)
    return len(hs) / (N_HEADS * (layer + (1 if kind == "mlp" else 0))) if hs else None


def mass_of(r):
    return {(int(a), str(b), int(c), str(d)): float(v) for a, b, c, d, v in r["mass"]}


def head_share(m, chk, layer, kind, any_pos=False):
    hs = set(in_scope(CLASSES[chk]["heads"], layer, kind))
    if not hs:
        return None
    att = sum(v for (l, k, h, r), v in m.items() if k == "attn" and h >= 0)
    v = sum(x for (l, k, h, r), x in m.items() if k == "attn" and (l, h) in hs and (any_pos or r == CLASSES[chk]["role"]))
    return v / att if att > 0 else float("nan")


def best_rank(m, chk, layer, kind):
    """Rank (1 = top) of the best class head at its role among all (head, role) attention cells."""
    hs = set(in_scope(CLASSES[chk]["heads"], layer, kind))
    if not hs:
        return None
    cells = defaultdict(float)
    for (l, k, h, r), v in m.items():
        if k == "attn" and h >= 0:
            cells[(l, h, r)] += v
    order = sorted(cells, key=lambda c: -cells[c])
    for i, c in enumerate(order):
        if (c[0], c[1]) in hs and c[2] == CLASSES[chk]["role"]:
            return i + 1
    return None


def breakdown(m):
    tot = sum(m.values()) or 1.0
    kind = defaultdict(float); role_kind = defaultdict(float); band = defaultdict(float); attn_band = defaultdict(float)
    for (l, k, h, r), v in m.items():
        kind[k] += v / tot
        role_kind["%s|%s" % (r, k)] += v / tot
        b = "L0-2" if l <= 2 else ("L3-5" if l <= 5 else "L6+")
        band["%s|%s" % (b, k)] += v / tot
        if k == "attn":
            attn_band[b] += v
    at = sum(attn_band.values()) or 1.0
    return dict(kind=dict(kind), role_kind=dict(role_kind), band=dict(band),
                attn_band={b: v / at for b, v in attn_band.items()}, total=tot)


def rule(s, nulls, br):
    import statistics
    ns = [x for x in nulls if x is not None and x == x]
    if s is None or not ns:
        return None
    mu = statistics.mean(ns)
    p95 = sorted(ns)[min(len(ns) - 1, int(round(0.95 * (len(ns) - 1))))]
    return bool(s >= 2 * mu and s > p95 and (br is None or s >= 2 * br))


def main():
    out = {}
    md = ["# 129 A: attribution-mass re-read of 117 / 120 structure rows", ""]
    for tag, p in SRC.items():
        by = defaultdict(list)
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            by[(r["seed"], r["circuit"], r["frame"])].append(r)
        seeds = sorted({k[0] for k in by})
        res = dict(breakdown={}, full_as_circuit=[], loose=[])
        for seed in seeds:
            layer, kind, _ = parse(seed)
            pw = by[(seed, "-", "full")][0]["pathway"]
            fm = mass_of(by[(seed, "-", "full")][0])
            res["breakdown"]["%s|full" % seed] = breakdown(fm)
            for cname in ("corpus", "task"):
                cr = by.get((seed, cname, "circuit"))
                if not cr:
                    continue
                cm = mass_of(cr[0])
                res["breakdown"]["%s|%s" % (seed, cname)] = breakdown(cm)
                nm = [mass_of(x) for x in by.get((seed, cname, "null"), [])]
                res["breakdown"]["%s|%s-null-mean" % (seed, cname)] = {
                    "kind": {k: sum(breakdown(m)["kind"].get(k, 0) for m in nm) / max(1, len(nm))
                             for k in ("attn", "mlp", "resid_mid", "resid")}}
                for chk in EXPECT[pw]:
                    br = base_rate(chk, layer, kind)
                    if br is None:
                        continue
                    nulls = [head_share(m, chk, layer, kind) for m in nm]
                    fs = head_share(fm, chk, layer, kind)
                    cs = head_share(cm, chk, layer, kind)
                    res["full_as_circuit"].append(dict(seed=seed, circuit=cname, check=chk, full=fs, circ=cs, base=br,
                                                       full_pass=rule(fs, nulls, br), circ_pass=rule(cs, nulls, br),
                                                       full_pass_no_base=rule(fs, nulls, None),
                                                       circ_pass_no_base=rule(cs, nulls, None)))
                    nulls_any = [head_share(m, chk, layer, kind, True) for m in nm]
                    res["loose"].append(dict(
                        seed=seed, circuit=cname, check=chk, base=br,
                        circ_role=cs, full_role=fs,
                        circ_any=head_share(cm, chk, layer, kind, True), full_any=head_share(fm, chk, layer, kind, True),
                        circ_any_pass_1x=rule(head_share(cm, chk, layer, kind, True), nulls_any, None) and
                        head_share(cm, chk, layer, kind, True) >= br,
                        circ_role_ge_base=cs is not None and cs >= br,
                        circ_rank=best_rank(cm, chk, layer, kind), full_rank=best_rank(fm, chk, layer, kind),
                        n_cells=sum(1 for (l, k, h, r) in cm if k == "attn" and h >= 0)))
        out[tag] = res
        # ---- markdown summary
        fa = res["full_as_circuit"]
        md += ["## %s" % tag, "",
               "Full model scored as if it were a circuit, by 117's pre-stated head rule (vs the same circuit's 20 "
               "nulls; 2x null mean, > p95, >= 2x uniform base):",
               "- full model passes %d / %d head checks (without the 2x-base clause: %d / %d); circuits %d / %d (%d / %d)"
               % (sum(bool(x["full_pass"]) for x in fa), len(fa), sum(bool(x["full_pass_no_base"]) for x in fa),
                  len(fa), sum(bool(x["circ_pass"]) for x in fa), len(fa), sum(bool(x["circ_pass_no_base"]) for x in fa),
                  len(fa)),
               "- full model share >= 2x base: %d / %d; >= 1x base: %d / %d"
               % (sum(1 for x in fa if x["full"] is not None and x["full"] >= 2 * x["base"]), len(fa),
                  sum(1 for x in fa if x["full"] is not None and x["full"] >= x["base"]), len(fa)), ""]
        by_chk = defaultdict(list)
        for x in fa:
            by_chk[x["check"]].append(x)
        md += ["| check | n | full >= 2x base | full >= 1x base | full passes rule | circuit passes rule | median full/base |",
               "|---|---|---|---|---|---|---|"]
        for c, xs in sorted(by_chk.items()):
            rs = sorted(x["full"] / x["base"] for x in xs if x["full"] is not None)
            md.append("| %s | %d | %d | %d | %d | %d | %.2f |" % (
                c, len(xs), sum(1 for x in xs if x["full"] >= 2 * x["base"]), sum(1 for x in xs if x["full"] >= x["base"]),
                sum(bool(x["full_pass"]) for x in xs), sum(bool(x["circ_pass"]) for x in xs), rs[len(rs) // 2]))
        lo = res["loose"]
        md += ["", "Looser readings of the circuits' head checks: any position, >= 1x base and > 2x null mean & > p95: "
               "%d / %d; role-position share >= 1x base: %d / %d. Best published head rank (median): circuit %s, "
               "full model %s." % (
                   sum(bool(x["circ_any_pass_1x"]) for x in lo), len(lo), sum(bool(x["circ_role_ge_base"]) for x in lo),
                   len(lo), sorted(x["circ_rank"] or 999 for x in lo)[len(lo) // 2],
                   sorted(x["full_rank"] or 999 for x in lo)[len(lo) // 2]), ""]
        # kind breakdown
        md += ["Attribution mass by site kind (share of total |attr|), median over targets:", "",
               "| frame | attn | resid_mid | mlp | resid | attn mass in L0-2 (share of attn) |", "|---|---|---|---|---|---|"]
        for fr in ("full", "corpus", "task"):
            ks = [v for k, v in res["breakdown"].items() if k.endswith("|" + fr)]
            def med(f):
                xs = sorted(f(v) for v in ks)
                return xs[len(xs) // 2] if xs else float("nan")
            md.append("| %s | %.3f | %.3f | %.3f | %.3f | %.2f |" % (
                fr, med(lambda v: v["kind"].get("attn", 0)), med(lambda v: v["kind"].get("resid_mid", 0)),
                med(lambda v: v["kind"].get("mlp", 0)), med(lambda v: v["kind"].get("resid", 0)),
                med(lambda v: v["attn_band"].get("L0-2", 0))))
        md.append("")
    (HERE / "out" / "a_mass.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (HERE / "out" / "a_mass.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
