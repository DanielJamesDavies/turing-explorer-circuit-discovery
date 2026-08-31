"""Collect every comparison arm from the three arenas into one tidy
points.jsonl for the frontier figure. One row per (arena, method, seed,
size-variant): {arena, family, method, seed, n, f0, fm, sup, cf}.

Key mapping: cross-architecture rows use ampF0/ampFM; the home matrix
uses free0/freeM/cf; the Ge replication uses ampF0/ampFMd/cf_amp. All
are the zero-fill / mean-fill faithfulness and drive of the same exam
family; the caption states the naming.

Families: OURS (tri-amp / tri-mask, lambda-sweep curves), EXT (external
methods as shipped: circuit-tracer direct-edge ranking, circuit-tracer
pruned circuits, SFC attribution patching, home attribution arms, Ge et
al.), HYB (external selection + our fitted amplitudes), LAD (nulls,
support-matched nulls, coact).

  python collect.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
E = HERE.parent
OUT = HERE / "points.jsonl"


def emit(fh, arena, family, method, seed, n, f0, fm, sup, cf):
    if f0 is None or n is None or n == 0:
        return
    fh.write(json.dumps({"arena": arena, "family": family, "method": method,
                         "seed": seed, "n": int(n), "f0": f0, "fm": fm,
                         "sup": sup, "cf": cf}) + "\n")


def cross_arch(fh, arena, rows_file):
    for line in open(rows_file):
        r = json.loads(line)
        arm = r.get("arm", "")
        if arm == "anchor_support":
            continue
        seed = "L%d/%d" % (r["layer"], r["latent"])
        args = (r["n"], r.get("ampF0"), r.get("ampFM"), r.get("sup"), r.get("cf_amp"))
        if arm.startswith("triamp400"):
            emit(fh, arena, "OURS", "tri-amp", seed, *args)
        elif arm.startswith("gate400"):
            emit(fh, arena, "OURS", "tri-mask", seed, *args)
        elif arm == "sfc" or arm.startswith("sfc_x") or arm == "sfc_full":
            emit(fh, arena, "EXT", "sfc", seed, *args)
        elif arm == "sfc_amp":
            emit(fh, arena, "HYB", "sfc+amp", seed, *args)
        elif arm == "ct_seed_rooted_matched_amp":
            emit(fh, arena, "HYB", "ct-rooted+amp", seed, *args)
        elif arm == "theirs" or arm.startswith("theirs_x") or arm == "theirs_full":
            emit(fh, arena, "EXT", "ct-direct", seed, *args)
        elif arm.startswith("ct_published"):
            emit(fh, arena, "EXT", "ct-published", seed, *args)
        elif arm.startswith("ct_seed_pinned"):
            continue   # == published within noise; stated in text
        elif arm.startswith("ct_seed_rooted"):
            emit(fh, arena, "EXT", "ct-rooted", seed, *args)
        elif arm.startswith("nullsup"):
            emit(fh, arena, "LAD", "support-null", seed, *args)
        elif arm.startswith("null"):
            emit(fh, arena, "LAD", "null", seed, *args)
        elif arm == "coact_raw":
            emit(fh, arena, "LAD", "coact", seed, *args)
        elif arm == "coact_amp":
            emit(fh, arena, "LAD", "coact+amp", seed, *args)


def home(fh):
    # ours + nulls from the 22-seed panel
    for line in open(E / "029-panel/rows.jsonl"):
        r = json.loads(line)
        seed = "c%d/%d" % (r["comp_idx"], r["latent"])
        args = (r.get("n"), r.get("ampF0"), r.get("ampFMd"), r.get("sup"), r.get("cf_amp"))
        if r["arm"] in ("triamp400", "triamp100"):
            emit(fh, "turingllm", "OURS", "tri-amp", seed, *args)
        elif r["arm"] == "gate400":
            emit(fh, "turingllm", "OURS", "tri-mask", seed, *args)
        elif r["arm"].startswith("null"):
            emit(fh, "turingllm", "LAD", "null", seed, *args)
    # external attribution arms from the definitive matrix
    for line in open(E / "031-matrix/rows.jsonl"):
        r = json.loads(line)
        seed = "c%d/%d" % (r["comp_idx"], r["latent"])
        base = r["arm"].split("@")[0]
        name = {"abl_ig": "abl-gradient", "cf_ig": "cf-gradient",
                "resto": "restoration"}.get(base, base)
        emit(fh, "turingllm", "EXT", name, seed,
             r.get("n"), r.get("free0"), r.get("freeM"), r.get("sup"), r.get("cf"))
    # Ge et al. hierarchical attribution
    for line in open(E / "034-ge-replication/rows.jsonl"):
        r = json.loads(line)
        seed = "c%d/%d" % (r["comp_idx"], r["latent"])
        emit(fh, "turingllm", "EXT", "ge-hier", seed,
             r.get("n"), r.get("ampF0"), r.get("ampFMd"), r.get("sup"), r.get("cf_amp"))


def main():
    fh = OUT.open("w")
    cross_arch(fh, "gemma-tc", E / "038-transcoder-compare-gemma/ours_gtc_rows.jsonl")
    cross_arch(fh, "llama-tc", E / "035-transcoder-compare/ours_llama_rows.jsonl")
    home(fh)
    fh.close()
    import collections
    c = collections.Counter()
    for line in open(OUT):
        r = json.loads(line)
        c[(r["arena"], r["family"], r["method"])] += 1
    for k in sorted(c):
        print("%-12s %-5s %-14s %3d points" % (k[0], k[1], k[2], c[k]))


if __name__ == "__main__":
    main()
