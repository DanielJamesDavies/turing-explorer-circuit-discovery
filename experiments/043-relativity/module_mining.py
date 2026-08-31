"""MODULE MINING across audited circuits: which latents recur as
causal members of MULTIPLE seeds, and in what blocks?

Builds the latent x seed edge-weight matrix from every edge-audit
jsonl present (|necessity| as weight), then reports:
  * INFRASTRUCTURE: latents with weight in >= 3 circuits spanning
    different topics -- arena machinery, to be subtracted from any
    concept reading;
  * MODULES: latents shared by >= 2 circuits, grouped by their
    seed-signature (which circuits they serve) -- candidate reusable
    knowledge blocks (e.g. a cosmology block serving boson AND
    relativity);
  * PRIVATE: per-seed count of members serving only that seed.

CPU-only; reads edge_audit_*.jsonl files.

  python module_mining.py
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
AUDITS = {                       # file -> (seed name, topic family)
    "edge_audit_c35_13633.jsonl": ("boson", "physics"),
    "edge_audit_c29_3736.jsonl": ("relativity", "physics"),
    "edge_audit_c20_23753.jsonl": ("rel-corrections", "physics"),
    "edge_audit_c11_18699.jsonl": ("einstein-phrase", "physics"),
    "edge_audit_c29_3736_know400.jsonl": ("relativity-know", "physics"),
    "edge_audit_c35_13633_know400.jsonl": ("boson-know", "physics"),
}
THRESH = 0.02                    # |necessity| floor for "carries weight"


def main():
    W = defaultdict(dict)        # (site, latent) -> {seed: weight}
    seeds = []
    for fn, (name, _fam) in AUDITS.items():
        p = HERE / fn
        if not p.exists():
            continue
        seeds.append(name)
        for line in open(p):
            r = json.loads(line)
            if "necessity" not in r:
                continue
            w = abs(r["necessity"])
            if w >= THRESH:
                W[(r["site"], r["latent"])][name] = w

    multi = {k: v for k, v in W.items() if len(v) >= 2}
    # collapse know-arm duplicates of the same base seed for span counts
    def span(v):
        return {s.replace("-know", "") for s in v}

    infra = {k: v for k, v in multi.items() if len(span(v)) >= 3}
    print("audits loaded: %s" % ", ".join(seeds))
    print("\n== INFRASTRUCTURE (weight >= %.2f in >= 3 distinct circuits)"
          % THRESH)
    print("%-10s %-7s %5s  %s" % ("site", "latent", "total", "serves"))
    for k, v in sorted(infra.items(), key=lambda kv: -sum(kv[1].values())):
        print("%-10s %-7d %5.2f  %s"
              % (k[0], k[1], sum(v.values()),
                 ", ".join("%s:%.2f" % (s, w) for s, w in
                           sorted(v.items(), key=lambda x: -x[1]))))

    print("\n== SHARED MODULES (2 distinct circuits, not infrastructure)")
    groups = defaultdict(list)
    for k, v in multi.items():
        sp = frozenset(span(v))
        if len(sp) == 2:
            groups[sp].append((sum(v.values()), k, v))
    for sp, members in sorted(groups.items(),
                              key=lambda kv: -len(kv[1])):
        print("\n-- serves {%s}: %d shared members"
              % (", ".join(sorted(sp)), len(members)))
        for tot, k, v in sorted(members, reverse=True)[:8]:
            print("   %-10s %-7d total %.2f" % (k[0], k[1], tot))

    print("\n== PRIVATE MEMBER COUNTS (weight >= %.2f, single circuit)"
          % THRESH)
    per = defaultdict(int)
    for k, v in W.items():
        if len(v) == 1:
            per[list(v)[0]] += 1
    for s in seeds:
        print("   %-16s %d private weighted members" % (s, per.get(s, 0)))


if __name__ == "__main__":
    main()
