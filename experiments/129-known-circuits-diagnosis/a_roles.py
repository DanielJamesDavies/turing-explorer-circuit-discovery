"""129 A2 (CPU, JSON): share of total |attribution| by token role (all site kinds) per target and frame, from
out/a_mass.json (a_mass.py). Shows which positions each circuit keeps relative to the full model.

  python -I experiments/129-known-circuits-diagnosis/a_roles.py -> out/a_roles.md
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    a = json.loads((HERE / "out" / "a_mass.json").read_text())
    md = ["# 129 A2: share of |attribution| by role (117 rows; all site kinds)", ""]
    for tag in ("117",):
        bd = a[tag]["breakdown"]
        seeds = sorted({k.split("|")[0] for k in bd})
        roles = sorted({rk.split("|")[0] for v in bd.values() if "role_kind" in v for rk in v["role_kind"]})
        md += ["| target | frame | " + " | ".join(roles) + " |", "|---|---|" + "---|" * len(roles)]
        for s in seeds:
            for fr in ("full", "corpus", "task"):
                v = bd.get("%s|%s" % (s, fr))
                if not v:
                    continue
                rr = defaultdict(float)
                for rk, x in v["role_kind"].items():
                    rr[rk.split("|")[0]] += x
                md.append("| %s | %s | %s |" % (s, fr, " | ".join("%.3f" % rr.get(r, 0) for r in roles)))
    (HERE / "out" / "a_roles.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
