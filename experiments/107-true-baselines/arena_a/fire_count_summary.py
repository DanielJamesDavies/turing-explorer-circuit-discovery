"""out/panel/fire_count/fire_count.json (fire_count_tcz.py) -> fire_count.md. CPU, no torch."""
import json
import statistics as S
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent / "out" / "panel" / "fire_count"
d = json.load(open(D / "fire_count.json"))
R = d["results"]


def mr(x, f="%g"):
    return ("%s (%s–%s)" % (f, f, f)) % (x["median"], x["min"], x["max"])


L = ["# Arena A: WCM members firing on a single prompt (2026-11-05, `fire_count_tcz.py`, CPU)", "",
     "Panel circuits (v4, λ 1e-3; 9.z.* from the rerun). Per circuit: the 8 highest-peak strongest-TRAIN prompts the fit",
     "used (`rec[\"D_train\"]`, arm B for all 11) and the 8 highest-peak held-out strongest (`rec[\"strong\"][\"held\"]`),",
     "each truncated after the target's anchor (`rec arg`), one clean forward through the site model with BOS prepended",
     "and hidden from every site (the driver's `ARENA_A_BOS=1`). Counts are median (min–max) over the 8 prompts.", "",
     "- **n** = members (position-collapsed `(layer, kind, latent)`, seed excluded).",
     "- **any** = members with code > 0 at some position ≤ anchor (latents that fire on the prompt).",
     "- **(m,pos)** = (member, position) pairs with code > 0 at positions 0..anchor of the prompt. BOS is never a site",
     "  row, so it is excluded by construction; the prompt's first real token (Ge's position 1) is included.",
     "- **anchor** = members with code > 0 at the anchor.",
     "- **all** = every active latent in the target's upstream dictionaries (all real z / z2 / tc sites upstream of the",
     "  target = where members can live), latent-level / (latent, pos)-level, same prompts.",
     "- z2 (L5 / L7) = Kissane latents 24,576.., its own site; ids are disjoint, so no latent counts twice.",
     "- Every member lies in an upstream site (checked).", ""]
L += ["| target | pass | n | train any | train (m,pos) | train anchor | held any | held (m,pos) | held anchor | "
      "any / n (tr / ho) | all latents tr / ho | all (lat,pos) tr / ho | prompt len tr / ho |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in R:
    t, h = r["train"], r["held"]
    L.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %.2f / %.2f | %g / %g | %g / %g | %g / %g |" % (
        r["key"], "**PASS**" if r["passed"] else "fail", r["n_members"],
        mr(t["mem_any"]), mr(t["mem_pos"]), mr(t["mem_anchor"]), mr(h["mem_any"]), mr(h["mem_pos"]),
        mr(h["mem_anchor"]), t["mem_any"]["median"] / r["n_members"], h["mem_any"]["median"] / r["n_members"],
        t["all_any"]["median"], h["all_any"]["median"], t["all_pos"]["median"], h["all_pos"]["median"],
        t["anchor"]["median"] + 1, h["anchor"]["median"] + 1))


def over(rows, which, f):
    return S.median(r[which][f]["median"] for r in rows)


L += ["", "## Overall (median over circuits of the per-circuit medians)", "",
      "| set | circuits | n | train any | train (m,pos) | train anchor | held any | held (m,pos) | held anchor |",
      "|---|---|---|---|---|---|---|---|---|"]
for name, rows in (("all", R), ("excl. vacuous 6.z.8399", [r for r in R if r["key"] != "6.z.8399"]),
                   ("passed (3)", [r for r in R if r["passed"]])):
    L.append("| %s | %d | %g | %g | %g | %g | %g | %g | %g |" % (
        name, len(rows), S.median(r["n_members"] for r in rows), over(rows, "train", "mem_any"),
        over(rows, "train", "mem_pos"), over(rows, "train", "mem_anchor"), over(rows, "held", "mem_any"),
        over(rows, "held", "mem_pos"), over(rows, "held", "mem_anchor")))
err = max(max(r["train"]["max_rel_err"], r["held"]["max_rel_err"]) for r in R)
L += ["", "**Sanity check:** the target's code at the anchor vs the record's stored peak, all %d prompts: max relative "
      "error **%.1e** (criterion ≤ 1e-4)." % (sum(len(r["prompts"][w]) for r in R for w in r["prompts"]), err), ""]

ref = d.get("reference") or {}
if ref.get("ge"):
    L += ["## Reference: Ge / Dunefsky smoke circuits (already per prompt; different targets, K = 2 prompts each)", "",
          "Ge hierarchical (`out/ge_smoke_up/`), nodes at the τ / a_t grid point nearest 1e-3 / 1e-2 / 1e-1, BOS",
          "(position 0) rows dropped; (latent, pos) / distinct latents:", "",
          "| target | prompt | len | τ/a_t≈1e-3 | ≈1e-2 | ≈1e-1 |", "|---|---|---|---|---|---|"]
    for g in ref["ge"]:
        if "by_tau" not in g:
            continue
        b = g["by_tau"]
        L.append("| %s | %d | %d | %s | %s | %s |" % (g["key"], g["j"], g["T"], *(
            "%d / %d" % (b[t]["latent_pos"], b[t]["latents"]) for t in ("0.001", "0.01", "0.1"))))
if ref.get("dunefsky"):
    by = defaultdict(dict)
    for x in ref["dunefsky"]:
        if "num_branches" in x:
            by[(x["key"], x["j"])][x["num_branches"]] = (x["n_tc_latents"], x["n_tc_latent_token"])
    nbs = sorted({nb for v in by.values() for nb in v})
    L += ["", "Dunefsky (`out/dunefsky_smoke/`, num_iters 5): distinct tc latents (root excluded) / (latent, token) "
          "rows, by num_branches:", "", "| target | prompt | " + " | ".join("b%d" % nb for nb in nbs) + " |",
          "|---|---|" + "---|" * len(nbs)]
    for (k, j), v in sorted(by.items()):
        L.append("| %s | %d | %s |" % (k, j, " | ".join("%s / %s" % v[nb] if nb in v else "–" for nb in nbs)))
L += ["", "## Caveats", "",
      "- 6.z.8399 is the vacuous target (e0 > a_pos); its strongest prompts mostly anchor at position 0 (1-token",
      "  prompts), so its counts are not informative. 4.z.22556 anchors at positions 2-7 (3-8-token prompts), so its",
      "  (m,pos) counts are short-prompt counts.",
      "- These are clean-run firing counts, not attributions: a member that fires is not necessarily used on that prompt.",
      "- The Ge / Dunefsky smoke targets are not panel targets; they only show the per-prompt scale of those methods.",
      "  Ge's node lists may include nodes at the root's own site (as their graph does)."]
(D / "fire_count.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
