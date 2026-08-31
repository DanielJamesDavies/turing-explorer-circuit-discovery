"""Patch theirs_gtc.py: after each attribute(), run circuit-tracer's own
pruning (as published, logit-rooted) AND the seed-rooted variant, and
record the seed's circuit from each. The direct-edge ranking is kept as
a secondary output (appendix size-sweep only; it no longer carries the
circuit-tracer name)."""
from pathlib import Path

p = Path(__file__).parent / "theirs_gtc.py"
s = p.read_text(encoding="utf-8")


def rep(old, new):
    global s
    assert s.count(old) == 1, ("ANCHOR", old[:70], s.count(old))
    s = s.replace(old, new)


rep("from collections import defaultdict\n",
    "from collections import Counter, defaultdict\n")
rep("def main():\n    from circuit_tracer import attribute\n",
    "def main():\n    from circuit_tracer import attribute\n\n"
    "    import ct_prune\n")
rep("        weight = defaultdict(float)\n        n_graphs, consec, no_row = 0, 0, 0\n",
    "        weight = defaultdict(float)\n        n_graphs, consec, no_row = 0, 0, 0\n"
    "        # circuit-tracer pruning, per window: as published (logit root)\n"
    "        # and seed-rooted. Survivors are the seed's ANCESTORS in the\n"
    "        # pruned edge graph. Counted across windows for the survival-\n"
    "        # frequency ranking; union is the most generous single set.\n"
    "        pub_count, root_count = Counter(), Counter()\n"
    "        pub_seed_alive, root_seed_alive, pub_sizes, root_sizes = 0, 0, [], []\n")
rep("            contrib = adj[tgt_rows][:, :n_sel][:, src_ok].abs().sum(0)\n",
    "            # --- THEIR pruning (nothing of ours until seed_circuit) ---\n"
    "            pr = ct_prune.prune_published(g)\n"
    "            alive, mem, _ = ct_prune.seed_circuit(g, pr, L, sl)\n"
    "            if alive:\n"
    "                pub_seed_alive += 1\n"
    "                pub_sizes.append(len(mem))\n"
    "                pub_count.update(mem)\n"
    "            # --- their body, root = the seed's rows (labelled adaptation) ---\n"
    "            rows_s, _ = ct_prune.seed_rows(g, L, sl)\n"
    "            pr2 = ct_prune.prune_rooted(g, rows_s)\n"
    "            alive2, mem2, _ = ct_prune.seed_circuit(g, pr2, L, sl)\n"
    "            if alive2:\n"
    "                root_seed_alive += 1\n"
    "                root_sizes.append(len(mem2))\n"
    "                root_count.update(mem2)\n"
    "            del pr, pr2\n"
    "            contrib = adj[tgt_rows][:, :n_sel][:, src_ok].abs().sum(0)\n")
rep('''               "ranking": [[l, f, round(w, 6)] for (l, f), w in ranked[:20000]]}''',
    '''               "ranking": [[l, f, round(w, 6)] for (l, f), w in ranked[:20000]],
               # circuit-tracer AS PUBLISHED (prune_graph 0.8/0.98, logit root)
               "ct_published": {
                   "seed_alive_windows": pub_seed_alive,
                   "size_per_window": pub_sizes,
                   "union": [[l, f] for (l, f) in sorted(pub_count)],
                   "freq": [[l, f, c] for (l, f), c in pub_count.most_common()]},
               # their pruning body, rooted at the seed (adaptation)
               "ct_seed_rooted": {
                   "seed_alive_windows": root_seed_alive,
                   "size_per_window": root_sizes,
                   "union": [[l, f] for (l, f) in sorted(root_count)],
                   "freq": [[l, f, c] for (l, f), c in root_count.most_common()]}}''')
rep('''        print("[L%d %d] %d graphs (%d prompts had no seed row) | %d distinct "
              "upstream features | exported top-%s"
              % (L, sl, n_graphs, no_row, len(ranked), n_ref or "?"), flush=True)''',
    '''        print("[L%d %d] %d graphs (%d prompts had no seed row) | %d distinct "
              "upstream features | exported top-%s"
              % (L, sl, n_graphs, no_row, len(ranked), n_ref or "?"), flush=True)
        print("   ct_published: seed alive in %d/%d windows | circuit size/window "
              "med %s | union %d\\n   ct_seed_rooted: alive %d/%d | size med %s | union %d"
              % (pub_seed_alive, n_graphs,
                 sorted(pub_sizes)[len(pub_sizes) // 2] if pub_sizes else None,
                 len(pub_count), root_seed_alive, n_graphs,
                 sorted(root_sizes)[len(root_sizes) // 2] if root_sizes else None,
                 len(root_count)), flush=True)''')
rep('OUT = Path(os.environ.get("OUT", str(HERE / "theirs_gtc_nodes.jsonl")))',
    'OUT = Path(os.environ.get("OUT", str(HERE / "theirs_gtc_pruned.jsonl")))')
p.write_text(s, encoding="utf-8", newline="")
import ast; ast.parse(s)
print("theirs_gtc.py patched: pruning per window, output ->", "theirs_gtc_pruned.jsonl")
