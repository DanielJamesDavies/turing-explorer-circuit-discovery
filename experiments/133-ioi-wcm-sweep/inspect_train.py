"""Print the per-term training curves of 081 out_v4 IOI fits (CPU)."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "experiments/081-task-circuits/out_v4/gpt2-small/train.jsonl"
task = sys.argv[2] if len(sys.argv) > 2 else "ioi"
for line in open(path):
    r = json.loads(line)
    if r.get("task") != task:
        continue
    ks = [k for k in r if k.startswith("curve")]
    print(r["arm"], r["lam"], r["draw"], "n", r.get("n_kept"), "L0", r.get("loss_initial"), "L1", r.get("loss_final"),
          "scale", r.get("logit_scale"), "floor", r.get("dual_norm_floor"), "m", r.get("mean_m_kept"),
          "amp", (r.get("amp_stats") or {}).get("median"))
    for k in ks:
        v = r[k]
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, list):
                    print("   ", k, kk, [round(x, 3) if isinstance(x, float) else x for x in vv[::25]])
        elif isinstance(v, list):
            print("   ", k, [round(x, 3) if isinstance(x, float) else x for x in v[::25]])
    print("    keys", [k for k in r if k not in ks])
