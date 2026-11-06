"""Freeze availability-based development targets and seeded check targets.

No WCM or scrubbing scores are read. This is a balanced availability pilot,
not a random population sample. Copy verified source records without editing.
"""
import hashlib
import json
from pathlib import Path
import random
import shutil

HERE = Path(__file__).parent
ROOT = HERE.parent.parent


def cell(key):
    layer, kind, _ = key.split(".")
    return (0 if int(layer) <= 4 else 1 if int(layer) <= 7 else 2), kind


def main():
    source = ROOT / "experiments/094-tl-v2-records"
    smoke40 = ROOT / "experiments/086-heldout-rebuild/targets_smoke40.txt"
    first10 = (source / "targets_smoke10.txt").read_text().split()
    candidates = smoke40.read_text().split()
    chosen, development, check = [], [], []
    rng = random.Random(20261028)
    for band in range(3):
        for kind in ("attn", "mlp", "resid"):
            available = [k for k in first10 if cell(k) == (band, kind)]
            development.append(available[0])
            if len(available) > 1:
                check.append(available[1])
            else:
                remaining = sorted(k for k in candidates if cell(k) == (band, kind) and k not in first10)
                check.append(rng.choice(remaining))
            chosen.extend([development[-1], check[-1]])
    data = dict(seed=20261028, rule="first smoke target per cell development; existing second smoke target or seeded smoke40 draw check",
                no_scores_used=True, population_claim=False, development=development, check=check,
                targets=chosen, frame_sha256=hashlib.sha256(smoke40.read_bytes()).hexdigest())
    path = HERE / "panel.json"
    if path.exists() and json.loads(path.read_text()) != data:
        raise RuntimeError("panel already frozen differently")
    path.write_text(json.dumps(data, indent=2) + "\n")
    (HERE / "targets_panel18.txt").write_text("\n".join(chosen) + "\n")
    (HERE / "targets_development9.txt").write_text("\n".join(development) + "\n")
    (HERE / "targets_check9.txt").write_text("\n".join(check) + "\n")
    destination = HERE / "out/records/ctx_v2"
    destination.mkdir(parents=True, exist_ok=True)
    for key in chosen:
        old = source / "out/smoke10/ctx_v2" / (key + ".pt")
        if old.exists():
            new = destination / old.name
            if new.exists() and new.read_bytes() != old.read_bytes():
                raise RuntimeError("different record already exists: " + str(new))
            if not new.exists():
                shutil.copyfile(old, new)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
