"""Copy the context records of a target list into a small directory (byte copies, never rewritten), so a donor test's
R-pool filter reads only those records. Skips files already present.

  python experiments/104-v3-cross-model/subset_records.py <src ctx dir> <targets.txt> <dst dir>
"""
import shutil
import sys
from pathlib import Path

src, targets, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
keys = [k for k in targets.read_text().split() if k]
for k in keys:
    if not (dst / ("%s.pt" % k)).exists():
        shutil.copyfile(src / ("%s.pt" % k), dst / ("%s.pt" % k))
print("%d records in %s" % (len(list(dst.glob("*.pt"))), dst))
