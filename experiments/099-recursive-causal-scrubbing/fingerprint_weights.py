"""Content fingerprints for the fixed TuringLLM checkpoint and SAE bank."""
import json
from pathlib import Path
import time
from run_baseline import HERE, ROOT, file_hash, write_json


def main():
    paths = [ROOT / "models/TuringLLM/model_1722550239_03986.pt"]
    paths += sorted((ROOT / "models/TuringLLM/SAE").glob("sae-*/*.pth"))
    result = {}
    start = time.time()
    for p in paths:
        before = p.stat()
        digest = file_hash(p)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("weight changed while hashing: " + str(p))
        result[str(p.relative_to(ROOT))] = dict(bytes=after.st_size, modified_ns=after.st_mtime_ns, sha256=digest)
        write_json(HERE / "out/weight_fingerprints.json", result)
    print(json.dumps(dict(files=len(paths), bytes=sum(v["bytes"] for v in result.values()),
                          seconds=round(time.time() - start, 1))))


if __name__ == "__main__":
    main()
