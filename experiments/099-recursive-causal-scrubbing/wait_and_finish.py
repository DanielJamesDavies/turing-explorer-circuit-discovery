"""Sequential dependency gate for this active run, not a scheduled monitor."""
import json
from pathlib import Path
import time
from run_baseline import HERE


def main():
    check = HERE / "out/panel18/source_check.json"
    end = time.monotonic() + 3600
    print("Waiting for the frozen panel's successful source/record completion gate.", flush=True)
    while not check.exists():
        if time.monotonic() > end:
            raise TimeoutError("panel did not complete within one hour")
        time.sleep(5)
    if json.loads(check.read_text())["changed_during_run"]:
        raise RuntimeError("panel source changed; diagnostics not started")
    manifest = json.loads((check.parent / "manifest.json").read_text())
    if any(not (check.parent / "targets" / (key + ".json")).exists() for key in manifest["targets"]):
        raise RuntimeError("panel incomplete; diagnostics not started")
    from finish_diagnostics import main as finish
    finish()


if __name__ == "__main__":
    main()
