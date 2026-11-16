"""115: set up one driver OUT per held-out variant, holding COPIES of the production circuits (no refit).

062 driver.py MODE=main loads OUT/main/circuits/<key>.pt when it exists (Runner.fit) and then only scores, so an OUT
with the production circuits and the production version markers (WCM_VERSION v4, WCM_R) re-scores them unchanged.
The production OUT (T3 archive) is never written. CPU only.

  python -X utf8 experiments/115-docdisjoint-rescore-pilot/prep_rescore.py
Creates out/rs_<variant>/{WCM_VERSION, WCM_R, main/circuits/<key>.pt} for variant in orig, doc, topic.
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parents[1] / "experiments" / "109-pod-run-analysis" / "out" / "archives"
OUT_V4 = ARCH / "pub-turingllm-T3-main" / "root/turing-tl/experiments/094-tl-v2-records/out_v2"
VARIANTS = ("orig", "doc", "topic")


def main():
    keys = [k for k in (HERE / "targets_pilot.txt").read_text().split() if k]
    for v in VARIANTS:
        out = HERE / "out" / ("rs_%s" % v)
        (out / "main" / "circuits").mkdir(parents=True, exist_ok=True)
        for marker in ("WCM_VERSION", "WCM_R"):
            shutil.copyfile(OUT_V4 / marker, out / marker)
        for k in keys:
            dst = out / "main" / "circuits" / ("%s.pt" % k)
            if not dst.exists():
                shutil.copyfile(OUT_V4 / "main" / "circuits" / ("%s.pt" % k), dst)
        print("%s: %d circuits" % (out, len(list((out / "main" / "circuits").glob("*.pt")))))


if __name__ == "__main__":
    main()
