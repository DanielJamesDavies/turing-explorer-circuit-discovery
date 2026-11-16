"""Print the structure of one or more .pt files (circuits, ctx records). Usage: python -I -X utf8 inspect_pt.py F..."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch  # noqa: E402


def show(o, ind=0, depth=0):
    p = " " * ind
    if hasattr(o, "__dict__") and not isinstance(o, (dict, list)) and not torch.is_tensor(o):
        print(p + "OBJ", type(o))
        o = vars(o)
    if isinstance(o, dict):
        for k, v in list(o.items())[:60]:
            if torch.is_tensor(v):
                print(p + str(k) + ": tensor", tuple(v.shape), v.dtype, v.flatten()[:6].tolist())
            elif isinstance(v, (dict, list, tuple)) and depth < 3:
                print(p + str(k) + ":", type(v).__name__, len(v))
                show(v, ind + 2, depth + 1)
            elif hasattr(v, "__dict__") and depth < 3:
                print(p + str(k) + ": OBJ", type(v))
                show(v, ind + 2, depth + 1)
            else:
                print(p + str(k) + ":", repr(v)[:200])
    elif isinstance(o, (list, tuple)):
        for v in o[:3]:
            if torch.is_tensor(v):
                print(p + "tensor", tuple(v.shape), v.dtype, v.flatten()[:6].tolist())
            elif isinstance(v, (dict, list, tuple)) or hasattr(v, "__dict__"):
                print(p + "-")
                show(v, ind + 2, depth + 1)
            else:
                print(p + repr(v)[:200])


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print("=====", f)
        show(torch.load(f, map_location="cpu", weights_only=False))
