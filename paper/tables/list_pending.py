"""List every placeholder in paper/main.tex: each \\pending{...} and \\pendingcell, with line, section and text.

Written 2026-10-27 (DAN-152). Comment text (after an unescaped %) is ignored, so only placeholders that reach the PDF
are listed. A \\pendingcell inside a row macro (\\newcommand{\\mrrow}{...}) is listed once at its definition, with the
number of rows that use the macro, so the printed cell count is the count that reaches the PDF.

Run from the repo root: python -X utf8 paper/tables/list_pending.py [--tex paper/main.tex] [--markdown]
--markdown prints a table for paper/resources/pending-checklist.md; the default is one line per placeholder.
The group column is a keyword guess at what fills the placeholder; check it by eye.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SECTION_RE = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?\{")
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
CAPTION_RE = re.compile(r"\\caption\{")
PENDING = "\\pending{"
CELL = "\\pendingcell"
ROWMACRO_RE = re.compile(r"\\newcommand\{\\(\w+)\}\{")

# (group, keywords) in priority order; the first group whose keyword occurs in the text or the section wins.
GROUPS = [
    ("DAN-185 info from Daniel", ["DAN-185"]),
    ("Public models (Pythia-70m, GPT-2 small)", ["DAN-40", "DAN-41", "DAN-42", "Pythia", "public model"]),
    ("Harder nulls", ["DAN-181", "harder null", "harder random", "anchor-active"]),
    ("Method comparison (DAN-159 / DAN-77 / DAN-43)", ["DAN-159", "DAN-77", "DAN-43", "083", "tab:matrix",
                                                       "fig:size-comparison", "app:method-comparison"]),
    ("Lambda sweep (190 targets)", ["v2 sweep", "sweep", "sec:weighted", "app:weighted-details"]),
    ("Case study", ["case-study", "case study"]),
    ("Resample re-scores (095 / DAN-184)", ["resample", "DAN-184"]),
    ("Output-targeted rerun (DAN-48)", ["DAN-48"]),
]
# The main 8k run goes first, so a placeholder that names it and a later group is listed under the 8k run.
GROUPS.insert(0, ("Main 8k run (headline, both reads, held-out)", ["DAN-157", "8k run", "pre-activation",
                                                                   "tab:main-result", "app:production", "app:heldout",
                                                                   "sec:main-result", "headline"]))


@dataclass
class Item:
    line: int
    kind: str  # "pending" or "cell"
    section: str
    text: str
    rows: int = 1  # rows that use the macro the cell sits in (1 outside a row macro)
    cells: int = 0  # \pendingcell occurrences on this source line (cells only)

    def groups(self) -> list[str]:
        """Every group whose keywords occur in the placeholder's own text; if none, the section decides."""
        for hay in (self.text, self.section):
            hit = [name for name, keys in GROUPS if any(k.lower() in hay.lower() for k in keys)]
            if hit:
                return hit
        return ["Other"]

    def group(self) -> str:
        return self.groups()[0]


def strip_comment(line: str) -> str:
    """Drop everything from the first unescaped %."""
    out, i = [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if c == "%":
            break
        out.append(c)
        i += 1
    return "".join(out)


def braced(text: str, start: int) -> tuple[str, int]:
    """Return the contents of the brace group whose '{' is at text[start - 1], and the index after its '}'."""
    depth, i = 1, start
    while i < len(text) and depth:
        if text[i] == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[start:i - 1], i


def scan(tex: Path) -> list[Item]:
    lines = [strip_comment(l) for l in tex.read_text(encoding="utf-8").splitlines()]
    body = "\n".join(lines)
    starts = [0]
    for l in lines:
        starts.append(starts[-1] + len(l) + 1)

    def line_of(pos: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    # Section context: the latest heading, plus the nearest label after it and the caption a placeholder sits in.
    heads = []
    for m in SECTION_RE.finditer(body):
        title, end = braced(body, m.end())
        heads.append((m.start(), m.group(1), " ".join(title.split()), end))
    for m in re.finditer(r"\\begin\{abstract\}", body):
        heads.append((m.start(), "section", "Abstract", m.end()))
    heads.sort()
    labels = [(m.start(), m.group(1)) for m in LABEL_RE.finditer(body)]
    captions = []
    for m in CAPTION_RE.finditer(body):
        _, end = braced(body, m.end())
        captions.append((m.start(), end))
    floats = []  # (start, end, label) of table / figure / longtable environments
    for m in re.finditer(r"\\begin\{(table|figure|longtable)\*?\}", body):
        end = body.find("\\end{" + m.group(1), m.end())
        lab = next((l for p, l in labels if m.start() <= p <= end), "")
        floats.append((m.start(), end, lab))

    def context(pos: int) -> str:
        sec = next((h for h in reversed(heads) if h[0] <= pos and h[1] != "paragraph"), None)
        name = sec[2] if sec else "(front matter)"
        sec_start = sec[0] if sec else 0
        par = next((h for h in reversed(heads) if sec_start < h[0] <= pos and h[1] == "paragraph"), None)
        # A section's own label sits right after its heading.
        lab = next((l for p, l in labels if sec and sec[3] <= p <= sec[3] + 40), "")
        bits = [name + (f" [{lab}]" if lab else "")]
        if par:
            bits.append(f"para: {par[2]}")
        cap = next((c for c in captions if c[0] <= pos < c[1]), None)
        flt = next((f for f in floats if f[0] <= pos < f[1]), None)
        if cap:
            bits.append(f"caption of {flt[2]}" if flt and flt[2] else "caption")
        elif flt and flt[2]:
            bits.append(f"in {flt[2]}")
        return "; ".join(bits)

    # Row macros that carry cells: name -> number of uses.
    macros = {}
    for m in ROWMACRO_RE.finditer(body):
        content, end = braced(body, m.end())
        if CELL in content:
            uses = len(re.findall(r"\\" + m.group(1) + r"(?![A-Za-z])", body)) - 1
            macros[(m.start(), end)] = (m.group(1), uses)

    items: list[Item] = []
    for m in re.finditer(re.escape(PENDING), body):
        if body[m.start():].startswith("\\pendingcell"):
            continue
        arg, _ = braced(body, m.end())
        items.append(Item(line_of(m.start()), "pending", context(m.start()), " ".join(arg.split())))
    cells: dict[int, Item] = {}  # one entry per source line; cells counts the placeholders on it
    for m in re.finditer(re.escape(CELL) + r"(?![A-Za-z])", body):
        before = body[max(0, m.start() - 24):m.start()]
        if before.endswith("\\newcommand{") or before.endswith("\\DeclareRobustCommand{"):
            continue  # the definition itself
        inside = next((v for k, v in macros.items() if k[0] <= m.start() < k[1]), None)
        ln = line_of(m.start())
        if ln not in cells:
            text = f"row macro \\{inside[0]}, used by {inside[1]} rows" if inside else ""
            cells[ln] = Item(ln, "cell", context(m.start()), text, inside[1] if inside else 1)
        cells[ln].cells += 1
    for it in cells.values():
        n = it.cells
        it.text = (f"{n} cells per row, {it.text}" if it.rows > 1 else f"{n} cell{'s' * (n > 1)}") \
            + f" ({n * it.rows} in the PDF)"
        items.append(it)
    items.sort(key=lambda it: (it.line, it.kind))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tex", default=str(REPO / "paper" / "main.tex"))
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    items = scan(Path(args.tex))
    n_pending = sum(it.kind == "pending" for it in items)
    n_cell_src = sum(it.cells for it in items if it.kind == "cell")
    n_cell_pdf = sum(it.cells * it.rows for it in items if it.kind == "cell")
    if args.markdown:
        by_group: dict[str, list[Item]] = {}
        for it in items:
            by_group.setdefault(it.group(), []).append(it)
        for name, _ in GROUPS + [("Other", [])]:
            if name not in by_group:
                continue
            print(f"\n### {name}\n")
            print("| Line | Kind | Where | Text | Also filled by |")
            print("|---|---|---|---|---|")
            for it in by_group[name]:
                text = it.text.replace("|", "\\|")
                also = "; ".join(it.groups()[1:])
                print(f"| {it.line} | {it.kind} | {it.section} | {text} | {also} |")
    else:
        for it in items:
            print(f"L{it.line:<5} {it.kind:<7} [{' + '.join(it.groups())}] {it.section}: {it.text}")
    print(f"\n{n_pending} \\pending, {n_cell_src} \\pendingcell in the source "
          f"({n_cell_pdf} cells in the PDF once row macros are expanded); comments excluded.")


if __name__ == "__main__":
    main()
