"""Executable finite-data examples, including an expectation-cancellation case."""
import json
from pathlib import Path
from scrubbing import Scrubber
from test_scrubbing import arithmetic, shared_ancestor


def main():
    rows = []
    for name, build in [("correct_arithmetic", lambda: arithmetic(True)),
                        ("wrong_feature_meaning", lambda: arithmetic(False)),
                        ("shared_ancestor_false_invariance", shared_ancestor)]:
        h, data = build()
        s = Scrubber(h, data)
        exact_mse, original_mean, scrubbed_mean = 0.0, 0.0, 0.0
        replay_error, summaries = 0.0, []
        for ref, x in enumerate(data):
            natural = h.G.evaluate(x)
            dist = s.enumerate_distribution(ref)
            original_mean += natural / len(data)
            scrubbed_mean += sum(y * p for y, p in dist.items()) / len(data)
            exact_mse += sum((y - natural) ** 2 * p for y, p in dist.items()) / len(data)
            for seed in range(32):
                output, trace = s.run(ref, seed)
                replay_error = max(replay_error, abs(output - s.replay_tree(trace)))
                summaries.append(trace.summary())
        rows.append(dict(case=name, exact_context_paired_mse=exact_mse, natural_mean=original_mean,
                         scrubbed_mean=scrubbed_mean, max_tree_replay_error=replay_error,
                         runs=len(summaries), example_coverage=summaries[0]))
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    (out / "toy_report.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Recursive reference-engine examples", "", "Exact distributions on finite toy datasets.", "",
             "| Case | Natural mean | Scrubbed mean | Context-paired MSE | Maximum tree replay error |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['natural_mean']:.3g} | {r['scrubbed_mean']:.3g} | "
                     f"{r['exact_context_paired_mse']:.3g} | {r['max_tree_replay_error']:.3g} |")
    lines += ["", "The wrong hypotheses preserve the unconditional mean in these examples. Raw mean recovery alone",
              "therefore misses their errors. A paired squared-error (or classification disagreement) metric can be",
              "included in the model graph, with the clean reference output treated as its label. This is a choice",
              "of explanandum, not a modification of the sampling algorithm.", ""]
    (out / "toy_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
