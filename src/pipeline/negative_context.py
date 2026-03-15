from .runtime import get_runtime
from store.context import mid_ctx, neg_ctx, top_ctx
from store.neg_context import NegCtxStats, build_neg_ctx


def build_negative_contexts() -> None:
    runtime = get_runtime()
    print("--- ANN Step: Building Negative Contexts ---")
    assert runtime.seq_repr is not None

    try:
        neg_stats: NegCtxStats = build_neg_ctx(runtime.seq_repr, top_ctx, mid_ctx, neg_ctx)
        neg_ctx.save("outputs/neg_ctx.pt")
        neg_stats.save("outputs/neg_ctx_stats.json")
        neg_stats.print_summary(neg_ctx.num_ctx_sequences)
        print("  ✓ neg_ctx built and saved")
    except ImportError as error:
        print(f"  ✗ neg_ctx skipped: {error}")
    print("")
