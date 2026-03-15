def run() -> None:
    from .candidate_selection import run_candidate_selection
    from .discovery import run_discovery
    from .first_pass import run_first_pass
    from .negative_context import build_negative_contexts
    from .persist import (
        offload_model_and_sae,
        offload_to_cpu,
        reload_model_and_sae,
        save_results,
    )
    from .runtime import initialize_resources, initialize_runtime
    from .second_pass import run_second_pass

    print("")
    initialize_runtime()

    initialize_resources()

    # First pass: latent stats + context stores
    run_first_pass()
    save_results()

    # ANN step: build negative contexts
    offload_to_cpu()
    offload_model_and_sae()
    build_negative_contexts()
    reload_model_and_sae()

    # Second pass: top co-activation
    run_second_pass()

    # Discovery: select candidate seeds then grow circuits
    candidates = run_candidate_selection()
    run_discovery(candidates)

    print("Pipeline completed successfully!")
    print("")
