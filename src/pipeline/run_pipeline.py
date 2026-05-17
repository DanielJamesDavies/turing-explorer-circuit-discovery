def run() -> None:
    from observability.timing import phase_timer
    from .candidate_selection import run_candidate_selection
    from .discovery import run_discovery
    from .first_pass import run_first_pass
    from .negative_context import build_negative_contexts
    from .persist import (
        offload_model_and_sae,
        offload_to_cpu,
        reload_model_and_sae,
        build_search_cache_if_enabled,
        save_results,
    )
    from .runtime import get_runtime, initialize_resources, initialize_runtime
    from .second_pass import run_second_pass
    from config import config

    print("")
    with phase_timer("initialize runtime"):
        initialize_runtime()

    with phase_timer("initialize resources / data loading"):
        initialize_resources()

    # First pass: latent stats + context stores
    with phase_timer("first pass: latent stats and contexts"):
        run_first_pass()
    with phase_timer("persistence: first-pass outputs"):
        save_results()

    # ANN step: build negative contexts
    with phase_timer("offload stores to CPU"):
        offload_to_cpu()
    with phase_timer("offload model and SAE"):
        offload_model_and_sae()
    with phase_timer("negative context build"):
        build_negative_contexts()
    with phase_timer("reload model and SAE"):
        reload_model_and_sae()

    # Second pass: top co-activation
    with phase_timer("second pass: top coactivation"):
        run_second_pass()

    # Discovery: select candidate seeds then grow circuits
    with phase_timer("candidate selection"):
        candidates = run_candidate_selection()
    with phase_timer("discovery"):
        run_discovery(candidates)

    # Cluster contrast discovery (opt-in via "cluster_contrast" in config.discovery.methods)
    if "cluster_contrast" in list(config.discovery.methods):
        from .cluster_discovery import run_cluster_contrast_discovery
        runtime = get_runtime()
        assert runtime.model is not None
        assert runtime.bank is not None
        assert runtime.loader is not None
        with phase_timer("cluster contrast discovery"):
            run_cluster_contrast_discovery(runtime.model, runtime.bank, runtime.loader)

    runtime = get_runtime()
    if config.persist.search_cache_enabled and config.persist.build_search_cache_after_pipeline:
        assert runtime.bank is not None
        assert runtime.loader is not None
        from store.context import top_ctx

        with phase_timer("search cache build"):
            build_search_cache_if_enabled(
                top_ctx,
                runtime.bank,
                runtime.loader,
                output_path="outputs/search_cache.parquet",
            )

    print("Pipeline completed successfully!")
    print("")
