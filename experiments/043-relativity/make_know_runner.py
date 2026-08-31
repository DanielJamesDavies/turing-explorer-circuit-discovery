"""Generate know_runner.py from runner.py: adds the knowledge-circuit
arms (echo penalty + neg-suppress), the echo-corr precompute, its own
rows/members files, and a completeness guard keyed to the new arms.
Run from the repo root:  python experiments/043-relativity/make_know_runner.py
"""
import ast
from pathlib import Path

HERE = Path(__file__).parent
s = (HERE / "runner.py").read_text(encoding="utf-8")

s = s.replace(
    '"""RELATIVITY CONCEPT CIRCUITS --',
    '"""KNOWLEDGE-CIRCUIT ARMS (echo penalty + neg-suppress): the\n'
    "relativity runner with three fit variants per seed:\n"
    "  know400     tri-amp + ECHO PENALTY (member_penalty =\n"
    "              corr(a_i, a_seed)^2 over train probes, weight ECHO_W)\n"
    "              + NEG-SUPPRESS (reproduce the seed's natural silence\n"
    "              on the stored negctx hard negatives, weight NEG_W)\n"
    "  echoamp400  echo penalty only\n"
    "  negamp400   neg-suppress only\n"
    "One in-run null per seed at the know400 size. Writes to\n"
    "know_rows.jsonl / know_members.jsonl -- never the production files.\n"
    "\nOriginal header follows.\n\nRELATIVITY CONCEPT CIRCUITS --", 1)

s = s.replace('ROWS_PATH = HERE / "rows.jsonl"',
              'ROWS_PATH = HERE / "know_rows.jsonl"')
s = s.replace('MEM_FH = (HERE / "members.jsonl").open("a")',
              'MEM_FH = (HERE / "know_members.jsonl").open("a")')
s = s.replace(
    'SMOKE = os.environ.get("SMOKE") == "1"',
    'SMOKE = os.environ.get("SMOKE") == "1"\n'
    'ECHO_W = float(os.environ.get("ECHO_W", 5e-3))\n'
    'NEG_W = float(os.environ.get("NEG_W", 0.5))')

old = '''        arm_specs = [("triamp400", True, cfg.steps, 1e-3),
                     ("triamp100", True, 100, lam100),
                     ("gate400", False, cfg.steps, 1e-3)]'''
assert s.count(old) == 1
s = s.replace(old, '''        arm_specs = [("know400", True, cfg.steps, 1e-3),
                     ("echoamp400", True, cfg.steps, 1e-3),
                     ("negamp400", True, cfg.steps, 1e-3)]''')
s = s.replace('if tag in ("triamp400", "gate400"):',
              'if tag in ("know400", "echoamp400", "negamp400"):')
s = s.replace('''            if tag == "triamp400":
                n_ref = r["n"]''',
              '''            if tag == "know400":
                n_ref = r["n"]''')
s = s.replace(
    '''                if (r["comp_idx"], r["latent"], r["arm"]) == (comp_idx, sl,
                                                              "triamp400"):''',
    '''                if (r["comp_idx"], r["latent"], r["arm"]) == (comp_idx, sl,
                                                              "know400"):''')

old = '''        if (comp_idx, sl, "null%d" % (n_null - 1)) in done:'''
assert s.count(old) == 1
s = s.replace(old, '''        _needed = {"know400", "echoamp400", "negamp400",
                   "null%d" % (n_null - 1)}
        if all((comp_idx, sl, a) in done for a in _needed):''')

old = '''            scores, prov = run_learned_mask(inference, bank, objective="pos",
                                            **kw)'''
assert s.count(old) == 1
s = s.replace(old, '''            if kw.pop("_echo", False) and echo_pen is not None:
                kw.update(member_penalty=echo_pen,
                          member_penalty_weight=ECHO_W)
            else:
                kw.pop("_echo", None)
            if kw.pop("_negsup", False):
                kw.update(neg_suppress_weight=NEG_W)
            scores, prov = run_learned_mask(inference, bank, objective="pos",
                                            **kw)''')

old = '''        def fit(free_amp, steps, lam, support_members=None):'''
assert s.count(old) == 1
s = s.replace(old, '''        def fit(free_amp, steps, lam, support_members=None,
                echo=False, negsup=False):''')
old = '''            kw = dict(sites=UP, seed_layer=layer, seed_kind=kind,'''
assert s.count(old) == 1
s = s.replace(old, '''            kw = dict(_echo=echo, _negsup=negsup,
                      sites=UP, seed_layer=layer, seed_kind=kind,''')
old = '''            t0 = time.time()
            alphas, st = fit(fa, steps, lam)
            r = score(alphas, st, tag, time.time() - t0)'''
assert s.count(old) == 1
s = s.replace(old, '''            t0 = time.time()
            alphas, st = fit(fa, steps, lam,
                             echo=(tag in ("know400", "echoamp400")),
                             negsup=(tag in ("know400", "negamp400")))
            r = score(alphas, st, tag, time.time() - t0)''')

ECHO_BLOCK = '''        # ---- ECHO-CORR PRECOMPUTE (knowledge-circuit arms) ----------
        # corr(a_latent, a_seed)^2 per site over the TRAIN probe stream,
        # via streaming (sum, sum-of-squares, cross) accumulators; a
        # latent that merely copies the seed's own signal gets a high
        # penalty entry. Two passes: capture the seed trace, then the
        # per-site statistics against it.
        def _echo_corr():
            eps = 1e-6
            y_parts = []

            def cb_y(layer_idx, activations):
                if layer_idx != layer:
                    return
                with torch.no_grad():
                    ki_ = bank.kinds.index(kind)
                    ta, ti = bank.encode(activations[ki_], kind, layer_idx)
                    hit = (ti == sl)
                    y_parts.append(torch.where(
                        hit, ta.float(),
                        torch.zeros_like(ta.float())).amax(-1))

            for s0 in range(0, int(pt_tr.shape[0]), EVAL_BS):
                inference.forward(pt_tr[s0:s0 + EVAL_BS], num_gen=1,
                                  tokenize_final=False,
                                  activations_callback=cb_y,
                                  return_activations=False)
            y = torch.cat(y_parts)                            # [W, T]
            N = float(y.numel())
            sy, syy = float(y.sum()), float((y * y).sum())
            acc = {st_: [torch.zeros(D_SAE, device=y.device)
                         for _ in range(3)] for st_ in UP}
            state = {}

            def cb_s(layer_idx, activations):
                with torch.no_grad():
                    for ki_, kd in enumerate(bank.kinds):
                        st_ = (layer_idx, kd)
                        if st_ not in acc:
                            continue
                        ta, ti = bank.encode(activations[ki_], kd, layer_idx)
                        B, T, K = ta.shape
                        v = ta.float().reshape(-1)
                        ii = ti.reshape(-1).long()
                        yr = state["y"][:B, :T, None].expand(B, T, K)
                        yr = yr.reshape(-1).to(v.device)
                        a = acc[st_]
                        a[0].index_add_(0, ii, v)
                        a[1].index_add_(0, ii, v * v)
                        a[2].index_add_(0, ii, v * yr)

            for s0 in range(0, int(pt_tr.shape[0]), EVAL_BS):
                state["y"] = y[s0:s0 + EVAL_BS]
                inference.forward(pt_tr[s0:s0 + EVAL_BS], num_gen=1,
                                  tokenize_final=False,
                                  activations_callback=cb_s,
                                  return_activations=False)
            out = {}
            my, vy = sy / N, max(syy / N - (sy / N) ** 2, eps)
            for st_, (sv, svv, svy) in acc.items():
                mv = sv / N
                var = (svv / N - mv * mv).clamp_min(0.0)
                cov = svy / N - mv * my
                corr = cov / ((var * vy).sqrt() + eps)
                out[st_] = (corr * corr).detach()
            return out

        echo_pen = _echo_corr()
        _top = max(float(v.max()) for v in echo_pen.values())
        print("[echo] max corr^2 %.3f | latents with corr^2 > 0.5: %d"
              % (_top,
                 sum(int((v > 0.5).sum()) for v in echo_pen.values())),
              flush=True)

'''
i = s.index('        arm_specs = [("know400"')
s = s[:i] + ECHO_BLOCK + s[i:]

(HERE / "know_runner.py").write_text(s, encoding="utf-8", newline="")
ast.parse(s)
print("know_runner.py written and compiles")
