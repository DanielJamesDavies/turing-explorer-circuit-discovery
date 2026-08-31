"""Add to both scoring harnesses:
  HYB=1     fit amplitudes (support-restricted, lambda 0 -- the coact_amp
            machinery) on the matched-size head of whichever external
            ranking THEIRS_FILE/ARM_PREFIX point at -> arm "<prefix>_amp";
            and on ct_seed_rooted's frequency head when SCORE_CT=1
            -> "ct_seed_rooted_matched_amp".
  LAMTAG    suffix for the tri-amp / tri-mask arms so a lambda sweep
            writes triamp400<LAMTAG> rows instead of colliding with the
            production arms; nulls and opt-in arms stay untouched.
"""
import ast
from pathlib import Path

E = Path(__file__).parent
for harness in ["038-transcoder-compare-gemma/ours_gtc.py",
                "035-transcoder-compare/ours_llama.py"]:
    p = E / harness
    s = p.read_text(encoding="utf-8")

    def rep(old, new, n=1):
        global s
        assert s.count(old) >= 1, (harness, old[:70])
        s = s.replace(old, new, n)

    # knobs
    rep('ARM_PREFIX = os.environ.get("ARM_PREFIX", "theirs")',
        'ARM_PREFIX = os.environ.get("ARM_PREFIX", "theirs")\n'
        'HYB = os.environ.get("HYB") == "1"\n'
        'LAMTAG = os.environ.get("LAMTAG", "")')

    # tagged tri-amp / tri-mask arms (definitions and guards)
    rep('if (L, sl, "triamp400") not in done:', 'if (L, sl, "triamp400" + LAMTAG) not in done:')
    rep('score(fit(True, LAM), "triamp400", time.time() - t0)["n"]',
        'score(fit(True, LAM), "triamp400" + LAMTAG, time.time() - t0)["n"]')
    rep('if (L, sl, "gate400") not in done:', 'if (L, sl, "gate400" + LAMTAG) not in done:')
    rep('score(fit(False, LAM), "gate400", time.time() - t0)', 'score(fit(False, LAM), "gate400" + LAMTAG, time.time() - t0)')
    # n_ref fallback read stays on the untagged production arm
    # completeness guard: a sweep run must not be skipped as "complete"
    rep('        needed = {"null%d" % (N_NULL - 1)}',
        '        needed = {"null%d" % (N_NULL - 1)}\n'
        '        if LAMTAG:\n'
        '            needed |= {"triamp400" + LAMTAG, "gate400" + LAMTAG}')

    # hybrid on the external matched head (inside the THEIRS block)
    rep('''                score(ma, ARM_PREFIX, 0.0)''',
        '''                score(ma, ARM_PREFIX, 0.0)
        if HYB and THEIRS and (L, sl, ARM_PREFIX + "_amp") not in done:
            tm = THEIRS.get((L, sl))
            if tm:
                support = {}
                for lyr, f in tm:
                    support.setdefault(int(lyr), []).append(int(f))
                support = {k: torch.tensor(v, dtype=torch.long, device=DEV)
                           for k, v in support.items()}
                t0 = time.time()
                score(fit(True, 0.0, support=support), ARM_PREFIX + "_amp",
                      time.time() - t0)''')

    # hybrid on ct_seed_rooted's frequency head
    rep('''                score(_ma, _tag, 0.0)''',
        '''                score(_ma, _tag, 0.0)
        if HYB and CT_PRUNED.get((L, sl), {}).get("ct_seed_rooted") and n_ref \\
                and (L, sl, "ct_seed_rooted_matched_amp") not in done:
            _mem = CT_PRUNED[(L, sl)]["ct_seed_rooted"]["freq"][:n_ref]
            if _mem:
                support = {}
                for _lyr, _f in _mem:
                    support.setdefault(int(_lyr), []).append(int(_f))
                support = {k: torch.tensor(v, dtype=torch.long, device=DEV)
                           for k, v in support.items()}
                t0 = time.time()
                score(fit(True, 0.0, support=support),
                      "ct_seed_rooted_matched_amp", time.time() - t0)''')
    p.write_text(s, encoding="utf-8", newline="")
    ast.parse(s)
    print("patched", harness)
