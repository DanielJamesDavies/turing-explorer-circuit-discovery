"""Experiment 108 arm table (shared by fit_arm.py, plan.py, analyze.py; stdlib only).

Every arm is production v4 (062 driver, WCM_VERSION=v4, lambda 1e-3, TuringLLM) with only the effective batch, the step
count, the peak lr and (the s2 replicates) the R donor seed changed. Every arm runs at micro-batch 4 with gradient
accumulation (accum = batch / 4), so the peak VRAM is production's at every depth.

Peak lr of the 100-step arms: matched to production's lr budget sum(lr) under the v4 schedule (geometric, lr_min_frac
0.05, warm-up round(0.1 * steps)), the budget the v4 x3.15 multiplier was defined by (driver.py:97-101,
latent_pilot README part 2: sum(lr) sets both the pruning walk and the decay budget sum(lr) * wd). lr_sum() below is
learned_mask.lr_at (src/circuit/instrument/learned_mask.py:1630-1650) transcribed; check_budget() asserts the
production sum 21.40 that the spec quotes (paper/protocol.md §0.3).
"""

PROD_PEAK = 0.05 * 3.15          # config lr 0.05 (src/config.py:682) x WCM_LR_MULT["v4"] (driver.py:104) = 0.1575
PROD_STEPS = 400
MICRO = 4                        # the micro-batch of every arm (production's batch; deepest TuringLLM peak ~9.9 GB)
R_SEED_2 = 20261108              # the s2 replicates' R donor-draw seed (production: 20261017, src/config.py:954)
N_TRAIN = 48                     # training contexts per target (32 strongest + 16 mid-band, arm B)


def lr_at(step, steps, peak, warmup_frac=0.1, lr_min_frac=0.05):
    """learned_mask.lr_at for lr_schedule="geometric" with warm-up."""
    n_warm = int(round(float(warmup_frac) * int(steps)))
    floor = float(peak) * float(lr_min_frac)
    if step < n_warm:
        return floor + (float(peak) - floor) * (step + 1) / n_warm
    frac = (step - n_warm) / max(int(steps) - n_warm - 1, 1)
    return float(peak) * float(lr_min_frac) ** frac


def lr_sum(steps, peak):
    return sum(lr_at(s, steps, peak) for s in range(int(steps)))


def matched_peak(steps):
    """The peak lr whose sum(lr) over `steps` equals production's (400 steps at 0.1575)."""
    return PROD_PEAK * lr_sum(PROD_STEPS, 1.0) / lr_sum(steps, 1.0)


PEAK_100 = round(matched_peak(100), 6)   # 0.620595 (x3.940 production's peak)

# name: (effective batch, steps, peak lr, R seed or None = production's), in run order (a partial run stays useful)
ARMS = {
    "B4":    (4, 400, PROD_PEAK, None),       # control = production v4 exactly (accum 1)
    "B48":   (48, 100, PEAK_100, None),       # full batch: all 48 training contexts every step (accum 12)
    "B16m":  (16, 100, PEAK_100, None),       # batch 16, compute-matched to B4 (1,600 context passes), lr-budget matched
    "B16":   (16, 400, PROD_PEAK, None),      # batch 16, production steps and lr (4x B4's compute)
    "B4s2":  (4, 400, PROD_PEAK, R_SEED_2),   # B4 with another R donor seed (seed-overlap reference)
    "B48s2": (48, 100, PEAK_100, R_SEED_2),   # B48 with that seed
}
ORDER = list(ARMS)
TIMING_ARMS = ("B4", "B16")


def arm(name):
    if name not in ARMS:
        raise SystemExit("ARM must be one of %s, got %r" % (ORDER, name))
    b, s, lr, rs = ARMS[name]
    if b % MICRO:
        raise SystemExit("batch %d is not a multiple of the micro-batch %d" % (b, MICRO))
    return dict(name=name, batch=b, steps=s, peak_lr=lr, r_seed=rs, micro=MICRO, accum=b // MICRO,
                lr_sum=round(lr_sum(s, lr), 4), context_passes=b * s)


def n_sites(key):
    """Upstream SAE sites of a TuringLLM target (3 kinds per layer, attn < mlp < resid; spec §2.1)."""
    l, k, _ = key.split(".")
    return 3 * int(l) + {"attn": 0, "mlp": 1, "resid": 2}[k]


def check_budget():
    s = lr_sum(PROD_STEPS, PROD_PEAK)
    assert abs(s - 21.40) < 0.01, s                      # the spec's sum(lr) for production v4
    for name in ORDER:
        a = arm(name)
        assert abs(a["lr_sum"] - s) < 0.01 or a["steps"] == PROD_STEPS and a["peak_lr"] == PROD_PEAK, a
    return s


if __name__ == "__main__":
    print("production sum(lr) %.4f; 100-step matched peak %.6f (x%.4f)" % (check_budget(), PEAK_100,
                                                                           PEAK_100 / PROD_PEAK))
    for n in ORDER:
        print(arm(n))
