"""B3 tests (CPU, no model, no SAE): the public-model target frame. Quotas (512 per model), exclusions (no upstream
SAE site; the frame's eligibility >= 200 firing windows and the 50% hub cap, counted per component), disjoint / nested
subsets, both D1 frames, determinism (byte-identical files), and the frozen-list guard.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest experiments/082-public-runs/test_082_targets.py -q
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_targets_public as T  # noqa: E402

D_SAE = 32768
NW = 10000                                     # synthetic corpus window total


def site(key):
    return key.rsplit(".", 1)[0]


def latent(key):
    return int(key.rsplit(".", 1)[1])


def comp_of(family):
    return {"%d.%s" % (layer, kind): c for c, layer, kind, _ in T.components(family)}


def seq_count(family, seed=1, live_frac=0.5, hub_frac=0.05):
    """Synthetic latent_stats.seq_count over NW windows: per component a random half of the latents fire in
    200..4999 windows (eligible), the rest in 0..199 (exactly 199 for some, the edge of the rule), and a few hubs in
    5001..NW windows (5000 = exactly 50%: kept)."""
    n_comp = len(T.components(family))
    g = np.random.default_rng(seed)
    sc = g.integers(0, 200, size=(n_comp, D_SAE))
    live = g.random((n_comp, D_SAE)) < live_frac
    sc[live] = g.integers(200, 5000, size=int(live.sum()))
    hub = g.random((n_comp, D_SAE)) < hub_frac
    sc[hub] = g.integers(5001, NW + 1, size=int(hub.sum()))
    sc[:, 0], sc[:, 1], sc[:, 2] = 5000, 5001, 199
    return sc


def fired(family, seed=0, **kw):
    return T.draw(family, "fired", seed, seq_count=seq_count(family), n_windows=NW, **kw)


class TestQuotasAndExclusions:
    @pytest.mark.parametrize("family,n_comp,excluded,per_site", [
        ("pythia-70m", 16, {"0.attn", "0.mlp"}, {32}),
        ("gpt2-small", 47, {"0.attn"}, {10, 11}),
    ])
    def test_512_per_model_and_excluded_sites(self, family, n_comp, excluded, per_site):
        lists, man = fired(family)
        per = Counter(site(k) for k in lists["main"])
        assert len(lists["main"]) == 512 == man["counts"]["main"] and len(per) == n_comp
        assert set(per.values()) == per_site
        if family == "gpt2-small":
            assert Counter(per.values()) == {11: 42, 10: 5}
        assert set(man["excluded"]) == excluded and not excluded & set(per)
        every = {"%d.%s" % (layer, kind) for _, layer, kind, _ in T.components(family)}
        assert set(per) == every - excluded and set(man["eligible"]) == set(per)
        assert all(not any(site(k) == e for e in excluded) for name in lists for k in lists[name])

    def test_gpt2_keeps_resid_mid_and_layer_0_non_attention_sites(self):
        lists, _ = fired("gpt2-small")
        sites = {site(k) for k in lists["main"]}
        assert {"0.resid_mid", "0.mlp", "0.resid", "11.resid_mid"} <= sites

    def test_unknown_family_and_frame_are_refused(self):
        with pytest.raises(ValueError):
            T.draw("turingllm", "all", 0)
        with pytest.raises(ValueError):
            T.draw("pythia-70m", "survivors", 0)
        with pytest.raises(ValueError, match="needs latent_stats"):
            T.draw("pythia-70m", "fired", 0)
        with pytest.raises(ValueError, match="shape"):
            T.draw("pythia-70m", "fired", 0, seq_count=np.zeros((18, 100)), n_windows=NW)
        with pytest.raises(ValueError, match="n_windows"):
            T.draw("pythia-70m", "fired", 0, seq_count=seq_count("pythia-70m"))


class TestEligibility:
    @pytest.mark.parametrize("family", ["pythia-70m", "gpt2-small"])
    def test_every_list_is_inside_the_200_window_and_hub_cap_frame(self, family):
        sc = seq_count(family)
        lists, man = T.draw(family, "fired", 0, seq_count=sc, n_windows=NW)
        comp = comp_of(family)
        for name, keys in lists.items():
            v = np.array([sc[comp[site(k)], latent(k)] for k in keys])
            assert (v >= 200).all() and (v <= NW // 2).all(), name
        assert man["main_below_min_seqs"] == 0 and man["main_above_hub_cap"] == 0
        assert man["min_seqs"] == 200 and man["hub_frac"] == 0.5 and man["hub_cap_windows"] == 5000.0

    def test_edges_200_kept_199_dropped_50pct_kept_above_dropped(self):
        keep, ex = T.eligibility(np.array([0, 1, 199, 200, 5000, 5001, 9999]), "fired", 200, 0.5, 10000)
        assert keep.tolist() == [3, 4]
        assert ex == dict(dead=1, below_min=2, hub=2)

    def test_exclusion_counts_per_component_add_up(self):
        sc = seq_count("pythia-70m")
        _, man = T.draw("pythia-70m", "fired", 0, seq_count=sc, n_windows=NW)
        comp = comp_of("pythia-70m")
        tot = Counter()
        for c in man["components"]:
            row = sc[comp[c["site"]]]
            ex = c["excluded"]
            assert ex["dead"] == int((row == 0).sum())
            assert ex["below_min"] == int(((row > 0) & (row < 200)).sum())
            assert ex["hub"] == int((row > 5000).sum())
            assert c["frame"] + sum(ex.values()) == D_SAE
            tot.update(ex)
        assert man["latents_excluded"] == dict(tot)

    def test_thin_component_is_a_shortfall_and_the_total_is_kept(self):
        sc = np.full((18, D_SAE), 300, dtype=np.int64)
        sc[5, 10:] = 0                                                        # 1.resid: only 10 latents eligible
        lists, man = T.draw("pythia-70m", "fired", 0, seq_count=sc, n_windows=NW)
        per = Counter(site(k) for k in lists["main"])
        assert per["1.resid"] == 10 and man["shortfalls"] == {"1.resid": 22}
        assert len(lists["main"]) == 512                                      # redistributed to the other components
        assert per.get("1.resid", 0) + sum(per.values()) - per["1.resid"] == 512
        assert Counter(site(k) for k in lists["pilot"]).get("1.resid", 0) == 0   # nothing left for the pilot
        assert len(lists["pilot"]) == 30 and not set(lists["pilot"]) & set(lists["main"])

    def test_all_frame_counts_expected_skips_and_ignores_counts(self):
        sc = seq_count("pythia-70m")
        lists, man = T.draw("pythia-70m", "all", 0, seq_count=sc, n_windows=NW)
        comp = comp_of("pythia-70m")
        below = sum(sc[comp[site(k)], latent(k)] < 200 for k in lists["main"])
        assert man["main_below_min_seqs"] == below > 0 and man["frame_option"].startswith("(a)")
        assert lists == T.draw("pythia-70m", "all", 0)[0]                  # seq_count only counts, never selects


class TestSubsets:
    @pytest.mark.parametrize("family,n_unw", [("pythia-70m", 128), ("gpt2-small", 0)])
    def test_sizes_pilot_disjoint_subsets_inside_main(self, family, n_unw):
        lists, man = fired(family)
        main = set(lists["main"])
        want = dict(main=512, pilot=30, stable=32, nulls=128)
        if n_unw:
            want["unweighted"] = n_unw
        assert {k: len(v) for k, v in lists.items()} == want
        assert not set(lists["pilot"]) & main
        for name in want:
            if name not in ("main", "pilot"):
                assert set(lists[name]) <= main, name
            assert len(set(lists[name])) == len(lists[name])
        assert man["relations"]["pilot_disjoint_from_main"] is True
        assert man["relations"]["unweighted_subset_of_main"] is bool(n_unw)

    @pytest.mark.parametrize("family", ["pythia-70m", "gpt2-small"])
    def test_subsets_are_stratified_evenly_over_components(self, family):
        lists, man = fired(family)
        n_comp = len(man["eligible"])
        for name in ("pilot", "stable", "nulls", "unweighted"):
            if name not in lists:
                continue
            per = Counter(site(k) for k in lists[name])
            counts = [per.get(s, 0) for s in man["eligible"]]
            assert max(counts) - min(counts) <= 1 and sum(counts) == len(lists[name])
            assert min(counts) == len(lists[name]) // n_comp

    @pytest.mark.parametrize("family,per_sub", [("pythia-70m", {2}), ("gpt2-small", {0, 1})])
    def test_nulls_panel_stratified_with_subsample_first(self, family, per_sub):
        lists, man = fired(family)
        nl = lists["nulls"]
        per = Counter(site(k) for k in nl)
        assert len(nl) == len(set(nl)) == 128 and max(per.values()) - min(per.values()) <= 1
        if family == "pythia-70m":
            assert set(per.values()) == {8}
        sub_n = man["nulls_sub_n"]
        assert sub_n == 32
        sub = Counter(site(k) for k in nl[:sub_n])
        assert set(sub.get(s, 0) for s in man["eligible"]) == per_sub and sum(sub.values()) == 32
        assert nl[:sub_n] == T.order(nl[:sub_n]) and nl[sub_n:] == T.order(nl[sub_n:])
        assert all(c["nulls"] == per[c["site"]] and c["nulls_sub"] == sub.get(c["site"], 0) for c in man["components"])
        b = fired(family, n_nulls=47, n_nulls_sub=10)[0]                     # main / others ignore the nulls sizes
        assert all(b[k] == lists[k] for k in lists if k != "nulls") and len(b["nulls"]) == 47

    def test_allocate_respects_capacity_and_total(self):
        alloc = T.allocate(100, {0: 3, 1: 50, 2: 50, 3: 0}, T.rng(0, 9))
        assert sum(alloc.values()) == 100 and alloc[0] == 3 and alloc[3] == 0
        assert abs(alloc[1] - alloc[2]) <= 1
        assert sum(T.allocate(500, {0: 3, 1: 4}, T.rng(0, 9)).values()) == 7          # short: everything


class TestDeterminism:
    def test_same_inputs_and_seed_give_byte_identical_files(self, tmp_path):
        outs = []
        for d in ("a", "b"):
            lists, man = fired("pythia-70m", 7)
            paths, mpath = T.write(tmp_path / d, "pythia-70m", lists, man)
            outs.append({p.name: p.read_bytes() for p in list(paths.values()) + [mpath]})
        assert outs[0] == outs[1] and len(outs[0]) == 6
        man = json.loads(outs[0]["targets_pythia-70m_manifest.json"])
        assert set(man["files"]) == {"targets_pythia-70m_%s.txt" % n for n in ("main", "pilot", "unweighted",
                                                                                "stable", "nulls")}
        assert man["files"]["targets_pythia-70m_main.txt"] == T.md5(tmp_path / "a" / "targets_pythia-70m_main.txt")
        assert man["seed"] == 7 and b"\r\n" not in outs[0]["targets_pythia-70m_main.txt"]

    def test_seed_changes_the_draw_and_main_ignores_subset_sizes(self):
        a = fired("gpt2-small", 0)[0]
        assert set(a["main"]) != set(fired("gpt2-small", 1)[0]["main"])
        b = fired("gpt2-small", 0, n_pilot=47, n_stable=10)[0]
        assert a["main"] == b["main"] and set(a["pilot"]) <= set(b["pilot"])     # 30 is a prefix of 47 per site
        assert len(b["pilot"]) == 47 and len(b["stable"]) == 10

    def test_lists_use_the_062_order(self):
        lists, _ = fired("pythia-70m")
        assert lists["main"] == T.order(lists["main"]) and lists["main"] != sorted(lists["main"])
        assert all(lists[n] == T.order(lists[n]) for n in ("pilot", "stable", "unweighted"))

    def test_frozen_lists_are_not_overwritten(self, tmp_path):
        lists, man = fired("pythia-70m", 0)
        T.write(tmp_path, "pythia-70m", lists, man)
        T.write(tmp_path, "pythia-70m", lists, man)                          # identical: a no-op
        other, man1 = fired("pythia-70m", 1)
        with pytest.raises(SystemExit, match="refusing"):
            T.write(tmp_path, "pythia-70m", other, man1)
        assert (tmp_path / "targets_pythia-70m_main.txt").read_text().split() == lists["main"]
        T.write(tmp_path, "pythia-70m", other, man1, force=True)
        assert (tmp_path / "targets_pythia-70m_main.txt").read_text().split() == other["main"]


class TestCli:
    def test_cli_reads_latent_stats_window_total_and_records_md5(self, tmp_path, capsys):
        import torch
        stores = tmp_path / "stores"
        stores.mkdir()
        torch.save({"seq_count": torch.from_numpy(seq_count("pythia-70m"))}, stores / "latent_stats.pt")
        (stores / "data_pass_manifest.json").write_text(json.dumps(dict(n_sequences=NW)))
        T.main(["--family", "pythia-70m", "--frame", "fired", "--seed", "20261024", "--stores", str(stores),
                "--out", str(tmp_path / "out")])
        man = json.loads((tmp_path / "out" / "targets_pythia-70m_manifest.json").read_text())
        assert man["latent_stats_md5"] == T.md5(stores / "latent_stats.pt") and man["counts"]["main"] == 512
        assert man["n_windows"] == NW and man["seed"] == 20261024
        out = capsys.readouterr().out
        assert "excluded (no upstream SAE site): 0.attn, 0.mlp" in out and "hub cap 50%" in out

    def test_cli_fired_frame_requires_stores_and_window_total(self, tmp_path):
        import torch
        with pytest.raises(SystemExit):
            T.main(["--family", "pythia-70m", "--frame", "fired", "--seed", "0", "--out", str(tmp_path)])
        stores = tmp_path / "s"
        stores.mkdir()
        torch.save({"seq_count": torch.from_numpy(seq_count("pythia-70m"))}, stores / "latent_stats.pt")
        with pytest.raises(SystemExit):
            T.main(["--family", "pythia-70m", "--frame", "fired", "--seed", "0", "--stores", str(stores),
                    "--out", str(tmp_path / "o")])
