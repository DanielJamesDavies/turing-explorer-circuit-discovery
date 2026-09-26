import torch

from circuit.context_split import stratified_order, stratified_split, stratified_subset


def test_64_contexts_hold_out_16_with_rotating_offsets():
    scores = torch.arange(64, 0, -1).float()          # index 0 strongest
    train, held = stratified_split(scores)
    assert len(held) == 16 and len(train) == 48
    assert held[:8] == [2, 7, 8, 13, 18, 23, 24, 29]  # ranks 3, 8, 9, 14, 19, 24, 25, 30 (1-based)
    assert 0 in train and 1 in train                  # the two strongest always train
    offsets = [h % 4 for h in held]
    assert sorted(offsets) == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4


def test_split_follows_scores_not_list_order():
    scores = torch.tensor([1.0, 5.0, 3.0, 9.0, 7.0, 2.0, 8.0, 4.0])
    train, held = stratified_split(scores)
    assert set(train) | set(held) == set(range(8)) and not set(train) & set(held)
    assert 3 in train and 6 in train                  # the two highest scores
    assert len(held) == 2


def test_held_out_count_is_exact_for_any_n():
    for n in range(1, 130):
        train, held = stratified_split(torch.rand(n))
        assert len(held) == int(round(n * 0.25))
        assert len(train) + len(held) == n


def test_order_puts_train_first_so_list_slicing_still_works():
    scores = torch.rand(64)
    order = stratified_order(scores)
    train, held = stratified_split(scores)
    n_train = 64 - int(round(64 * 0.25))
    assert order[:n_train] == train and order[n_train:] == held


def test_offset_draws_a_different_sample_of_the_same_shape():
    scores = torch.arange(48, 0, -1).float()
    a_train, a_held = stratified_split(scores, holdout_frac=1 / 3, keep_top=2)
    b_train, b_held = stratified_split(scores, holdout_frac=1 / 3, keep_top=2, offset=3)
    assert len(a_train) == len(b_train) == 32 and a_train != b_train
    assert 0 in b_train and 1 in b_train


def test_subset_spans_the_ranking():
    scores = torch.arange(48, 0, -1).float()
    pick = stratified_subset(scores, 16)
    assert len(pick) == 16 and pick == sorted(pick)
    assert min(pick) <= 2 and max(pick) >= 45
