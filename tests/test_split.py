import random
from collections import Counter

from ddet.split import Group, stratified_group_split

RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


def make_groups(n=300, seed=1):
    rng = random.Random(seed)
    groups = []
    for i in range(n):
        classes = Counter()
        for _ in range(rng.randint(0, 3)):
            # класс 0 частый, класс 4 редкий
            classes[rng.choices(range(5), weights=[50, 25, 15, 8, 2])[0]] += rng.randint(1, 4)
        groups.append(Group(f"g{i}", rng.randint(1, 6), classes))
    return groups


def test_every_group_gets_exactly_one_split():
    groups = make_groups()
    result = stratified_group_split(groups, RATIOS)
    assert set(result) == {g.id for g in groups}
    assert set(result.values()) <= set(RATIOS)


def test_image_ratios_close_to_target():
    groups = make_groups()
    result = stratified_group_split(groups, RATIOS)
    total = sum(g.n_images for g in groups)
    per = Counter()
    for g in groups:
        per[result[g.id]] += g.n_images
    for s, r in RATIOS.items():
        assert abs(per[s] / total - r) < 0.05, (s, per[s] / total)


def test_class_ratios_close_to_target():
    groups = make_groups()
    result = stratified_group_split(groups, RATIOS)
    for c in range(4):
        per = Counter()
        for g in groups:
            per[result[g.id]] += g.classes[c]
        total = sum(per.values())
        for s, r in RATIOS.items():
            assert abs(per[s] / total - r) < 0.07, (c, s, per[s] / total)


def test_rare_class_reaches_every_split():
    # Три группы с редким классом и много фона — раньше весь редкий класс мог уйти в одну выборку.
    groups = [Group(f"bg{i}", 5) for i in range(100)]
    groups += [Group(f"rare{i}", 3, Counter({9: 4})) for i in range(3)]
    result = stratified_group_split(groups, RATIOS)
    assert {result[f"rare{i}"] for i in range(3)} == set(RATIOS)


def test_deterministic_for_same_seed():
    groups = make_groups()
    assert stratified_group_split(groups, RATIOS, seed=3) == stratified_group_split(groups, RATIOS, seed=3)


def test_zero_ratio_split_is_unused():
    groups = make_groups(50)
    result = stratified_group_split(groups, {"train": 0.8, "val": 0.2, "test": 0})
    assert "test" not in result.values()


def test_fixed_groups_keep_their_split_and_count_in_balance():
    groups = make_groups()
    fixed = {g.id: "test" for g in groups[:60]}          # много закреплённых в test
    result = stratified_group_split(groups, RATIOS, fixed=fixed)
    assert all(result[g] == "test" for g in fixed)
    free = [g for g in groups if g.id not in fixed]
    assert sum(result[g.id] == "test" for g in free) < 5  # test уже переполнен — новые туда почти не идут
