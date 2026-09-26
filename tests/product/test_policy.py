from guard.domain.models import Detection
from guard.domain.policy import NO_PLATE, AlertPolicy, Settings


def spag(x, conf=0.9):
    return Detection("defects", "spaghetti", conf, (x, 100, x + 60, 160))


def run(policy, frames):
    fired = []
    for k, dets in enumerate(frames):
        fired += policy.update(dets, k * 5.0).fired
    return fired


def test_same_place_confirms():
    p = AlertPolicy(Settings(window=5, min_hits=3), has_scene=False)
    assert run(p, [[spag(100)]] * 4) == ["spaghetti"]


def test_jumping_detections_do_not_confirm():
    p = AlertPolicy(Settings(window=5, min_hits=3), has_scene=False)
    assert run(p, [[spag(x)] for x in (0, 150, 300, 450, 75, 225, 375, 525)]) == []


def test_without_spatial_jumping_would_fire():
    p = AlertPolicy(Settings(window=5, min_hits=3, spatial=False), has_scene=False)
    assert run(p, [[spag(x)] for x in (0, 200, 400)]) == ["spaghetti"]


def test_growing_clump_confirms():
    p = AlertPolicy(Settings(window=5, min_hits=3), has_scene=False)
    frames = [[Detection("defects", "spaghetti", 0.9, (100, 100, 100 + w, 100 + w))] for w in (30, 45, 60, 80)]
    assert run(p, frames) == ["spaghetti"]


def test_low_confidence_ignored_and_stop_only_allowlist():
    p = AlertPolicy(Settings(window=3, min_hits=2, defect_conf=0.7), has_scene=False)
    assert run(p, [[spag(100, 0.6)]] * 5) == []
    p = AlertPolicy(Settings(window=3, min_hits=2, baseline_s=0), has_scene=False)
    v = None
    for k in range(4):
        v = p.update([Detection("defects", "garbage", 0.9, (1, 1, 20, 20))], k)
    assert v is not None and "garbage" in v.problems and v.stop_for == ()


def test_plate_missing_and_hand_pause():
    s = Settings(window=3, min_hits=2)
    p = AlertPolicy(s, has_scene=True)
    assert NO_PLATE in run(p, [[]] * 4)
    p = AlertPolicy(s, has_scene=True)
    hand = Detection("scene", "hand", 0.9, (0, 0, 10, 10))
    assert p.update([hand, spag(100)], 0).judged is False
    assert p.update([spag(100)], 5).judged is False          # ещё 10 с после руки не судим
    assert p.update([spag(100)], 11).judged is True


def test_muted_never_fires_and_cooldown():
    p = AlertPolicy(Settings(window=2, min_hits=1, muted=["spaghetti"], spatial=False), has_scene=False)
    assert run(p, [[spag(100)]] * 5) == []
    p = AlertPolicy(Settings(window=2, min_hits=1, cooldown_s=100, spatial=False), has_scene=False)
    fired = [c for k in range(30) for c in p.update([spag(100)], k * 10.0).fired]
    assert fired == ["spaghetti"] * 3                        # 0, 100, 200 с


def test_stale_hits_after_gap_do_not_fire():
    """Регрессия: пластину не было видно 15 минут, после кулдауна сработало на старых попаданиях."""
    p = AlertPolicy(Settings(window=5, min_hits=3, cooldown_s=600), has_scene=False)
    assert run(p, [[spag(100)]] * 5) == ["spaghetti"]
    assert p.update([], 20 + 900).fired == ()            # первый кадр после перерыва — пустой стол


def test_alert_needs_problem_on_current_frame():
    p = AlertPolicy(Settings(window=5, min_hits=3, cooldown_s=0), has_scene=False)
    fired = [p.update(d, k * 5.0).fired for k, d in enumerate([[spag(100)]] * 4 + [[]] * 2)]
    assert fired[3] == ("spaghetti",) and fired[4] == () and fired[5] == ()     # без кулдауна — всё равно тихо


def test_gap_resets_history():
    p = AlertPolicy(Settings(window=5, min_hits=3, interval_s=5), has_scene=False)
    run(p, [[spag(100)]] * 3)                              # место + два подтверждения
    assert p.update([spag(100)], 10 + 120).fired == ()    # два старых + одно новое — не три


def test_hand_resets_history():
    p = AlertPolicy(Settings(window=5, min_hits=3, hand_hold_s=10), has_scene=True)
    hand = Detection("scene", "hand", 0.9, (0, 0, 50, 50))
    plate = Detection("scene", "pei_plate", 0.9, (0, 0, 640, 480))
    seq = [[spag(100), plate]] * 2 + [[hand, plate]] + [[spag(100), plate]] * 3
    fired = [f for k, d in enumerate(seq) for f in p.update(d, k * 5.0).fired]
    assert fired == []                                     # после руки история с нуля
    assert p.update([spag(100), plate], 30.0).fired == ()
    assert p.update([spag(100), plate], 35.0).fired == ("spaghetti",)


def test_print_background_ignored_but_new_garbage_and_spaghetti_count():
    """Тестовая полоска AD5M: «мусор» с первых минут на одном месте — фон; новый мусор и спагетти — нет."""
    p = AlertPolicy(Settings(window=3, min_hits=2, notify_on=["garbage"], baseline_s=60), has_scene=False)
    line = Detection("defects", "garbage", 0.95, (236, 238, 284, 262))
    for k in range(20):                                  # 100 с: полоска всё время на месте
        v = p.update([line], k * 5.0)
        assert "garbage" not in v.problems and v.ignored == (line,)
    new = Detection("defects", "garbage", 0.9, (400, 300, 440, 330))
    assert [p.update([line, new], 100 + k * 5.0).fired for k in range(3)][-1] == ("garbage",)
    q = AlertPolicy(Settings(window=3, min_hits=2, baseline_s=60), has_scene=False)
    assert "spaghetti" in q.update([spag(100)], 0.0).problems   # сбой в первые минуты — не фон
