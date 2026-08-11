from types import SimpleNamespace

from state_machine.driving_mode_monitor import obstacle_mode_for_class
from state_machine.head_to_head_state_machine import nearest_ahead


def obstacle(obstacle_id, s):
    return SimpleNamespace(id=obstacle_id, s_center=s)


def test_monitor_keeps_unknown_distinct_from_dynamic():
    mode, _ = obstacle_mode_for_class("TRAILING", "UNKNOWN")
    assert mode == "OBSTACLE_TRAILING_UNKNOWN"


def test_monitor_reports_confirmed_dynamic_only():
    mode, _ = obstacle_mode_for_class("OVERTAKE", "DYNAMIC")
    assert mode == "DYNAMIC_OBSTACLE_OVERTAKE"


def test_nearest_target_has_no_two_metre_cutoff():
    gap, target = nearest_ahead(
        [obstacle(1, 7.0), obstacle(2, 4.0)],
        current_s=1.0,
        track_length=20.0,
        horizon=10.0,
    )
    assert gap == 3.0
    assert target.id == 2


def test_nearest_target_handles_finish_line_wrap():
    gap, target = nearest_ahead(
        [obstacle(1, 0.4), obstacle(2, 8.0)],
        current_s=19.4,
        track_length=20.0,
        horizon=5.0,
    )
    assert abs(gap - 1.0) < 1e-9
    assert target.id == 1
