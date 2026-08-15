from types import SimpleNamespace

from state_machine.driving_mode_monitor import (
    DrivingModeMonitor,
    diagnostic_description,
    diagnostic_detail_text,
    obstacle_mode_for_class,
)
from state_machine.head_to_head_state_machine import (
    HeadToHeadStateMachine,
    nearest_ahead,
)


def obstacle(obstacle_id, s):
    return SimpleNamespace(id=obstacle_id, s_center=s)


def test_monitor_keeps_unknown_distinct_from_dynamic():
    mode, _ = obstacle_mode_for_class("TRAILING", "UNKNOWN")
    assert mode == "OBSTACLE_TRAILING_UNKNOWN"


def test_monitor_reports_confirmed_dynamic_only():
    mode, _ = obstacle_mode_for_class("OVERTAKE", "DYNAMIC")
    assert mode == "DYNAMIC_OBSTACLE_OVERTAKE"


def test_monitor_describes_force_trailing_training_gate():
    description = diagnostic_description("predictor", "TRAINING")
    assert "force_trailing" in description
    assert "랩 수 부족" in description


def test_monitor_preserves_grid_failure_location():
    detail = diagnostic_detail_text(
        {"path_index": 7, "x_m": 1.25, "y_m": -0.3})
    assert "path_index=7" in detail
    assert "x_m=1.25" in detail
    assert "y_m=-0.3" in detail


def test_monitor_logs_diagnostic_only_when_status_changes():
    class Logger:
        def __init__(self):
            self.messages = []

        def info(self, message):
            self.messages.append(message)

        def warn(self, message, **_kwargs):
            self.messages.append(message)

    logger = Logger()
    monitor = SimpleNamespace(
        last_diagnostic_status={},
        get_logger=lambda: logger,
    )
    training = SimpleNamespace(data=(
        '{"source":"predictor","status":"TRAINING",'
        '"detail":{"lap_count":0.2}}'
    ))
    ready = SimpleNamespace(data=(
        '{"source":"predictor","status":"LEARNED_READY",'
        '"detail":{"lap_count":0.5}}'
    ))

    DrivingModeMonitor.diagnostic_cb(monitor, training)
    DrivingModeMonitor.diagnostic_cb(monitor, training)
    DrivingModeMonitor.diagnostic_cb(monitor, ready)

    assert len(logger.messages) == 2
    assert "TRAINING" in logger.messages[0]
    assert "LEARNED_READY" in logger.messages[1]


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


def test_held_target_is_propagated_but_invisible_frames_do_not_extend_ttl():
    clock = SimpleNamespace(now=0.0)
    machine = SimpleNamespace(
        _last_trailing_target=None,
        _last_trailing_target_at=None,
        trailing_target_hold_sec=0.75,
        track_length=20.0,
        now_sec=lambda: clock.now,
    )
    target = SimpleNamespace(
        s_start=4.8,
        s_end=5.2,
        s_center=5.0,
        vs=2.0,
        is_visible=True,
    )
    HeadToHeadStateMachine._remember_trailing_target(machine, target)

    clock.now = 0.5
    held = HeadToHeadStateMachine._held_trailing_target(machine)
    assert held.s_center == 6.0
    assert held.is_visible is False

    HeadToHeadStateMachine._remember_trailing_target(machine, held)
    clock.now = 0.8
    assert HeadToHeadStateMachine._held_trailing_target(machine) is None


def free_dbg(*records):
    return {"is_init": True, "n_obs": len(records), "obs": list(records)}


def blocked(branch, obstacle_id=1):
    return {"id": obstacle_id, "branch": branch, "blocked": True}


def cleared(branch, obstacle_id=2):
    return {"id": obstacle_id, "branch": branch, "blocked": False}


def test_beyond_path_only_block_is_ignored():
    # The whole point of the fix: with prediction off every dynamic obstacle
    # takes dyn/nopred, and one past the end of a non-closed avoidance path
    # must not refuse that path.
    wpnts = SimpleNamespace(free_dbg=free_dbg(blocked("dyn/nopred/beyond_path")))
    assert HeadToHeadStateMachine._blocked_only_beyond_path(None, wpnts) is True


def test_real_block_alongside_beyond_path_still_refuses():
    wpnts = SimpleNamespace(free_dbg=free_dbg(
        blocked("dyn/nopred/beyond_path", 1),
        blocked("static/geom", 2),
    ))
    assert HeadToHeadStateMachine._blocked_only_beyond_path(None, wpnts) is False


def test_predicted_opponent_block_still_refuses():
    # dyn/pred is the branch a running predictor uses; it must be untouched so
    # relaunching with prediction:=true behaves exactly as before.
    wpnts = SimpleNamespace(free_dbg=free_dbg(blocked("dyn/pred")))
    assert HeadToHeadStateMachine._blocked_only_beyond_path(None, wpnts) is False


def test_nothing_blocked_is_not_a_correction():
    wpnts = SimpleNamespace(free_dbg=free_dbg(cleared("static/geom")))
    assert HeadToHeadStateMachine._blocked_only_beyond_path(None, wpnts) is False


def test_uninitialised_or_missing_debug_is_not_a_correction():
    assert HeadToHeadStateMachine._blocked_only_beyond_path(
        None, SimpleNamespace(free_dbg=None)) is False
    assert HeadToHeadStateMachine._blocked_only_beyond_path(
        None, SimpleNamespace()) is False
    assert HeadToHeadStateMachine._blocked_only_beyond_path(
        None, SimpleNamespace(free_dbg={"is_init": False, "obs": [
            blocked("dyn/nopred/beyond_path")]})) is False


def test_dynamic_overtake_refused_when_planner_not_launched():
    machine = SimpleNamespace(dynamic_avoidance_enabled=False)
    assert HeadToHeadStateMachine._check_overtaking_mode(machine) is False
