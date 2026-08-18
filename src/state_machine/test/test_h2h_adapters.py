from types import SimpleNamespace

from state_machine.driving_mode_monitor import (
    DrivingModeMonitor,
    diagnostic_description,
    diagnostic_detail_text,
    obstacle_mode_for_class,
)
from state_machine.h2h_state_machine import (
    H2HStateMachine,
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
    H2HStateMachine._remember_trailing_target(machine, target)

    clock.now = 0.5
    held = H2HStateMachine._held_trailing_target(machine)
    assert held.s_center == 6.0
    assert held.is_visible is False

    H2HStateMachine._remember_trailing_target(machine, held)
    clock.now = 0.8
    assert H2HStateMachine._held_trailing_target(machine) is None


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
    assert H2HStateMachine._blocked_only_beyond_path(None, wpnts) is True


def test_real_block_alongside_beyond_path_still_refuses():
    wpnts = SimpleNamespace(free_dbg=free_dbg(
        blocked("dyn/nopred/beyond_path", 1),
        blocked("static/geom", 2),
    ))
    assert H2HStateMachine._blocked_only_beyond_path(None, wpnts) is False


def test_predicted_opponent_block_still_refuses():
    # dyn/pred is the branch a running predictor uses; it must be untouched so
    # relaunching with prediction:=true behaves exactly as before.
    wpnts = SimpleNamespace(free_dbg=free_dbg(blocked("dyn/pred")))
    assert H2HStateMachine._blocked_only_beyond_path(None, wpnts) is False


def test_nothing_blocked_is_not_a_correction():
    wpnts = SimpleNamespace(free_dbg=free_dbg(cleared("static/geom")))
    assert H2HStateMachine._blocked_only_beyond_path(None, wpnts) is False


def test_uninitialised_or_missing_debug_is_not_a_correction():
    assert H2HStateMachine._blocked_only_beyond_path(
        None, SimpleNamespace(free_dbg=None)) is False
    assert H2HStateMachine._blocked_only_beyond_path(
        None, SimpleNamespace()) is False
    assert H2HStateMachine._blocked_only_beyond_path(
        None, SimpleNamespace(free_dbg={"is_init": False, "obs": [
            blocked("dyn/nopred/beyond_path")]})) is False


def test_dynamic_overtake_refused_when_planner_not_launched():
    machine = SimpleNamespace(dynamic_avoidance_enabled=False)
    assert H2HStateMachine._check_overtaking_mode(machine) is False


# --- planner-validated span (R5) ---------------------------------------------

def span_machine(merger, avoidance, wpnts_data, cur_s=5.0, max_s=21.9):
    return SimpleNamespace(
        merger=merger, cur_avoidance_wpnts=avoidance,
        cur_s=cur_s, max_s=max_s)


def test_validated_span_bounds_only_the_dynamic_cache():
    avoidance = SimpleNamespace(name='dynamic')
    machine = span_machine([11.0, 15.0], avoidance, avoidance)
    span = H2HStateMachine._planner_validated_span(machine, avoidance)
    assert span == 6.0


def test_validated_span_ignores_the_static_cache():
    # spline_node avoids a stationary obstacle and publishes no merger; its
    # path stays checked to its end exactly as before.
    avoidance = SimpleNamespace(name='dynamic')
    static = SimpleNamespace(name='static')
    machine = span_machine([11.0, 15.0], avoidance, static)
    assert H2HStateMachine._planner_validated_span(machine, static) is None


def test_validated_span_wraps_at_the_seam():
    avoidance = SimpleNamespace(name='dynamic')
    machine = span_machine([2.0, 6.0], avoidance, avoidance, cur_s=20.0)
    span = H2HStateMachine._planner_validated_span(machine, avoidance)
    assert abs(span - 3.9) < 1e-9


def test_validated_span_absent_before_the_planner_has_published():
    avoidance = SimpleNamespace(name='dynamic')
    machine = span_machine(None, avoidance, avoidance)
    assert H2HStateMachine._planner_validated_span(machine, avoidance) is None


# --- overtake sustainability: per-term verdict and source handover ------------

def make_handover(committed_static=True, committed_available=True,
                  committed_free=True, other_fresh=True, other_free=True,
                  dynamic_enabled=True):
    machine = H2HStateMachine.__new__(H2HStateMachine)
    calls = []
    machine.calls = calls
    machine.logged = []
    machine.static_overtaking_mode = committed_static
    machine.dynamic_avoidance_enabled = dynamic_enabled
    machine.cur_static_avoidance_wpnts = SimpleNamespace(
        name="static", enabled=True, stamp=None, free_dbg=None,
        hyst_timer_sec=4.0, killing_timer_sec=10.0)
    machine.cur_avoidance_wpnts = SimpleNamespace(
        name="dynamic", enabled=True, stamp=None, free_dbg=None,
        hyst_timer_sec=1.0, killing_timer_sec=2.0)
    machine.static_avoidance_wpnts = "static_src"
    machine.avoidance_wpnts = "dynamic_src"
    committed = (machine.cur_static_avoidance_wpnts if committed_static
                 else machine.cur_avoidance_wpnts)

    def availability(_src, data):
        assert data is committed, "availability is only for the committed cache"
        calls.append("avail:" + data.name)
        return committed_available

    def latest(_src, data):
        assert data is not committed, "latest_wpnts is only for the handover"
        calls.append("latest:" + data.name)
        return other_fresh

    def free(data):
        calls.append("free:" + data.name)
        return committed_free if data is committed else other_free

    machine._check_availability = availability
    machine._check_latest_wpnts = latest
    machine._check_free_frenet = free
    machine._check_on_spline = lambda _d: False
    machine.now_sec = lambda: 0.0
    machine.get_logger = lambda: SimpleNamespace(
        info=lambda msg, **k: machine.logged.append(msg))
    return machine


def test_sustains_without_touching_the_other_source():
    machine = make_handover()
    assert machine._check_overtaking_mode_sustainability() is True
    assert machine.calls == ["avail:static", "free:static"]
    assert machine.logged == []


def test_both_terms_are_evaluated_so_the_refusal_can_name_one():
    # Short-circuiting would skip the free check and the log could only ever
    # say "unavailable", which is what made every abort look identical.
    machine = make_handover(committed_available=False, committed_free=False,
                            other_fresh=False)
    assert machine._check_overtaking_mode_sustainability() is False
    assert "avail:static" in machine.calls and "free:static" in machine.calls
    assert "available=False" in machine.logged[0]
    assert "path_free=False" in machine.logged[0]


def test_refusal_names_the_blocking_obstacle():
    machine = make_handover(committed_free=False, other_fresh=False)
    machine.cur_static_avoidance_wpnts.free_dbg = {"obs": [
        {"id": 1000000, "branch": "dyn/pred", "blocked": True,
         "free_dist": -0.07, "gap": 2.4}]}
    machine._check_overtaking_mode_sustainability()
    assert "obs 1000000" in machine.logged[0]
    assert "dyn/pred" in machine.logged[0]


def test_refusal_says_when_the_planner_never_published():
    machine = make_handover(committed_available=False, other_fresh=False)
    machine._check_overtaking_mode_sustainability()
    assert "never published" in machine.logged[0]


def test_static_hands_over_to_dynamic_when_the_target_starts_moving():
    machine = make_handover(committed_static=True, committed_free=False)
    assert machine._check_overtaking_mode_sustainability() is True
    assert machine.static_overtaking_mode is False
    assert machine.calls[-2:] == ["latest:dynamic", "free:dynamic"]


def test_dynamic_hands_over_to_static_when_the_target_stops():
    machine = make_handover(committed_static=False, committed_free=False)
    assert machine._check_overtaking_mode_sustainability() is True
    assert machine.static_overtaking_mode is True
    assert machine.calls[-2:] == ["latest:static", "free:static"]


def test_handover_refused_when_the_other_path_is_blocked():
    machine = make_handover(committed_free=False, other_free=False)
    assert machine._check_overtaking_mode_sustainability() is False
    assert machine.static_overtaking_mode is True


def test_no_handover_to_dynamic_when_the_planner_is_not_launched():
    machine = make_handover(committed_free=False, dynamic_enabled=False)
    assert machine._check_overtaking_mode_sustainability() is False
    assert not any(c.startswith("latest") for c in machine.calls)
