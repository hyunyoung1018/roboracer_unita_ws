"""Trailing must survive the state machine arming a static evasion.

H2HController.trailing_active() widens trailing to OVERTAKE, but it reads
self.opponent, which is BehaviorStrategy.trailing_targets - and the shared node
fills that list only in TRAILING. So the override could never fire: the target
it needs was cleared one node upstream, and entering OVERTAKE still released
the car from the trailing command onto the path's planned speed.

Measured 2026-08-21 over 314 s: 50 OVERTAKE entries, each an acceleration from
about 2.5 m/s onto a 4 m/s path with the opponent two metres ahead.
"""

from types import SimpleNamespace

from state_machine.h2h_state_machine import H2HStateMachine
from state_machine.states_types import StateType


def opponent(s=6.0, d=0.0, obstacle_id=41):
    return SimpleNamespace(id=obstacle_id, s_center=s, d_center=d, vs=2.0,
                           s_start=s - 0.2, s_end=s + 0.2,
                           is_visible=True, is_static=False)


def machine(state, static_mode, opponents=(), shared_targets=()):
    m = H2HStateMachine.__new__(H2HStateMachine)
    m.cur_state = state
    m.static_overtaking_mode = static_mode
    m.behavior_strategy = SimpleNamespace(trailing_targets=list(shared_targets))
    m._opponent_obstacles = list(opponents)
    m._opponent_stamp = 0.0
    m.opponent_stream_timeout_sec = 0.5
    m.track_length = 20.0
    m.max_s = 20.0
    m.cur_s = 0.0
    m.interest_horizon_m = 9.0
    m.trailing_lateral_threshold_m = 0.6
    m.now_sec = lambda: 0.0
    m.local_wpnts_src = state
    m.get_farthest_target = lambda src: (list(shared_targets), src)
    return m


def assign(m):
    H2HStateMachine.assign_trailing_target(m)
    return m.behavior_strategy.trailing_targets


def test_a_static_evasion_still_names_the_opponent():
    """The fix. OVERTAKE on the static cache keeps a target to hold a gap to."""
    m = machine(StateType.OVERTAKE, static_mode=True, opponents=[opponent()])
    targets = assign(m)
    assert len(targets) == 1 and int(targets[0].id) == 41


def test_the_shared_answer_wins_when_it_has_one():
    """TRAILING is unchanged: whatever get_farthest_target said stands."""
    theirs = opponent(obstacle_id=7)
    m = machine(StateType.TRAILING, static_mode=False, opponents=[opponent()],
                shared_targets=[theirs])
    assert assign(m) == [theirs]


def test_a_lane_change_evasion_is_left_alone():
    """That path is planned around the opponent's future; a fixed gap to it
    would forbid the pass the path exists to make."""
    m = machine(StateType.OVERTAKE, static_mode=False, opponents=[opponent()])
    assert assign(m) == []


def test_no_opponent_means_no_target():
    m = machine(StateType.OVERTAKE, static_mode=True, opponents=[])
    assert assign(m) == []


def test_a_stale_opponent_stream_names_nobody():
    m = machine(StateType.OVERTAKE, static_mode=True, opponents=[opponent()])
    m._opponent_stamp = -10.0
    assert assign(m) == []


def test_every_other_state_is_untouched():
    for state in (StateType.RACELINE, StateType.RECOVERY, StateType.FTGONLY):
        m = machine(state, static_mode=True, opponents=[opponent()])
        assert assign(m) == []
