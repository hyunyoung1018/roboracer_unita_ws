"""Entering OVERTAKE must not be a discontinuity.

Covers H2HController. Both overrides answer the shared class's own values
whenever the head-to-head condition is absent, which is what keeps
time_trials - running the unmodified Controller through controller_manager -
untouched by either of them.

The run that forced both, 2026-08-21 over 314 s: 50 OVERTAKE entries and 135
exits, one source change every 1.7 s, take-to-drop median 0.41 s. Each entry
ended gap control and stepped the heading gain by 1.54x on the same tick the
path jumped about 0.3 m sideways.
"""

from controller.combined.src.Controller import Controller
from controller.h2h_controller import H2HController


def controller(state, opponent=(0.0, 0.0, 0.0, False, True),
               trail_while_overtaking=True, ramp=0.5):
    """A real instance, uninitialised.

    __new__ rather than a stub because the overrides call the zero-argument
    super(), which needs self to be an instance of the class. Nothing the
    constructor sets up is reached by either method under test.
    """
    c = H2HController.__new__(H2HController)
    c.state = state
    c.opponent = list(opponent) if opponent is not None else None
    c.trail_while_overtaking = trail_while_overtaking
    c.overtake_gain_ramp_sec = ramp
    c._overtake_gain = 1.0
    return c


# ------------------------------------------------- holding the gap through it
def test_trailing_still_trails():
    assert controller("TRAILING").trailing_active()


def test_overtaking_trails_too():
    """The case that used to release the car onto the path's planned speed."""
    assert controller("OVERTAKE").trailing_active()


def test_overtaking_does_not_trail_when_switched_off():
    # The A/B. With it false this class answers exactly what its parent does.
    assert not controller("OVERTAKE", trail_while_overtaking=False).trailing_active()


def test_no_opponent_never_trails():
    # trailing_targets empty: nothing to hold a gap to, in any state. This is
    # what makes the override inert in time trials even if it ever ran there.
    for state in ("TRAILING", "OVERTAKE", "RACELINE"):
        assert not controller(state, opponent=None).trailing_active()


def test_other_states_are_unchanged():
    for state in ("RACELINE", "RECOVERY", "FTGONLY", "START", "LOSTLINE"):
        assert not controller(state).trailing_active()


def test_the_shared_class_is_not_widened():
    # The parent must keep refusing OVERTAKE, or time trials changes with it.
    parent = Controller.__new__(Controller)
    parent.state = "OVERTAKE"
    parent.opponent = [0.0, 0.0, 0.0, False, True]
    assert not parent.trailing_active()


# --------------------------------------------------------- ramping the gain
def test_the_gain_starts_where_the_shared_class_starts():
    assert controller("RACELINE").overtake_gain_scale(0.02) == 1.0


def test_the_gain_does_not_step_on_entry():
    """One tick into OVERTAKE the gain has barely moved.

    The shared class drops to 0.65 on this tick. At 50 Hz and a 0.5 s ramp the
    first step is 4% of the way, so an evasion abandoned in 0.41 s never sees
    most of the reduction - and never has to hand it back.
    """
    c = controller("OVERTAKE")
    first = c.overtake_gain_scale(0.02)
    assert 0.98 < first < 1.0


def test_the_gain_reaches_the_shared_value_if_the_manoeuvre_lasts():
    c = controller("OVERTAKE")
    for _ in range(200):  # 4 s at 50 Hz
        value = c.overtake_gain_scale(0.02)
    assert abs(value - 0.65) < 0.01


def test_the_gain_comes_back_on_exit():
    c = controller("OVERTAKE")
    for _ in range(200):
        c.overtake_gain_scale(0.02)
    c.state = "RACELINE"
    for _ in range(200):
        value = c.overtake_gain_scale(0.02)
    assert abs(value - 1.0) < 0.01


def test_a_zero_ramp_restores_the_shared_step():
    c = controller("OVERTAKE", ramp=0.0)
    assert c.overtake_gain_scale(0.02) == 0.65
    c.state = "RACELINE"
    assert c.overtake_gain_scale(0.02) == 1.0


def test_a_repeated_stamp_holds_the_gain_rather_than_freezing_it_low():
    # dt comes from message stamps; a repeated one must not be treated as a
    # step of zero seconds towards the target, nor divide anything.
    c = controller("OVERTAKE")
    c.overtake_gain_scale(0.02)
    held = c._overtake_gain
    assert c.overtake_gain_scale(0.0) == held
    assert c.overtake_gain_scale(-1.0) == held


def test_the_ramp_is_in_seconds_not_ticks():
    """Halving dt and doubling the count reaches the same place."""
    slow = controller("OVERTAKE")
    for _ in range(25):
        slow.overtake_gain_scale(0.02)
    fast = controller("OVERTAKE")
    for _ in range(50):
        fast.overtake_gain_scale(0.01)
    assert abs(slow._overtake_gain - fast._overtake_gain) < 0.01
