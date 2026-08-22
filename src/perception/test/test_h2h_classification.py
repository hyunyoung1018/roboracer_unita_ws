"""Classifying a box that wobbles, without believing a car that moves.

The shared tracker reads velocity off one frame-to-frame difference, so the
few centimetres the L-shape fit moves every frame as the viewing angle changes
became 0.6 m/s at 20 Hz and a stationary box read DYNAMIC. Its own median
filter is the cure, but it only runs once a track is ALREADY static - the
bootstrap this override breaks.

The wobble and drift figures here are measured, off the run of 2026-08-18:
apparent d moved 0.25 to 0.37 m over the 2 s approach from 5.5 m to 1 m, on
every one of ~90 tracks.
"""

from types import SimpleNamespace

import pytest

from perception.h2h_tracking_node import HeadToHeadTrackingNode

TRACK = 20.0
DT = 0.05          # 20 Hz
PARAMS = {
    'static_position_samples': 10,
    'static_min_samples': 6,
    'static_speed_threshold': 0.15,
    'static_classification_median': True,
    'static_confirm_frames': 1,
    'dynamic_confirm_frames': 4,
    'static_speed_hysteresis_mps': 0.10,
}


def node(**overrides):
    n = HeadToHeadTrackingNode.__new__(HeadToHeadTrackingNode)
    params = dict(PARAMS, **overrides)
    n.get_parameter = lambda name: SimpleNamespace(value=params[name])
    n.track_length = TRACK
    return n


def track(s_samples, d_samples):
    return SimpleNamespace(s_history=list(s_samples), d_history=list(d_samples))


def speed(n, tr):
    return n._classification_speed(tr, DT)


def wobble(n_samples, amplitude, seed=0):
    """Alternating +/- wobble: zero mean, no drift."""
    return [amplitude * (1 if i % 2 == 0 else -1) for i in range(n_samples)]


# --------------------------------------------------------------- stationary
def test_a_wobbling_box_reads_as_stopped():
    # 3 cm frame-to-frame wobble: 0.6 m/s to a single difference, and the
    # reason boxes were being trailed as opponents.
    n = node()
    tr = track([5.0 + w for w in wobble(20, 0.03)],
               [0.20 + w for w in wobble(20, 0.03)])
    assert speed(n, tr) < 0.15


def test_a_single_bad_fit_is_rejected_outright():
    n = node()
    s = [5.0] * 20
    d = [0.20] * 20
    d[-3] = 0.60          # one 40 cm outlier
    assert speed(n, track(s, d)) < 0.15


# ------------------------------------------------------------------ moving
def test_an_opponent_at_walking_pace_still_reads_as_moving():
    n = node()
    s = [5.0 + 1.0 * DT * i for i in range(20)]     # 1.0 m/s
    assert speed(n, track(s, [0.0] * 20)) == pytest.approx(1.0, rel=0.05)


def test_a_slow_opponent_is_still_above_the_threshold():
    n = node()
    s = [5.0 + 0.4 * DT * i for i in range(20)]     # 0.4 m/s
    assert speed(n, track(s, [0.0] * 20)) > 0.15


def test_lateral_motion_counts_too():
    n = node()
    d = [0.0 + 0.5 * DT * i for i in range(20)]
    assert speed(n, track([5.0] * 20, d)) == pytest.approx(0.5, rel=0.05)


# ------------------------------------------------------------------- wrap
def test_speed_is_correct_across_the_finish_line():
    n = node()
    s = [(19.5 + 1.0 * DT * i) % TRACK for i in range(20)]
    assert speed(n, track(s, [0.0] * 20)) == pytest.approx(1.0, rel=0.05)


def test_a_box_sitting_on_the_finish_line_is_not_moving():
    n = node()
    s = [(19.99 + w) % TRACK for w in wobble(20, 0.03)]
    assert speed(n, track(s, [0.0] * 20)) < 0.15


# ------------------------------------------------------- bootstrap and gates
def test_classifiable_at_the_same_frame_count_as_before():
    # static_min_samples is 6; the estimate must exist by then, not later.
    n = node()
    tr = track([5.0] * 6, [0.2] * 6)
    assert speed(n, tr) is not None


def test_no_verdict_from_a_single_frame():
    n = node()
    assert speed(n, track([5.0], [0.2])) is None


def test_no_verdict_without_a_time_base():
    n = node()
    assert n._classification_speed(track([5.0] * 20, [0.2] * 20), 0.0) is None


# ------------------------------------------------------------ the real drift
def test_the_measured_approach_drift_is_reported_honestly():
    """0.30 m of apparent lateral drift over 2 s is 0.15 m/s, and it is real.

    This is the systematic component measured on every track in the run - not
    wobble, since every obstacle drifted the same way. Medians remove the
    wobble around it; they do not and must not remove the drift itself, so
    this still lands on the threshold. Filtering was necessary here, not
    sufficient - see the note in the commit.
    """
    n = node()
    d = [0.15 - 0.15 * DT * i for i in range(20)]   # 0.15 m/s of drift
    assert speed(n, track([5.0] * 20, d)) == pytest.approx(0.15, rel=0.05)


# ------------------------------------------------------------- hysteresis
def confirming(n, was_static, speeds):
    """Feed a sequence of classification speeds and return the class each frame."""
    tr = SimpleNamespace(obstacle=SimpleNamespace(is_static=was_static),
                         class_streak=0)
    out = []
    for value in speeds:
        verdict = n._confirmed_class(tr, value)
        tr.obstacle.is_static = verdict
        out.append(verdict)
    return out


def test_one_fast_frame_does_not_unstick_a_box():
    # The chatter this exists to stop: obs 50 went STA -> DYN on a single frame.
    n = node()
    assert confirming(n, True, [0.02, 0.9, 0.02, 0.02]) == [True] * 4


def test_leaving_static_takes_the_full_count():
    n = node()
    # dynamic_confirm_frames is 4, and the speed has to clear 0.15 + 0.10.
    assert confirming(n, True, [0.9] * 5) == [True, True, True, False, False]


def test_becoming_static_is_not_delayed():
    # static_confirm_frames is 1: entering STATIC costs exactly what it did
    # before, because it is already the rare direction on this car.
    n = node()
    assert confirming(n, False, [0.02] * 3) == [True, True, True]


def test_the_streak_resets_on_a_disagreeing_frame():
    n = node()
    # Three fast frames, one slow, then fast again: never reaches four in a row.
    assert confirming(n, True, [0.9, 0.9, 0.9, 0.02, 0.9, 0.9, 0.9]) == [True] * 7


def test_the_dead_band_argues_for_nothing():
    # 0.20 is over the threshold but under threshold + band: no evidence.
    n = node()
    assert confirming(n, True, [0.20] * 10) == [True] * 10
    assert confirming(n, False, [0.20] * 10) == [False] * 10


def test_a_real_opponent_is_confirmed_promptly():
    n = node()
    # 1 m/s clears the band by a mile; four frames is 0.2 s at 20 Hz.
    assert confirming(n, True, [1.0] * 4)[-1] is False


def test_hysteresis_is_tunable_to_off():
    n = node(static_confirm_frames=1, dynamic_confirm_frames=1,
             static_speed_hysteresis_mps=0.0)
    assert confirming(n, True, [0.9]) == [False]


# --- a car that stops is still a car --------------------------------------
#
# The extent floor is a claim about what the object physically IS. It used to
# be gated on the LIVE class, so the moment a stopped opponent was reclassified
# STATIC the floor was released and the parent's slow leak took the believed
# width from 0.40 m to 0.20 m in two seconds - and every planner downstream
# subtracts that width from the room either side.
#
# Latched on having been the SELECTED OPPONENT, never on reading DYNAMIC:
# reading DYNAMIC is nearly free (39 of 41 tracks did it for their whole life
# on 2026-08-19), so that would put the car floor under every box and inflate
# a 0.15 m cone to 0.40 m.

EXTENT_PARAMS = dict(
    PARAMS,
    dynamic_extent_min_m=0.202,
    dynamic_extent_max_m=0.42,
    dynamic_extent_decay_mps=0.5,
    opponent_car_size_hold_sec=3.0,
    extent_max_m=0.50,
    extent_decay_mps=0.05,
)


def extent_node(selected=None, **overrides):
    n = node(**dict(EXTENT_PARAMS, **overrides))
    n._opponent_track_id = selected
    return n


def extent_track(is_static, history=10, extents=None, was_opponent=False,
                 obstacle_id=7, stamp=0.0, opponent_age=0.0):
    obstacle = SimpleNamespace(id=obstacle_id, is_static=is_static,
                               s_center=3.0, d_center=0.0, vs=0.0, vd=0.0)
    t = SimpleNamespace(
        track_id=obstacle_id, obstacle=obstacle, stamp=stamp, missed=0,
        s_history=[3.0] * history, d_history=[0.0] * history,
        extents=list(extents if extents is not None else [0.202] * 4))
    if was_opponent:
        # `opponent_age` seconds since the selector last held this track.
        t.was_opponent_at = stamp - opponent_age
    return t


def extent_measurement(half=0.10):
    """A pillar-sized detection: what the lidar actually returns off a car."""
    return SimpleNamespace(id=7, s_start=3.0 - half, s_end=3.0 + half,
                           d_right=-half, d_left=half, s_center=3.0,
                           d_center=0.0, size=2 * half)


def test_a_stopped_opponent_keeps_the_car_floor():
    n = extent_node(selected=None)      # released, but it WAS the opponent
    stopped = extent_track(is_static=True, was_opponent=True)
    for _ in range(200):                       # ten seconds
        stopped.extents = n._hold_extents(stopped, extent_measurement(), DT)
    assert min(stopped.extents) == pytest.approx(0.202)


def test_without_the_latch_it_decays_to_the_pillar():
    """The bug, stated as a measurement."""
    n = extent_node(selected=None)
    stopped = extent_track(is_static=True, was_opponent=False)
    for _ in range(200):
        stopped.extents = n._hold_extents(stopped, extent_measurement(), DT)
    assert min(stopped.extents) == pytest.approx(0.10)


def test_a_box_reading_dynamic_is_still_only_a_box():
    """The 2026-08-22 failure. DYNAMIC is nearly free; a car footprint is not.

    Five obstacles at five different gaps, every one of them "non-static",
    every one reported 0.404 m wide against a real 0.15 to 0.20 - and on a
    course 1.30 m across the corridor closed:

        no room either side: left 0.69 m (taken), right -0.37 m (taken)
    """
    n = extent_node(selected=None)
    box = extent_track(is_static=False, was_opponent=False, obstacle_id=39,
                       extents=[0.075] * 4)
    for _ in range(200):
        box.extents = n._hold_extents(box, extent_measurement(half=0.075), DT)
    assert max(box.extents) == pytest.approx(0.075)


def test_only_the_selected_opponent_gets_the_car_floor():
    n = extent_node(selected=7)
    opponent = extent_track(is_static=False, obstacle_id=7, extents=[0.075] * 4)
    other = extent_track(is_static=False, obstacle_id=39, extents=[0.075] * 4)
    for _ in range(40):
        opponent.extents = n._hold_extents(
            opponent, extent_measurement(half=0.075), DT)
        other.extents = n._hold_extents(
            other, extent_measurement(half=0.075), DT)
    assert min(opponent.extents) == pytest.approx(0.202)
    assert max(other.extents) == pytest.approx(0.075)


def test_a_moving_opponent_is_unchanged():
    n = extent_node(selected=7)
    moving = extent_track(is_static=False, was_opponent=True)
    for _ in range(200):
        moving.extents = n._hold_extents(moving, extent_measurement(), DT)
    assert min(moving.extents) == pytest.approx(0.202)


# --- the ceiling bounds memory, not measurement ----------------------------
#
# A 0.50 m box that was briefly mis-acquired came out at 0.42: the measurement
# said 0.25 m per side and the ceiling clamped it to 0.21. Reporting an
# obstacle smaller than it was measured hands out clearance the car does not
# have.

def big_measurement(half):
    return SimpleNamespace(id=7, s_start=3.0 - half, s_end=3.0 + half,
                           d_right=-half, d_left=half, s_center=3.0,
                           d_center=0.0, size=2 * half)


def test_a_box_bigger_than_a_car_is_not_shrunk_to_car_size():
    n = extent_node(selected=7)
    track = extent_track(is_static=True, was_opponent=True)
    held = n._hold_extents(track, big_measurement(0.25), 0.05)
    assert all(h == pytest.approx(0.25) for h in held)


def test_the_pillar_is_still_raised_to_the_car_floor():
    n = extent_node(selected=7)
    track = extent_track(is_static=True, was_opponent=True)
    held = n._hold_extents(track, extent_measurement(half=0.10), 0.05)
    assert all(h == pytest.approx(0.202) for h in held)


def test_the_ceiling_still_bounds_what_is_only_remembered():
    """A one-frame 0.9 m fit is believed while it is measured, then decays -
    and stops at the ceiling rather than staying at 0.9 forever."""
    n = extent_node(selected=7)
    track = extent_track(is_static=True, was_opponent=True,
                         extents=[0.9] * 4)
    held = n._hold_extents(track, extent_measurement(half=0.10), 0.05)
    assert all(h == pytest.approx(0.21) for h in held)


# --- the latch expires -----------------------------------------------------

def test_a_stopped_opponent_is_still_a_car_moments_later():
    n = extent_node(selected=None)
    track = extent_track(is_static=True, was_opponent=True, opponent_age=1.0)
    assert n._believed_to_be_a_car(track)


def test_a_stale_acquisition_stops_marking_a_box_as_a_car():
    n = extent_node(selected=None)
    track = extent_track(is_static=True, was_opponent=True, opponent_age=9.0)
    assert not n._believed_to_be_a_car(track)


def test_the_clock_does_not_run_while_the_track_is_still_the_opponent():
    n = extent_node(selected=7)
    track = extent_track(is_static=True, was_opponent=True, opponent_age=9.0)
    assert n._believed_to_be_a_car(track)


def test_a_track_that_was_never_the_opponent_has_no_latch():
    n = extent_node(selected=None)
    track = extent_track(is_static=True, was_opponent=False)
    assert not n._believed_to_be_a_car(track)
