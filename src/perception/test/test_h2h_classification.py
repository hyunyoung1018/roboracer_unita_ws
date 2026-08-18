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
