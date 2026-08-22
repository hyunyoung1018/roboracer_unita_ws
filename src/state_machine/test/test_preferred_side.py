"""The map gets to say which way an overtake is taken.

_select_side picks whichever side has the most room at its tightest point.
On a corner that is the outside, every time - the inside is shorter and is
where an opponent running wide leaves a gap, and neither is anything the room
comparison can see. It also re-decides every tick off the opponent's measured
edge, which wanders about 10 cm frame to frame on this car, so when the two
sides are within that of each other the answer flips.

ot_sectors.yaml carries preferred_side per overtaking sector; the state machine
resolves it for the car's s and publishes it. "auto", and every sector that
does not set it, is the old behaviour exactly.
"""

from types import SimpleNamespace

from state_machine.state_machine_node import StateMachine


def machine(sides=(), cur_s=0.0, wpnt_dist=0.1):
    m = StateMachine.__new__(StateMachine)
    m.overtake_sides = [list(s) for s in sides]
    m.cur_s = cur_s
    m.waypoints_dist = wpnt_dist
    return m


def side_at(sides, cur_s):
    return StateMachine._preferred_side_here(machine(sides, cur_s))


def test_a_sector_that_names_a_side_gets_it():
    # index = s / waypoints_dist, so s=5.0 at 0.1 m spacing is waypoint 50.
    assert side_at([[0, 100, "left"]], cur_s=5.0) == "left"


def test_outside_every_sector_is_auto():
    assert side_at([[0, 100, "left"]], cur_s=50.0) == "auto"


def test_the_right_sector_of_several_answers():
    sides = [[0, 100, "left"], [101, 200, "right"], [201, 300, "auto"]]
    assert side_at(sides, cur_s=5.0) == "left"
    assert side_at(sides, cur_s=15.0) == "right"
    assert side_at(sides, cur_s=25.0) == "auto"


def test_no_sector_table_is_auto():
    assert side_at([], cur_s=5.0) == "auto"


def test_a_degenerate_spacing_does_not_divide_by_zero():
    assert StateMachine._preferred_side_here(
        machine([[0, 100, "left"]], cur_s=5.0, wpnt_dist=0.0)) == "auto"


# --- yeet_factor: a speed advantage, where it is safe to have one ----------
#
# Passing a moving car needs to be faster than it; a ceiling at exactly the
# raceline speed forbids that. But the raceline ceiling exists because ggv.csv
# is a flat 12.0 placeholder, and without it the path was planned near twice
# the speed of the line it departs from - the car accelerated into a bend with
# no grip budget and hit the wall.
#
# Reconciled by WHERE it applies: full lift where the path bends no more than
# the raceline (the straight the pass is made on), tapering to none at the
# path's own sharpest bend (the corner exit that crashed).

import numpy as np


def lifted(raceline_v, kappa, raceline_kappa, yeet):
    """The lift as update_velocity computes it."""
    extra = np.maximum(0.0, np.abs(kappa) - raceline_kappa)
    worst = extra.max()
    straightness = (1.0 - extra / worst) if worst > 1e-6 else np.ones_like(extra)
    return raceline_v * (1.0 + (yeet - 1.0) * straightness)


def test_a_path_no_sharper_than_the_line_gets_the_whole_lift():
    v = np.array([3.0, 3.0, 3.0])
    out = lifted(v, kappa=np.array([0.2, 0.2, 0.2]),
                 raceline_kappa=np.array([0.2, 0.2, 0.2]), yeet=1.25)
    assert np.allclose(out, 3.75)


def test_the_sharpest_bend_gets_none_of_it():
    v = np.array([3.0, 3.0, 3.0])
    out = lifted(v, kappa=np.array([0.2, 0.6, 0.2]),
                 raceline_kappa=np.array([0.2, 0.2, 0.2]), yeet=1.25)
    assert out[1] == 3.0                      # the swerve keeps the ceiling
    assert np.allclose(out[[0, 2]], 3.75)     # the straight does not


def test_the_taper_is_monotonic():
    v = np.ones(5) * 3.0
    out = lifted(v, kappa=np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
                 raceline_kappa=np.full(5, 0.2), yeet=1.25)
    assert list(out) == sorted(out, reverse=True)


def test_yeet_one_is_the_raceline_ceiling_exactly():
    v = np.array([3.0, 3.0])
    out = lifted(v, kappa=np.array([0.2, 0.9]),
                 raceline_kappa=np.array([0.2, 0.2]), yeet=1.0)
    assert np.allclose(out, 3.0)


def test_the_clamp_bounds_what_a_map_can_ask_for():
    for asked, expected in ((0.5, 1.0), (1.0, 1.0), (1.25, 1.25),
                            (2.0, 1.5), (10.0, 1.5)):
        assert float(np.clip(asked, 1.0, 1.5)) == expected
