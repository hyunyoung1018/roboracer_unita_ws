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
