"""FTG may brush an obstacle. It may not leave the course.

The gap search reads a wall and a cone as the same thing, because to a lidar
they are, so it can be steered off the track by a gap in the barriers exactly
as readily as around an obstacle. Which is the wrong way round for a race
where contact costs nothing and going off costs the run.

With ftg_use_map set, the map decides where the car MAY go and the scan
decides where it CAN go right now. The map is the hard constraint; obstacle
clearance is only a preference among what survives it. And when nothing
survives, the heading that stays on the map longest comes back - never
nothing, because stopping is the one outcome this must not produce.
"""

import math

import numpy as np

from controller.ftg.ftg import FTG


BEAMS = 361               # 0.5 deg over 180 deg
ANGLE_MIN = -math.pi / 2
ANGLE_INC = math.pi / (BEAMS - 1)


class Grid:
    """Map stub: free inside a half-plane / cone the test picks."""

    ready = True

    def __init__(self, predicate):
        self.predicate = predicate
        self.calls = 0

    def first_outside_index(self, xy):
        self.calls += 1
        for index, (x, y) in enumerate(np.asarray(xy, dtype=float)):
            if not self.predicate(float(x), float(y)):
                return index
        return None


def ftg(**kwargs):
    f = FTG.__new__(FTG)
    f.node = None
    f.DEBUG = False
    f.mapping = False
    f.MAX_LIDAR_DIST = 10.0
    f.MAX_SPEED = 1.5
    f.track_width = 0.8
    f.FRONT_FOV = math.radians(45.0)
    f.SMOOTH_RAD = 0.0
    f.DISP_THRESH = 0.5
    f.BUBBLE_M = 0.3
    f.STEER_EMA = 0.0
    f.MAX_STEER = 10.0        # unclipped, so the test reads the raw choice
    f.SPEED_SCALE = 1.0
    f.USE_MAP = True
    f.HEADING_STEP_DEG = 2.0
    f.MAP_PROBE_M = 1.5
    f.MAP_PROBE_STEP_M = 0.15
    f.MIN_LIDAR_CLEARANCE_M = 0.35
    f.FORWARD_BIAS = 0.35
    f.LASER_OFFSET_X = 0.0    # rays from the origin keeps the geometry readable
    f.velocity = 0.0
    f._steer_prev = None
    f.angle_min = ANGLE_MIN
    f.angle_inc = ANGLE_INC
    f.radians_per_elem = ANGLE_INC
    f.best_pnt = f.scan_pub = f.best_gap = None
    for name, value in kwargs.items():
        setattr(f, name, value)
    f.recompute_speeds()
    return f


def scan(fn):
    """Ranges from a function of bearing, in radians."""
    return [fn(ANGLE_MIN + i * ANGLE_INC) for i in range(BEAMS)]


# The scan these tests turn on: track ahead and to the right at 2 m, and
# something long and open from -10 deg leftwards - a gap in the barriers, or
# the infield. Measured, the plain gap search steers +22 deg into it.
#
# Note what does NOT work as this fixture, because it says something about
# the search: putting the opening further left (past +30) makes the gap
# search pick -7.5 deg instead, because the disparity bubble blown at the
# near edge eats the opening it just found. The scan has to be arranged so
# the opening SURVIVES the bubble before the map has anything to refuse.
def OPEN_LEFT(a):
    return 6.0 if a > math.radians(-10) else 2.0


def steer(f, ranges, grid=None, pose=(0.0, 0.0, 0.0)):
    _, angle = f.process_lidar(ranges, ANGLE_MIN, ANGLE_INC,
                               pose=pose, grid=grid)
    return angle


# ------------------------------------------------- the case this exists for
def test_a_hole_in_the_barrier_is_not_taken():
    """Open to the left, but only the front is track.

    The scan cannot tell the two apart - the left is the longer range and the
    plain gap search takes it. The map is what refuses it.
    """
    open_left = scan(OPEN_LEFT)
    forward_only = Grid(lambda x, y: x > 0.0 and abs(y) < 0.5)

    assert steer(ftg(), open_left, grid=forward_only) < math.radians(15)
    # Same scan, no map: the gap search goes for the hole.
    assert steer(ftg(USE_MAP=False), open_left) > math.radians(20)


def test_the_open_side_is_taken_when_the_map_allows_it():
    """The map refuses; it does not choose.

    OPEN_LEFT is the wrong scan to show this with, and usefully so: its open
    region reaches across straight ahead, so "go straight" IS taking it, and
    a test on the steering angle proves nothing. Block the front instead -
    0.5 m, clear of MIN_LIDAR_CLEARANCE_M so nothing is vetoed - and the turn
    has to be earned.
    """
    blocked_ahead_open_left = scan(
        lambda a: 6.0 if a > math.radians(20) else 0.5)
    all_free = Grid(lambda x, y: True)
    assert steer(ftg(), blocked_ahead_open_left, grid=all_free) \
        > math.radians(15)


# ------------------------------------------------- the scan keeps its veto
def test_a_short_beam_is_never_chosen():
    """Measured: the score alone already does this.

    min(beam, MAP_PROBE_M) means a 0.2 m heading scores 0.2 against a clear
    one's 1.5, and no forward bias inside the FOV closes that. So this passes
    with MIN_LIDAR_CLEARANCE_M at zero, and is here to say so rather than to
    guard the veto - see the next test for where the veto actually binds.
    """
    blocked_ahead = scan(
        lambda a: 0.2 if abs(a) < math.radians(10) else 5.0)
    all_free = Grid(lambda x, y: True)
    assert abs(steer(ftg(), blocked_ahead, all_free)) > math.radians(10)
    assert abs(steer(ftg(MIN_LIDAR_CLEARANCE_M=0.0), blocked_ahead, all_free)) \
        > math.radians(10)


def test_the_veto_holds_where_the_score_cannot_reach():
    """Nothing is on the map, so the fallback picks - and it must not pick a wall.

    This is the case the veto is FOR. `longest` is bookkept only for headings
    that already cleared MIN_LIDAR_CLEARANCE_M, so a heading the lidar says is
    0.15 m from something cannot become the answer just because it happened to
    be the first one tried. Without the veto it does exactly that.
    """
    wall_at_the_left_edge = scan(
        lambda a: 0.15 if a < math.radians(-40) else 4.0)
    nothing_on_the_map = Grid(lambda x, y: False)

    kept = steer(ftg(), wall_at_the_left_edge, nothing_on_the_map)
    assert kept > math.radians(-40)

    without = steer(ftg(MIN_LIDAR_CLEARANCE_M=0.0),
                    wall_at_the_left_edge, nothing_on_the_map)
    assert without <= math.radians(-40)


# ------------------------------------------------- never nothing
def test_no_heading_on_the_map_still_returns_one():
    """Off the map everywhere: take the heading that stays on it longest."""
    clear = scan(lambda a: 5.0)
    # Free only in a wedge to the right, and only for 0.45 m.
    stub = Grid(lambda x, y: math.hypot(x, y) < 0.45 and y < -0.1)
    angle = steer(ftg(), clear, grid=stub)
    assert angle < 0.0

    # And it is a real choice, not the straight-ahead default.
    assert angle != 0.0


def test_a_map_that_is_not_ready_leaves_the_gap_search_alone():
    class NotReady(Grid):
        ready = False

    open_left = scan(OPEN_LEFT)
    grid = NotReady(lambda x, y: False)
    assert steer(ftg(), open_left, grid=grid) > math.radians(20)
    assert grid.calls == 0


def test_no_pose_leaves_the_gap_search_alone():
    open_left = scan(OPEN_LEFT)
    grid = Grid(lambda x, y: False)
    assert steer(ftg(), open_left, grid=grid, pose=None) > math.radians(20)
    assert grid.calls == 0


def test_use_map_off_is_the_old_behaviour_exactly():
    open_left = scan(OPEN_LEFT)
    grid = Grid(lambda x, y: False)
    with_map_off = steer(ftg(USE_MAP=False), open_left, grid=grid)
    no_grid_at_all = steer(ftg(USE_MAP=False), open_left, grid=None)
    assert with_map_off == no_grid_at_all
    assert grid.calls == 0


# ------------------------------------------------- the preference, not the rule
def test_straight_ahead_wins_a_tie():
    clear = scan(lambda a: 5.0)
    assert abs(steer(ftg(), clear, Grid(lambda x, y: True))) < math.radians(3)


def test_forward_bias_can_be_turned_off():
    """With no bias the roomiest heading wins wherever it points."""
    a_bit_more_left = scan(
        lambda a: 5.0 if a > math.radians(20) else 1.0)
    assert steer(ftg(FORWARD_BIAS=0.0), a_bit_more_left,
                 Grid(lambda x, y: True)) > math.radians(20)


def test_the_probe_never_reaches_past_the_beam():
    """A heading is judged on the map only as far as the scan vouches for.

    Beyond the return the map would be answering about the far side of
    whatever is standing there.
    """
    near_wall = scan(lambda a: 0.5)
    seen = []
    grid = Grid(lambda x, y: seen.append(math.hypot(x, y)) or True)
    steer(ftg(), near_wall, grid=grid)
    assert seen and max(seen) <= 0.5 + 1e-9
