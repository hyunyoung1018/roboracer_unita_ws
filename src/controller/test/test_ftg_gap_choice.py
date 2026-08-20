"""A gap is metres, not beams.

_largest_run compares free runs by their length in BEAMS, and the room a gap
offers is a chord: 2 r sin(theta/2). The same angle is a different number of
metres at every range - a 20 deg run is 0.17 m at 0.5 m and 0.69 m at 2 m - so
comparing angles is biased towards the NEAREST gap, which is the one least
likely to fit a 0.28 m car through.

Two more things this file pins, both measured off the old code:

  the bubble    0.3 m of it is 37 deg at 0.4 m, and it is blown INTO the
                opening beside the obstacle. Two edges of one obstacle spent
                92% of an 80 deg FOV, and close quarters is where FTG runs.
  the fallback  was argmax on the RAW scan, so the branch taken when the
                search had failed was also the one with no mask and no
                bubbles in it.
"""

import math

import numpy as np

from controller.ftg.ftg import FTG


BEAMS = 361
ANGLE_MIN = -math.pi / 2
ANGLE_INC = math.pi / (BEAMS - 1)
# What process_lidar caps the bubble against: the FOV window it actually
# works on (+-45 deg at 0.5 deg), not the whole 180 deg scan.
FOV_BEAMS = 181


def ftg(**kwargs):
    f = FTG.__new__(FTG)
    f.node = None
    f.DEBUG = False
    f.mapping = False
    f.MAX_LIDAR_DIST = 10.0
    f.MAX_SPEED = 2.5
    f.track_width = 0.8
    f.FRONT_FOV = math.radians(45.0)
    f.SMOOTH_RAD = 0.0
    f.DISP_THRESH = 0.5
    f.BUBBLE_M = 0.3
    f.BUBBLE_MAX_FRAC = 0.25
    f.MIN_GAP_M = 0.34
    f.STEER_EMA = 0.0
    f.MAX_STEER = 10.0
    f.SPEED_SCALE = 1.0
    f.USE_MAP = False
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
    return [fn(ANGLE_MIN + i * ANGLE_INC) for i in range(BEAMS)]


def steer_deg(f, ranges):
    _, angle = f.process_lidar(ranges, ANGLE_MIN, ANGLE_INC)
    return math.degrees(angle)


# --------------------------------------------------------- a gap is metres
def test_a_wide_angle_gap_too_narrow_to_drive_is_not_chosen():
    """A near slot against a far opening, and the near one subtends more.

    Right: a 24 deg slot at 0.45 m, which is 0.19 m across - the car does not
    fit. Left: 14 deg at 3.0 m, which is 0.73 m and does. By beam count the
    right wins; by metres it is not a gap at all.
    """
    def ranges(a):
        if math.radians(-32) < a < math.radians(-8):
            return 0.45
        if math.radians(20) < a < math.radians(34):
            return 3.0
        return 0.2

    # DISP_THRESH high: this is about which gap is chosen, and a 0.2 -> 3.0
    # step would otherwise bubble the far gap away before the choice is
    # reached. The bubble has its own tests below.
    assert steer_deg(ftg(DISP_THRESH=99.0), scan(ranges)) > 15.0


def test_the_same_scan_goes_the_wrong_way_on_beam_count():
    """The old rule, kept as the thing being fixed rather than described."""
    def ranges(a):
        if math.radians(-32) < a < math.radians(-8):
            return 0.45
        if math.radians(20) < a < math.radians(34):
            return 3.0
        return 0.2

    f = ftg()
    proc = np.asarray(scan(ranges))
    free = proc >= f.track_width / 2.0
    gl, gr = f._largest_run(free)
    widest_by_beams = math.degrees(ANGLE_MIN + ((gl + gr) // 2) * ANGLE_INC)
    assert widest_by_beams < 0.0


def test_a_gap_is_only_as_wide_as_its_tightest_point():
    proc = np.full(BEAMS, 3.0)
    proc[100:110] = 0.4            # a pinch at the mouth of a deep opening
    narrow = FTG._gap_width_m(ftg(), proc, 100, 140)
    wide = FTG._gap_width_m(ftg(), proc, 111, 140)
    assert narrow < wide


def test_min_gap_is_tunable():
    def ranges(a):
        if math.radians(-32) < a < math.radians(-8):
            return 0.45
        if math.radians(20) < a < math.radians(34):
            return 3.0
        return 0.2

    # With nothing required to fit, the roomiest-in-metres still wins - the
    # rule is the chord, and MIN_GAP_M only says which gaps are candidates.
    assert steer_deg(ftg(MIN_GAP_M=0.0, DISP_THRESH=99.0), scan(ranges)) > 15.0


# --------------------------------------------------------------- the bubble
def test_a_close_obstacle_cannot_bubble_away_the_whole_view():
    f = ftg()
    uncapped = FTG._bubble_beams(f, 0.4)
    capped = FTG._bubble_beams(f, 0.4, FOV_BEAMS)
    assert capped < uncapped
    assert capped <= FOV_BEAMS * f.BUBBLE_MAX_FRAC


def test_the_cap_does_not_touch_a_normal_bubble():
    f = ftg()
    assert FTG._bubble_beams(f, 2.0, FOV_BEAMS) == FTG._bubble_beams(f, 2.0)


# ------------------------------------------------------------- the fallback
def test_nothing_fits_still_takes_the_roomiest_gap_not_the_raw_argmax():
    """Every gap is too narrow. The answer is still a gap, not a bearing.

    argmax on the raw scan would aim at the single farthest return, which sits
    behind the bubble blown at the obstacle edge beside it.
    """
    def ranges(a):
        if math.radians(-26) < a < math.radians(-18):
            return 0.6            # 0.08 m: does not fit
        if math.radians(10) < a < math.radians(26):
            return 0.9            # 0.25 m: does not fit either, but wider
        return 0.2

    assert steer_deg(ftg(DISP_THRESH=99.0), scan(ranges)) > 0.0


def test_the_raw_argmax_is_only_for_a_mask_that_left_nothing():
    f = ftg()
    proc = np.full(BEAMS, 0.1)
    proc[300] = 5.0
    free = np.zeros(BEAMS, dtype=bool)
    index, reason = FTG._choose_gap(f, free, proc)
    assert index == 300
    assert reason == 'no free bearings'


# ---------------------------------------------------- the speed schedule
def test_every_speed_step_is_reachable_at_this_car_s_lock():
    """MAX_STEER 0.4 rad is 22.9 deg, under the old 30 deg corner threshold.

    So CORNERS_SPEED could not be commanded at all and FTG's floor was 45% of
    ftg_max_speed at full lock. The thresholds are fractions of full lock now.
    """
    f = ftg(MAX_STEER=0.4)
    speeds = {f._speed_for(s * 0.4) for s in (0.0, 0.2, 0.5, 1.0)}
    assert len(speeds) == 4
    assert min(speeds) == f.CORNERS_SPEED


def test_full_lock_is_the_slowest_whatever_max_steer_is():
    for lock in (0.3, 0.4, 0.6):
        f = ftg(MAX_STEER=lock)
        assert f._speed_for(lock) == f.CORNERS_SPEED
        assert f._speed_for(0.0) == f.ULTRASTRAIGHTS_SPEED
