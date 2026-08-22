"""Selecting one opponent, and keeping it.

These gates moved out of the deleted stable_obstacle_router unchanged. The
second classifier that node also carried is gone - static/dynamic is the
tracker's speed verdict now - but selection was never the part that failed on
the car, and every case here is one it was written to survive.
"""

from types import SimpleNamespace

import pytest

from perception.h2h_opponent_selector import (
    DYNAMIC,
    STATIC,
    UNKNOWN,
    OpponentSelector,
    inside_forward_window,
    inside_opponent_corridor,
    unique_reidentification_candidate,
)

TRACK = 20.0
PARAMS = {
    'single_dynamic_opponent': True,
    'logical_opponent_id': 1000000,
    'opponent_width_m': 0.24,
    'opponent_boundary_margin_m': 0.03,
    'opponent_forward_min_m': 0.2,
    'opponent_forward_max_m': 8.0,
    'opponent_active_rear_m': -1.5,
    'dynamic_reid_timeout_sec': 1.0,
    'dynamic_reid_max_distance_m': 1.2,
    'dynamic_reid_max_lateral_m': 0.25,
    'dynamic_reid_ambiguity_margin_m': 0.20,
    'dynamic_speed_hold_sec': 0.75,
    'dynamic_speed_valid_min_mps': 0.15,
    'dynamic_speed_valid_max_mps': 6.0,
    'classification_debug': True,
    'opponent_acquire_speed_mps': 0.8,
    'opponent_acquire_frames': 10,
    'opponent_acquire_use_displacement': True,
    'opponent_acquire_window_sec': 1.0,
    'opponent_acquire_displacement_m': 0.25,
    'opponent_acquire_common_min_tracks': 3,
}

WAYPOINTS = [
    SimpleNamespace(s_m=float(i) * 0.5, d_left=0.9, d_right=0.9)
    for i in range(41)
]


def obstacle(obstacle_id, s, d=0.0, vs=1.0, visible=True):
    return SimpleNamespace(
        id=obstacle_id, s_center=s, d_center=d, vs=vs, is_visible=visible,
        is_static=False)


def selector(**overrides):
    params = dict(PARAMS, **overrides)
    return OpponentSelector(params.get)


def select(sel, obstacles, classes, now=0.0, ego_s=0.0, rolling=True):
    """One tick. `rolling` pre-fills what acquisition needs to see motion.

    Both mechanisms, so a test about the gates themselves reads the same under
    either setting of opponent_acquire_use_displacement: the streak the speed
    form counts, and a window of positions the displacement form measures.
    Tests that are ABOUT acquisition pass rolling=False and build their own.
    """
    if rolling:
        for obstacle in obstacles:
            sel.motion_streaks[int(obstacle.id)] = PARAMS[
                'opponent_acquire_frames']
            sel.motion_history[int(obstacle.id)] = rolling_history(
                obstacle.s_center, now)
    return sel.select(obstacles, classes, now, WAYPOINTS, TRACK, ego_s)


def rolling_history(s_end, now, speed=1.0, window=1.0, rate=20.0):
    """A track that has genuinely been moving, ending where it is now."""
    count = int(window * rate) + 1
    return [
        (now - window + i / rate, (s_end - speed * (window - i / rate)) % TRACK)
        for i in range(count)
    ]


# ------------------------------------------------------------------ gates
def test_corridor_refuses_a_footprint_that_does_not_fit():
    # Track half width 0.9; 0.24 wide opponent plus 0.03 needs |d| <= 0.75.
    assert inside_opponent_corridor(obstacle(1, 5.0, d=0.70), WAYPOINTS, TRACK, 0.24, 0.03)
    assert not inside_opponent_corridor(obstacle(1, 5.0, d=0.80), WAYPOINTS, TRACK, 0.24, 0.03)


def test_forward_window_reads_behind_as_negative_not_a_lap_ahead():
    # 0.4 m behind on a 20 m lap must not look like 19.6 m ahead.
    assert not inside_forward_window(obstacle(1, 19.6), 0.0, TRACK, 0.2, 8.0)
    assert inside_forward_window(obstacle(1, 19.6), 0.0, TRACK, -1.5, 8.0)


def test_reidentification_refuses_an_ambiguous_pair():
    assert unique_reidentification_candidate([(0.30, 7)], 1.2, 0.20) == (0.30, 7)
    # Two candidates 0.05 apart: guessing here produced an ID ping-pong.
    assert unique_reidentification_candidate([(0.30, 7), (0.35, 8)], 1.2, 0.20) is None
    # Far enough apart to be unambiguous again.
    assert unique_reidentification_candidate([(0.30, 7), (0.90, 8)], 1.2, 0.20)[1] == 7


# -------------------------------------------------------------- selection
def test_only_a_confirmed_dynamic_track_is_acquired():
    sel = selector()
    assert select(sel, [obstacle(1, 3.0)], {1: UNKNOWN}) is None
    assert select(sel, [obstacle(1, 3.0)], {1: STATIC}) is None
    assert select(sel, [obstacle(1, 3.0)], {1: DYNAMIC}).id == 1


def test_a_box_behind_the_car_never_becomes_the_opponent():
    sel = selector()
    assert select(sel, [obstacle(1, 19.0)], {1: DYNAMIC}) is None


def test_the_lock_survives_the_car_drawing_level():
    sel = selector()
    assert select(sel, [obstacle(1, 3.0)], {1: DYNAMIC}).id == 1
    # Alongside: forward gap now negative, inside opponent_active_rear_m.
    sel.stabilize_speed(obstacle(1, 19.4), 0.1)
    assert select(sel, [obstacle(1, 19.4)], {1: DYNAMIC}, now=0.2).id == 1


def test_the_lock_is_released_beyond_the_rear_allowance():
    sel = selector()
    select(sel, [obstacle(1, 3.0)], {1: DYNAMIC})
    sel.stabilize_speed(obstacle(1, 3.0), 0.0)
    # 2.0 m behind is past the -1.5 m allowance, and the grace period lapses.
    assert select(sel, [obstacle(1, 18.0)], {1: DYNAMIC}, now=2.0) is None
    assert sel.active_id is None


def test_a_second_moving_object_does_not_steal_the_lock():
    sel = selector()
    assert select(sel, [obstacle(1, 4.0)], {1: DYNAMIC}).id == 1
    both = [obstacle(1, 4.0), obstacle(2, 1.0)]
    assert select(sel, both, {1: DYNAMIC, 2: DYNAMIC}).id == 1


def test_acquisition_prefers_the_target_on_the_line():
    sel = selector()
    # The nearer one is off the line; the box-and-operator test.
    candidates = [obstacle(1, 2.0, d=0.55), obstacle(2, 4.0, d=0.02)]
    assert select(sel, candidates, {1: DYNAMIC, 2: DYNAMIC}).id == 2


def test_id_churn_hands_the_lock_over_to_an_unknown_track():
    sel = selector()
    assert select(sel, [obstacle(1, 4.0)], {1: DYNAMIC}).id == 1
    sel.stabilize_speed(obstacle(1, 4.0, vs=1.0), 0.0)
    # Old ID gone, a fresh track appears where it was: still UNKNOWN, because
    # it has not collected static_min_samples frames yet.
    assert select(sel, [obstacle(9, 4.2)], {9: UNKNOWN}, now=0.1) is None
    assert sel.active_id == 9


def test_a_confirmed_static_track_is_never_the_replacement():
    sel = selector()
    select(sel, [obstacle(1, 4.0)], {1: DYNAMIC})
    sel.stabilize_speed(obstacle(1, 4.0), 0.0)
    select(sel, [obstacle(9, 4.2)], {9: STATIC}, now=0.1)
    assert sel.active_id == 1


# ------------------------------------------------------------------ speed
def test_a_credible_speed_is_held_through_an_invisible_frame():
    sel = selector()
    sel.stabilize_speed(obstacle(1, 4.0, vs=2.0), 0.0)
    blind = obstacle(1, 4.2, vs=0.0, visible=False)
    sel.stabilize_speed(blind, 0.3)
    assert blind.vs == 2.0


def test_the_hold_expires():
    sel = selector()
    sel.stabilize_speed(obstacle(1, 4.0, vs=2.0), 0.0)
    blind = obstacle(1, 4.2, vs=0.0, visible=False)
    sel.stabilize_speed(blind, 2.0)
    assert blind.vs == 0.0


def test_an_impossible_speed_is_not_believed():
    # The 8.55 m/s frame for a stationary box: accepted before, and the
    # trailing controller then commanded acceleration towards it.
    sel = selector()
    sel.stabilize_speed(obstacle(1, 4.0, vs=2.0), 0.0)
    bad = obstacle(1, 4.2, vs=8.55)
    sel.stabilize_speed(bad, 0.1)
    assert sel.last_speed == 2.0


# ------------------------------------------------- acquisition needs motion
def test_a_box_reading_dynamic_is_not_acquired():
    """The 2026-08-19 run: eleven acquisitions with no opponent on the track.

    A box reads DYNAMIC because its apparent speed brushes the threshold. That
    costs two failures at once - it leaves /tracking/static_obstacles so the
    spline planner cannot plan around it, and h2h_spline_node folds it in as a
    wall and narrows the corridor to nothing.
    """
    sel = selector()
    box = obstacle(1, 3.0, vs=0.3)      # over the 0.15 class threshold, under 0.8
    for _ in range(20):
        assert select(sel, [box], {1: DYNAMIC}, rolling=False) is None


# --- the speed/streak form, kept for rollback -----------------------------
#
# opponent_acquire_use_displacement False restores it exactly, live. These
# three feed a track that reports vs while its POSITION never changes - which
# is the box case, and which is why the displacement form refuses it. They
# pin the old path, not the shipped one.
def test_a_rolling_car_is_acquired_once_it_has_rolled():
    sel = selector(opponent_acquire_use_displacement=False)
    car = obstacle(1, 3.0, vs=2.0)
    for _ in range(9):
        assert select(sel, [car], {1: DYNAMIC}, rolling=False) is None
    assert select(sel, [car], {1: DYNAMIC}, rolling=False).id == 1


def test_the_streak_resets_on_a_slow_frame():
    sel = selector(opponent_acquire_use_displacement=False)
    car = obstacle(1, 3.0, vs=2.0)
    slow = obstacle(1, 3.0, vs=0.1)
    for _ in range(9):
        select(sel, [car], {1: DYNAMIC}, rolling=False)
    select(sel, [slow], {1: DYNAMIC}, rolling=False)
    assert select(sel, [car], {1: DYNAMIC}, rolling=False) is None


def test_an_invisible_frame_neither_counts_nor_resets():
    sel = selector(opponent_acquire_use_displacement=False)
    car = obstacle(1, 3.0, vs=2.0)
    for _ in range(9):
        select(sel, [car], {1: DYNAMIC}, rolling=False)
    select(sel, [obstacle(1, 3.0, vs=0.0, visible=False)], {1: DYNAMIC},
           rolling=False)
    assert select(sel, [car], {1: DYNAMIC}, rolling=False).id == 1


def test_a_locked_opponent_that_stops_stays_the_opponent():
    """Acquisition only. An opponent stopping behind a box is still the
    opponent - the retained branch returns before the motion gate is asked."""
    sel = selector()
    assert select(sel, [obstacle(1, 3.0, vs=2.0)], {1: DYNAMIC}).id == 1
    stopped = obstacle(1, 3.0, vs=0.0)
    assert select(sel, [stopped], {1: DYNAMIC}, rolling=False).id == 1


def test_the_streak_is_forgotten_when_the_track_goes():
    sel = selector()
    for _ in range(10):
        select(sel, [obstacle(1, 3.0, vs=2.0)], {1: DYNAMIC}, rolling=False)
    select(sel, [], {}, rolling=False)
    assert 1 not in sel.motion_streaks


# --- the displacement form, which is what ships ---------------------------
#
# The speed form had to choose. A stationary box's apparent speed was measured
# at 0.5 to 2.9 m/s (20 Hz differentiation turns a 2.5 cm wobble into 0.5) and
# a real opponent runs at 0 to 4, so the ranges overlap completely: a floor
# high enough to reject the box also rejected any opponent slower than it, and
# the band between static_speed_threshold and that floor was owned by nothing.
# The car trailed a crawling opponent forever.
#
# Displacement does not make that trade. Noise is zero-mean and does not
# accumulate over a window; motion does. These two tests are the trade, and
# they pass together - which is the whole point of the change.

def feed(sel, obstacle_id, positions, start=0.0, rate=20.0, d=0.0,
         vs=0.0, visible=True, ego_s=0.0):
    """Drive `select` one real frame at a time, at 20 Hz, from `positions`."""
    result = None
    for index, s in enumerate(positions):
        now = start + index / rate
        obs = SimpleNamespace(id=obstacle_id, s_center=s % TRACK, d_center=d,
                              vs=vs, is_visible=visible, is_static=False)
        result = sel.select([obs], {obstacle_id: DYNAMIC}, now, WAYPOINTS,
                            TRACK, ego_s)
    return result


def test_a_box_wobbling_at_two_metres_a_second_is_not_acquired():
    """The measured failure: 2.9 m/s of apparent speed, going nowhere.

    Zero-mean wobble of 7 cm either way - larger than anything measured on the
    car - sustained for two full windows.
    """
    sel = selector()
    wobble = [3.0 + (0.07 if i % 2 else -0.07) for i in range(40)]
    assert feed(sel, 1, wobble, vs=2.9) is None


def test_a_single_frame_outlier_is_removed_outright():
    """Whatever its size. This is what the robust endpoints buy."""
    sel = selector()
    positions = [3.0] * 10 + [5.0] + [3.0] * 20
    assert feed(sel, 1, positions, vs=2.9) is None
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.0, abs=1e-9)


def test_the_worst_measured_localisation_step_stays_under_the_threshold():
    """A persisting step IS measured at full size - the threshold refuses it.

    14.5 cm in one frame is the worst jump measured on this car, and it sits
    at 0.145 m against a 0.25 m threshold. Pinned because a threshold that
    drifted under it would let localisation acquire an opponent.
    """
    sel = selector()
    step = [3.0] * 10 + [3.145] * 11
    sel.motion_history[1] = [(i / 20.0, s) for i, s in enumerate(step)]
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.145, abs=1e-6)
    assert 0.145 < PARAMS['opponent_acquire_displacement_m']


def test_an_opponent_crawling_at_a_third_of_a_metre_a_second_is_acquired():
    """The case the speed floor could never reach.

    0.3 m/s is under opponent_acquire_speed_mps at 0.8 AND at 0.5, so the
    streak form never acquired it however long it ran - and an unacquired
    opponent is excluded from every opponent-aware check downstream.
    """
    sel = selector()
    crawl = [3.0 + 0.3 * (i / 20.0) for i in range(40)]
    acquired = feed(sel, 1, crawl, vs=0.3)
    assert acquired is not None and acquired.id == 1


def test_the_window_has_to_be_full_before_anything_is_acquired():
    """Three frames of fast motion is not a second of observation."""
    sel = selector()
    assert feed(sel, 1, [3.0, 3.2, 3.4], vs=4.0) is None


def test_a_track_seen_twice_in_a_second_does_not_qualify():
    """Span, not sample count: two samples a second apart span a second."""
    sel = selector()
    obs = SimpleNamespace(id=1, s_center=3.0, d_center=0.0, vs=2.0,
                          is_visible=True, is_static=False)
    sel.select([obs], {1: DYNAMIC}, 0.0, WAYPOINTS, TRACK, 0.0)
    moved = SimpleNamespace(id=1, s_center=4.0, d_center=0.0, vs=2.0,
                            is_visible=True, is_static=False)
    assert sel.select([moved], {1: DYNAMIC}, 1.0, WAYPOINTS, TRACK, 0.0) is None


def test_displacement_is_measured_across_the_start_line():
    """s wraps; the offsets the medians run on are anchored, so it cannot."""
    sel = selector()
    crossing = [(TRACK - 0.5 + 0.5 * (i / 20.0)) for i in range(40)]
    acquired = feed(sel, 1, crossing, vs=0.5, ego_s=TRACK - 2.0)
    assert acquired is not None and acquired.id == 1


def test_an_invisible_frame_contributes_no_displacement():
    sel = selector()
    obs = SimpleNamespace(id=1, s_center=3.0, d_center=0.0, vs=0.0,
                          is_visible=False, is_static=False)
    for index in range(40):
        sel.select([obs], {1: DYNAMIC}, index / 20.0, WAYPOINTS, TRACK, 0.0)
    assert sel._acquire_displacement(1, TRACK) is None


def test_the_history_is_forgotten_when_the_track_goes():
    sel = selector()
    feed(sel, 1, [3.0 + 0.5 * (i / 20.0) for i in range(30)])
    sel.select([], {}, 2.0, WAYPOINTS, TRACK, 0.0)
    assert 1 not in sel.motion_history


# --- a pose correction is not motion ---------------------------------------
#
# Measured 2026-08-22, car stopped and boxes stopped: one obstacle's reported
# gap wandered 7.74 to 8.06 m and its near edge +0.03 to +0.25, while its
# measured WIDTH stayed inside 6 cm. Translated, not grown - by up to 0.19 m in
# one second, which is half the acquisition threshold spent before anything
# moves. The point cloud is transformed to the map frame before clustering, so
# a pose shift moves every obstacle together.
#
# What separates it from motion is that it is COMMON. Subtract the median
# across tracks and a shift cancels while a car does not.

# Robust endpoints take the MEDIAN of the outer fifth at each end rather than
# the first and last sample, so a pure ramp reads at 0.85 of its true extent.
# That is deliberate - it is what removes a single bad frame - and it applies
# equally to both sides of the subtraction, so a differential is exact.
RAMP = 0.85


def history(sel, obstacle_id, start_s, moved, now=1.0, window=1.0, rate=20.0):
    count = int(window * rate) + 1
    sel.motion_history[obstacle_id] = [
        (now - window + i / rate,
         (start_s + moved * (i / (count - 1))) % TRACK)
        for i in range(count)]


def test_a_shift_that_moves_everything_moves_nobody():
    """Three boxes, all translated 0.6 m by a pose correction."""
    sel = selector()
    for i, s in enumerate((3.0, 6.0, 9.0)):
        history(sel, i + 1, s, moved=0.6)
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.0, abs=1e-6)


def test_the_one_that_moved_alone_still_reads_as_moving():
    sel = selector()
    history(sel, 1, 3.0, moved=0.6)      # the opponent
    history(sel, 2, 6.0, moved=0.0)      # boxes, holding still
    history(sel, 3, 9.0, moved=0.0)
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.6 * RAMP, abs=1e-6)


def test_a_shift_plus_real_motion_leaves_the_motion():
    """Everything drifts 0.2; the opponent also covers 0.6 of its own."""
    sel = selector()
    history(sel, 1, 3.0, moved=0.8)
    history(sel, 2, 6.0, moved=0.2)
    history(sel, 3, 9.0, moved=0.2)
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.6 * RAMP, abs=1e-6)


def test_too_few_tracks_leaves_the_raw_measurement():
    """With one track the median is that track and everything would cancel."""
    sel = selector()
    history(sel, 1, 3.0, moved=0.6)
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.6 * RAMP, abs=1e-6)


def test_two_tracks_is_still_too_few():
    sel = selector()
    history(sel, 1, 3.0, moved=0.6)
    history(sel, 2, 6.0, moved=0.6)
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.6 * RAMP, abs=1e-6)


def test_the_quorum_is_tunable_and_zero_disables_it():
    sel = selector(opponent_acquire_common_min_tracks=0)
    for i, s in enumerate((3.0, 6.0, 9.0)):
        history(sel, i + 1, s, moved=0.6)
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.6 * RAMP, abs=1e-6)


def test_the_measured_stationary_wander_no_longer_acquires():
    """0.19 m of scene-wide translation, the worst measured in one second."""
    sel = selector()
    for i, s in enumerate((3.0, 6.0, 9.0, 12.0)):
        history(sel, i + 1, s, moved=0.19)
    moved = sel._acquire_displacement(1, TRACK)
    assert moved < PARAMS['opponent_acquire_displacement_m']


# --- the pose correction, measured against the car -------------------------
#
# The scene-median form needs three tracks and this course runs one: the
# acquisition log said "scene drift removed: none - only 1 track(s), needs 3"
# on the run that produced it. /car_state/odom is the fused pose in the map
# frame and /vesc/odom is dead reckoning in the odom frame; over a window the
# difference between how far each says the car went is what localisation put
# in, which is the same shift every obstacle picked up.

def ego(sel, moved_map, moved_odom, now=1.0, window=1.0, rate=20.0):
    count = int(window * rate) + 1
    for i in range(count):
        f = i / (count - 1)
        sel.note_ego(now - window + i / rate, moved_map * f, moved_odom * f)


def test_a_correction_that_moved_the_world_is_not_the_track_moving():
    sel = selector()
    history(sel, 1, 3.0, moved=0.5)
    ego(sel, moved_map=3.5, moved_odom=3.0)      # 0.5 m of correction
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.0, abs=1e-6)


def test_motion_beyond_the_correction_survives():
    sel = selector()
    history(sel, 1, 3.0, moved=1.0)
    ego(sel, moved_map=3.2, moved_odom=3.0)      # 0.2 m of correction
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(
        1.0 * RAMP - 0.2, abs=1e-6)


def test_agreeing_odometry_takes_nothing_off():
    sel = selector()
    history(sel, 1, 3.0, moved=0.6)
    ego(sel, moved_map=3.0, moved_odom=3.0)      # no correction at all
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.6 * RAMP, abs=1e-6)


def test_it_works_with_one_track_which_is_the_whole_point():
    sel = selector()
    history(sel, 1, 3.0, moved=0.5)
    ego(sel, moved_map=3.5, moved_odom=3.0)
    assert len(sel.motion_history) == 1
    assert sel._common_drift(TRACK, exclude_id=1) is None    # quorum short
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.0, abs=1e-6)


def test_no_ego_odometry_leaves_the_measurement_alone():
    sel = selector()
    history(sel, 1, 3.0, moved=0.6)
    assert sel._pose_correction() is None
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.6 * RAMP, abs=1e-6)


def test_a_partial_window_of_ego_samples_is_not_used():
    sel = selector()
    history(sel, 1, 3.0, moved=0.6)
    ego(sel, moved_map=3.5, moved_odom=3.0, window=0.3)
    assert sel._pose_correction() is None


def test_the_correction_never_drives_the_measurement_negative():
    sel = selector()
    history(sel, 1, 3.0, moved=0.1)
    ego(sel, moved_map=4.0, moved_odom=3.0)      # 1.0 m, larger than the move
    assert sel._acquire_displacement(1, TRACK) == pytest.approx(0.0, abs=1e-9)
