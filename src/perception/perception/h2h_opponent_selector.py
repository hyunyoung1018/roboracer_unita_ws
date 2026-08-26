"""Pick and hold the one head-to-head opponent.

The shared tracker deliberately reports every object it can see. Head-to-head
prediction has a stricter contract: exactly one physical opponent, inside the
drivable corridor and ahead of the ego car, carried across the tracker losing
and recreating its ID.

This used to live in stable_obstacle_router, next to a second obstacle
classifier that re-derived static/dynamic from the standard deviation of the
Frenet position. That classifier was wrong on the car and is gone; SELECTION is
a different question from CLASSIFICATION and it was never the part at fault, so
it moves here intact and now runs on the tracker's speed-based verdict instead.

Deliberately ROS-free. Every gate below is a safety gate, and keeping them out
of a node is what makes them straightforwardly unit-testable.
"""

import math

STATIC = "STATIC"
DYNAMIC = "DYNAMIC"
UNKNOWN = "UNKNOWN"


def circular_forward_delta(value, reference, track_length):
    return (float(value) - float(reference)) % float(track_length)


def _median(values):
    """Median without numpy - this module stays dependency-light on purpose."""
    ordered = sorted(values)
    count = len(ordered)
    if not count:
        return 0.0
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def circular_delta(a, b, track_length):
    """Signed shortest distance from ``b`` to ``a`` around a closed track."""
    if not track_length:
        return float(a) - float(b)
    half = 0.5 * float(track_length)
    return (float(a) - float(b) + half) % float(track_length) - half


def nearest_waypoint(s, waypoints, track_length):
    """Return the waypoint closest to ``s`` on a closed track."""
    if not waypoints or not track_length:
        return None
    target = float(s) % float(track_length)
    return min(
        waypoints,
        key=lambda waypoint: abs(
            (float(waypoint.s_m) - target + 0.5 * track_length)
            % track_length - 0.5 * track_length
        ),
    )


def inside_opponent_corridor(
        obstacle, waypoints, track_length, opponent_width, margin):
    """Check that the expected opponent footprint fits inside track bounds."""
    waypoint = nearest_waypoint(obstacle.s_center, waypoints, track_length)
    if waypoint is None:
        return False
    half_width = 0.5 * max(0.0, float(opponent_width))
    clearance = half_width + max(0.0, float(margin))
    right_limit = -float(waypoint.d_right) + clearance
    left_limit = float(waypoint.d_left) - clearance
    lateral = float(obstacle.d_center)
    return (
        math.isfinite(lateral)
        and right_limit <= lateral <= left_limit
    )


def inside_forward_window(
        obstacle, ego_s, track_length, minimum_distance, maximum_distance):
    """Check the opponent is in the forward head-to-head observation window.

    ``minimum_distance`` may be negative, which extends the window behind the
    ego car. A signed delta is used so that "0.4 m behind" is -0.4 rather than
    ``track_length - 0.4``; on a 21 m lap the unsigned form makes anything
    beside the car look like it is most of a lap ahead.
    """
    if ego_s is None or not track_length:
        return False
    gap = circular_forward_delta(obstacle.s_center, ego_s, track_length)
    if gap > 0.5 * float(track_length):
        gap -= float(track_length)
    return float(minimum_distance) <= gap <= float(maximum_distance)


def initial_candidate_key(obstacle, ego_s, track_length):
    """Prefer the raceline-like target, then the nearer forward target.

    In the box-and-operator test the box is placed on the driving line while the
    operator walks laterally beside it. Once acquired, the target is locked and
    this key is no longer consulted.
    """
    return (
        abs(float(obstacle.d_center)),
        circular_forward_delta(obstacle.s_center, ego_s, track_length),
        int(obstacle.id),
    )


def unique_reidentification_candidate(candidates, max_distance, ambiguity_margin):
    """Return one unambiguous ``(distance, id)`` candidate, otherwise ``None``."""
    candidates = sorted(candidates)
    if not candidates or candidates[0][0] > float(max_distance):
        return None
    if (
        len(candidates) > 1
        and candidates[1][0] <= float(max_distance)
        and candidates[1][0] - candidates[0][0] < float(ambiguity_margin)
    ):
        return None
    return candidates[0]


class OpponentSelector:
    """Lock one opponent, keep it through ID churn, and hold its speed.

    ``get_param(name)`` reads live so every threshold stays a runtime knob, and
    ``log(message)`` is optional so tests can run this with no ROS at all.
    """

    def __init__(self, get_param, log=None):
        self._param = get_param
        self._log = log if log is not None else (lambda message: None)
        self.active_id = None
        self.retired_ids = {}
        # Consecutive frames each track has been seen genuinely rolling. Only
        # consulted for ACQUISITION - see _moving_enough_to_acquire.
        self.motion_streaks = {}
        # {track id: [(time, s_center), ...]} over the acquisition window, for
        # the displacement test that replaced the streak. See
        # _acquire_displacement for why displacement and not speed.
        self.motion_history = {}
        self.last_speed = None
        self.last_speed_at = None
        self.last_s = None
        self.last_d = None
        self.last_position_at = None
        # Per-tick gate results, for the classification debug snapshot.
        self.gates = {}

    # ---------------------------------------------------------------- gates
    def _corridor_ok(self, obstacle, waypoints, track_length):
        return inside_opponent_corridor(
            obstacle, waypoints, track_length,
            self._param("opponent_width_m"),
            self._param("opponent_boundary_margin_m"))

    def _forward_ok(self, obstacle, ego_s, track_length):
        """Acquisition window, widened rearwards for the locked target.

        Keeping the opponent through the moment the car draws level with it is
        a different question from picking one out in the first place, so the
        two get different minimums. Without the rear allowance the opponent's
        forward gap falls through opponent_forward_min_m mid-overtake, it drops
        out of the candidate list, /tracking/dynamic_obstacles empties,
        opp_prediction reports NO_DYNAMIC_OBSTACLE and raises force_trailing,
        and the lane-change planner withdraws its path - at the exact moment
        the car is alongside and most needs it.
        """
        minimum = float(self._param("opponent_forward_min_m"))
        if self.active_id is not None and int(obstacle.id) == self.active_id:
            minimum = min(minimum, float(self._param("opponent_active_rear_m")))
        return inside_forward_window(
            obstacle, ego_s, track_length,
            minimum, self._param("opponent_forward_max_m"))

    # ------------------------------------------------------------ re-id
    def _projected_position(self, now, obstacles_by_id, track_length):
        active = obstacles_by_id.get(self.active_id)
        if active is not None:
            return float(active.s_center), float(active.d_center)
        if self.last_s is None or self.last_d is None:
            return None
        elapsed = max(0.0, now - self.last_position_at)
        projected_s = self.last_s + (self.last_speed or 0.0) * elapsed
        if track_length:
            projected_s %= track_length
        return projected_s, self.last_d

    def _maybe_handoff(self, obstacles, classes, now, waypoints, track_length,
                       ego_s):
        """Transfer the lock when the tracker reacquires the opponent as a new ID."""
        if not self._param("single_dynamic_opponent") or self.active_id is None:
            return None
        obstacles_by_id = {int(obs.id): obs for obs in obstacles}
        active = obstacles_by_id.get(self.active_id)
        if active is not None and active.is_visible:
            return None
        reference = self._projected_position(now, obstacles_by_id, track_length)
        if reference is None or self.last_position_at is None:
            return None
        timeout = float(self._param("dynamic_reid_timeout_sec"))
        if now - self.last_position_at > timeout:
            return None

        candidates = []
        for obstacle in obstacles:
            obstacle_id = int(obstacle.id)
            if obstacle_id == self.active_id or not obstacle.is_visible:
                continue
            # An ID the lock was just handed AWAY from may not take it straight
            # back. Without this the car logged an ID ping-pong - 7->8->7->8,
            # 39 handoffs across 67 tracker IDs in 75 s - and every flip wrote
            # the other cluster's s into last_s, so the next sample went
            # backwards and opponent_trajectory rejected it. The predictor sat
            # in TRAINING with force_trailing raised for the whole run.
            if obstacle_id in self.retired_ids:
                continue
            # A track that is confidently STATIC is a box; one that is already
            # confidently DYNAMIC is a second moving object (the operator), not
            # a replacement ID for the locked opponent. Only UNKNOWN - a track
            # too new to have earned either - can be the reacquisition.
            if classes.get(obstacle_id, UNKNOWN) != UNKNOWN:
                continue
            if not self._corridor_ok(obstacle, waypoints, track_length):
                continue
            if not self._forward_ok(obstacle, ego_s, track_length):
                continue
            ds = circular_delta(
                float(obstacle.s_center), reference[0], track_length)
            dd = float(obstacle.d_center) - reference[1]
            if abs(dd) > float(self._param("dynamic_reid_max_lateral_m")):
                continue
            candidates.append((math.hypot(ds, dd), obstacle_id))
        if not candidates:
            return None

        selected = unique_reidentification_candidate(
            candidates,
            self._param("dynamic_reid_max_distance_m"),
            self._param("dynamic_reid_ambiguity_margin_m"))
        if selected is None:
            return None
        distance, new_id = selected
        old_id = self.active_id
        self.active_id = new_id
        self.retired_ids[old_id] = now + timeout
        self.retired_ids.pop(new_id, None)
        self._log(
            f"dynamic opponent tracker ID changed {old_id} -> {new_id}; "
            f"preserving the lock and its speed (re-id distance {distance:.2f} m)")
        return new_id

    # ----------------------------------------------------------- selection
    def _expired(self, now):
        if self.active_id is None:
            return False
        timeout = float(self._param("dynamic_reid_timeout_sec"))
        return (
            self.last_position_at is None
            or now - self.last_position_at > timeout
        )

    def _update_motion_streaks(self, obstacles):
        """Count consecutive frames each visible track is genuinely rolling."""
        floor = float(self._param("opponent_acquire_speed_mps"))
        present = set()
        for obstacle in obstacles:
            obstacle_id = int(obstacle.id)
            present.add(obstacle_id)
            if not obstacle.is_visible:
                continue
            speed = float(obstacle.vs)
            if math.isfinite(speed) and abs(speed) >= floor:
                self.motion_streaks[obstacle_id] = (
                    self.motion_streaks.get(obstacle_id, 0) + 1)
            else:
                self.motion_streaks[obstacle_id] = 0
        for obstacle_id in list(self.motion_streaks):
            if obstacle_id not in present:
                del self.motion_streaks[obstacle_id]

    def _update_motion_history(self, obstacles, now, track_length):
        """Per-track position samples over the acquisition window."""
        window = float(self._param("opponent_acquire_window_sec"))
        present = set()
        for obstacle in obstacles:
            obstacle_id = int(obstacle.id)
            present.add(obstacle_id)
            # An invisible frame is a frame with no measurement in it. Holding
            # the last position and calling it a sample would let an occluded
            # track accumulate a span it never earned; skipping means it simply
            # does not qualify until it is seen again, which is the safe way
            # round.
            if not obstacle.is_visible:
                continue
            samples = self.motion_history.setdefault(obstacle_id, [])
            samples.append((float(now), float(obstacle.s_center)))
            cutoff = float(now) - window
            while len(samples) > 2 and samples[0][0] < cutoff:
                samples.pop(0)
        for obstacle_id in list(self.motion_history):
            if obstacle_id not in present:
                del self.motion_history[obstacle_id]

    def _common_drift(self, track_length, exclude_id=None):
        """How far EVERY track appears to have gone this window, or None.

        A pose correction moves the whole point cloud in the map frame, so
        every obstacle's s moves with it and none of them actually went
        anywhere. Measured on 2026-08-22 with the car stopped and the boxes
        stopped: one obstacle's reported gap wandered 7.74 to 8.06 m and its
        near edge +0.03 to +0.25, while its measured WIDTH stayed inside 6 cm.
        The box did not grow or turn - it was translated, repeatedly, by up to
        0.19 m in a single second. That is 48% of the acquisition threshold
        spent before anything moves.

        The one thing that separates it from motion is that it is COMMON. A
        pose shift moves all of them together; a car moves alone. So the median
        of the per-track displacements estimates the shift, and subtracting it
        leaves each track's own motion.

        Needs three tracks. With one, the median IS that track and every
        displacement cancels to zero - a real opponent would never be acquired.
        With two there is no majority to take a median of. Below that this
        returns None and the raw displacement stands, which is the behaviour
        this had before.
        """
        minimum = int(self._param("opponent_acquire_common_min_tracks"))
        if minimum <= 0:
            return None
        moves = []
        for obstacle_id in self.motion_history:
            if exclude_id is not None and int(obstacle_id) == int(exclude_id):
                continue
            signed = self._signed_displacement(obstacle_id, track_length)
            if signed is not None:
                moves.append(signed)
        # exclude_id was left out, so the caller's own track is not counted in
        # the quorum either.
        if len(moves) < minimum - 1:
            return None
        return _median(moves)

    def _signed_displacement(self, obstacle_id, track_length):
        """The window's displacement WITH ITS SIGN, or None. See _acquire_displacement."""
        samples = self.motion_history.get(int(obstacle_id))
        if not samples or len(samples) < 4:
            return None
        window = float(self._param("opponent_acquire_window_sec"))
        if samples[-1][0] - samples[0][0] < 0.8 * window:
            return None
        anchor = samples[-1][1]
        offsets = [circular_delta(s, anchor, track_length) for _, s in samples]
        edge = max(1, len(offsets) // 5)
        return _median(offsets[-edge:]) - _median(offsets[:edge])

    def _acquire_displacement(self, obstacle_id, track_length):
        """How far this track has actually GONE over the window, or None.

        None means the window is not full yet - too new, or seen too seldom -
        which is not the same as "has not moved" and must not read as one.

        WHY DISPLACEMENT AND NOT SPEED. Speed is the derivative of position,
        and at 20 Hz differentiating multiplies position noise by twenty: a
        stationary box wobbling 2.5 cm between frames reads as 0.5 m/s. That
        is not a threshold that was set too low, it is the wrong measurement -
        the apparent speed of a box (0.5 to 2.9 m/s, measured on the car) and
        the real speed of an opponent (0 to 4 m/s) occupy the same range, so
        NO threshold separates them.

        Displacement over a window does, because the two error terms behave
        differently in time. Detector noise is zero-mean, so it does not
        accumulate however long the window; real motion does. Over one second
        a box stays inside its own jitter, about 15 cm, while an opponent
        creeping at 0.3 m/s has gone 30 cm and one at 0.5 m/s has gone 50.
        Widen the window and the box column does not move while the opponent
        column does.

        That is also what fixes the slow opponent. The speed form had to
        choose between a floor high enough to reject a box and one low enough
        to accept a crawling car, and there was no value that did both -
        between static_speed_threshold (0.15) and the acquisition floor lay a
        band that the tracker called DYNAMIC and the selector would not take,
        so nothing owned it. Displacement does not make that trade: slow is
        compensated by looking longer.

        ROBUST ENDPOINTS, not the first and last sample. A single bad frame at
        either end of the window would otherwise be the whole measurement.
        Offsets are taken against the newest sample first, so the medians are
        of small signed numbers and the s=0 seam cannot reach them.

        WHAT THIS DOES NOT CATCH, measured on the bench: a single-frame
        outlier of any size reads as 0.000 m - the medians remove it outright -
        but a STEP that persists for the rest of the window is indistinguishable
        from motion, and is measured at its full size. The threshold is the
        only thing rejecting it. At 0.25 m over 1 s that covers every jump seen
        on this car (the worst measured is 0.145 m) with 1.7x to spare; a
        larger step would read as motion.

        The tuning axis for that is the window, not a consistency check.
        Lengthening the window scales what real motion covers while a step
        stays the same size, so window and threshold can rise together and buy
        step rejection: 2.0 s and 0.50 m still acquires a 0.3 m/s opponent
        (0.60 m) while refusing any step under half a metre. The cost is that
        acquisition takes that much longer. A consistency check across the two
        halves was considered and rejected - it would refuse an opponent
        accelerating from rest, which is a real racing case, to guard against a
        localisation jump that has never been measured.
        """
        samples = self.motion_history.get(int(obstacle_id))
        if not samples or len(samples) < 4:
            return None
        window = float(self._param("opponent_acquire_window_sec"))
        span = samples[-1][0] - samples[0][0]
        # A track seen twice in a second has not been observed for a second.
        if span < 0.8 * window:
            return None
        signed = self._signed_displacement(obstacle_id, track_length)
        if signed is None:
            return None
        # Everything the whole scene appears to have done is not this track's
        # doing. See _common_drift.
        common = self._common_drift(track_length, exclude_id=obstacle_id)
        if common is not None:
            signed -= common
        return abs(signed)

    def _moving_enough_to_acquire(self, obstacle, track_length=None):
        """Has this track actually been rolling, for long enough to believe?

        ACQUISITION ONLY. Being DYNAMIC is what the classifier says; being the
        opponent is a much stronger claim, and it should be, because the two
        mistakes do not cost the same. Acquiring a real opponent late costs a
        late trailing arm. Acquiring a BOX costs a crash - and not one crash
        but two at once, because the selected opponent is pulled out of
        /tracking/static_obstacles so the spline planner cannot plan around it,
        and h2h_spline_node then treats it as a wall and narrows the corridor
        to nothing. Measured on 2026-08-19 with no opponent on the track at
        all: eleven acquisitions, "left 1.30 -> 0.00 m, right 0.35 -> 0.00 m",
        and no avoidance path for a box that was sitting on the raceline.

        A box reads DYNAMIC because its APPARENT speed brushes the threshold
        for a frame or two - 0.5 to 2.9 m/s of it, measured, and mostly from
        the ego pose rather than from the box. What it does not do is roll at a
        credible speed for a run of frames, which is the one thing a real
        opponent does continuously. So that is what acquisition asks for.

        Nothing here holds a target that is already locked: an opponent that
        stops behind a box must stay the opponent, and _select's retained
        branch returns before this is ever consulted.
        """
        if bool(self._param("opponent_acquire_use_displacement")):
            moved = self._acquire_displacement(obstacle.id, track_length)
            if moved is None:
                return False
            return moved >= float(
                self._param("opponent_acquire_displacement_m"))
        # The speed form, kept whole so the change can be undone on the car
        # with one `ros2 param set` rather than a rebuild.
        frames = int(self._param("opponent_acquire_frames"))
        return self.motion_streaks.get(int(obstacle.id), 0) >= frames

    def select(self, obstacles, classes, now, waypoints, track_length, ego_s):
        """Return the one locked opponent obstacle, or ``None``.

        A retained active track wins even when something else is momentarily
        closer. If the previous target is still inside its re-identification
        grace period, publish no target rather than silently switching the
        predictor to a different moving object.
        """
        self.retired_ids = {
            obstacle_id: expires_at
            for obstacle_id, expires_at in self.retired_ids.items()
            if expires_at >= now
        }
        self._maybe_handoff(
            obstacles, classes, now, waypoints, track_length, ego_s)

        self._update_motion_streaks(obstacles)
        self._update_motion_history(obstacles, now, track_length)

        self.gates = {}
        candidates = []
        for obstacle in obstacles:
            corridor_ok = self._corridor_ok(obstacle, waypoints, track_length)
            forward_ok = self._forward_ok(obstacle, ego_s, track_length)
            self.gates[int(obstacle.id)] = (corridor_ok, forward_ok)
            if (classes.get(int(obstacle.id)) == DYNAMIC
                    and corridor_ok and forward_ok):
                candidates.append(obstacle)

        by_id = {int(obs.id): obs for obs in candidates}
        selected = by_id.get(self.active_id)
        if selected is not None:
            return selected

        if self._expired(now):
            old_id = self.active_id
            self.active_id = None
            self._log(
                f"dynamic opponent tracker {old_id} expired; "
                "waiting to acquire one replacement target")

        if self.active_id is not None:
            return None

        visible = [obs for obs in candidates
                   if obs.is_visible
                   and self._moving_enough_to_acquire(obs, track_length)]
        if not visible:
            return None
        selected = min(
            visible,
            key=lambda obs: initial_candidate_key(obs, ego_s, track_length))
        self.active_id = int(selected.id)
        # WHAT it measured, not just that it acquired. A box that trips this is
        # the failure to diagnose, and "how far did it think that thing went"
        # is the one number that separates the candidates - approach drift,
        # a corner-switch step, or a track that inherited another one's
        # history. Reported against the threshold so the margin is visible.
        moved = self._acquire_displacement(selected.id, track_length)
        samples = self.motion_history.get(int(selected.id), [])
        span = (samples[-1][0] - samples[0][0]) if len(samples) > 1 else 0.0
        # Whether the common-mode correction fired, and what it removed. Two
        # runs have now been settled by putting the measurement in the log
        # rather than reasoning about it, and the quorum is the next thing that
        # can silently not apply: the scene is often one or two tracks, and
        # opponent_acquire_common_min_tracks needs three.
        raw = self._signed_displacement(selected.id, track_length)
        common = self._common_drift(track_length, exclude_id=selected.id)
        tracks = len(self.motion_history)
        if moved is None:
            self._log(f"acquired dynamic opponent tracker {self.active_id}")
        else:
            drift = ("none - only "
                     f"{tracks} track(s), needs "
                     f"{int(self._param('opponent_acquire_common_min_tracks'))}"
                     if common is None else f"{common:+.3f} m over {tracks} tracks")
            self._log(
                f"acquired dynamic opponent tracker {self.active_id}: moved "
                f"{moved:.3f} m over {span:.2f} s "
                f"(raw {raw:+.3f}, scene drift removed: {drift}; "
                f"threshold {float(self._param('opponent_acquire_displacement_m')):.3f}), "
                f"now {circular_forward_delta(selected.s_center, ego_s, track_length):.2f} m "
                f"ahead at d={float(selected.d_center):+.2f}, "
                f"vs={float(selected.vs):+.2f}")
        return selected

    # ------------------------------------------------------------- speed
    def stabilize_speed(self, obstacle, now):
        """Carry the last credible speed across an invisible or noisy frame.

        Mutates ``obstacle`` in place, so hand it a copy. The hold is bounded
        by dynamic_speed_hold_sec and its timestamp is never refreshed by an
        invisible frame, so this cannot become a permanent ghost speed.
        """
        measured = float(obstacle.vs)
        valid = (
            math.isfinite(measured)
            and abs(measured) >= float(self._param("dynamic_speed_valid_min_mps"))
            and abs(measured) <= float(self._param("dynamic_speed_valid_max_mps"))
        )
        if obstacle.is_visible and valid:
            self.last_speed = measured
            self.last_speed_at = now
        elif (
            self.last_speed is not None
            and self.last_speed_at is not None
            and now - self.last_speed_at
            <= float(self._param("dynamic_speed_hold_sec"))
        ):
            obstacle.vs = float(self.last_speed)

        self.last_s = float(obstacle.s_center)
        self.last_d = float(obstacle.d_center)
        self.last_position_at = now
