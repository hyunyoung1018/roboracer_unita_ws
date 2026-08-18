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

        visible = [obs for obs in candidates if obs.is_visible]
        if not visible:
            return None
        selected = min(
            visible,
            key=lambda obs: initial_candidate_key(obs, ego_s, track_length))
        self.active_id = int(selected.id)
        self._log(
            f"acquired dynamic opponent tracker {self.active_id} as "
            f"logical opponent {int(self._param('logical_opponent_id'))}")
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
