"""Pure helpers for selecting one head-to-head opponent.

The shared tracker deliberately reports every object it can see.  Head-to-head
prediction has a stricter contract: exactly one physical opponent, inside the
drivable corridor and ahead of the ego car.  Keeping these calculations free
of ROS makes the safety gates straightforward to unit-test.
"""

import math


def circular_forward_delta(value, reference, track_length):
    return (float(value) - float(reference)) % float(track_length)


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
    """Check the opponent is in the forward head-to-head observation window."""
    if ego_s is None or not track_length:
        return False
    gap = circular_forward_delta(obstacle.s_center, ego_s, track_length)
    return float(minimum_distance) <= gap <= float(maximum_distance)


def initial_candidate_key(obstacle, ego_s, track_length):
    """Prefer the raceline-like target, then the nearer forward target.

    In the box-and-operator test the box is placed on the driving line while the
    operator walks laterally beside it.  Once acquired, the router locks the
    target and this key is no longer consulted.
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
