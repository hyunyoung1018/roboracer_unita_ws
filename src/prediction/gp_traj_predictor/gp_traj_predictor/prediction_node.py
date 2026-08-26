"""Shared numerical and ROS-message helpers for opponent prediction."""

import numpy as np
from visualization_msgs.msg import Marker, MarkerArray


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def circular_signed_delta(value, reference, length):
    return (float(value) - float(reference) + 0.5 * length) % length - 0.5 * length


def circular_forward_delta(value, reference, length):
    return (float(value) - float(reference)) % length


def periodic_interp(query, s, values, length):
    s = np.asarray(s, dtype=float) % length
    values = np.asarray(values, dtype=float)
    if len(s) == 0:
        return np.zeros_like(np.asarray(query, dtype=float))
    order = np.argsort(s)
    s, values = s[order], values[order]
    unique = np.concatenate([[True], np.diff(s) > 1e-6])
    s, values = s[unique], values[unique]
    extended_s = np.concatenate([s - length, s, s + length])
    extended_values = np.concatenate([values, values, values])
    return np.interp(np.asarray(query, dtype=float), extended_s, extended_values)


def nearest_dynamic(obstacles, ego_s, track_length, max_distance):
    candidates = [
        obstacle for obstacle in obstacles
        if not obstacle.is_static
        and circular_forward_delta(obstacle.s_center, ego_s, track_length) <= max_distance
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda obstacle: circular_forward_delta(obstacle.s_center, ego_s, track_length),
    )


def ccma_smooth(points, window=5):
    """Curvature-corrected moving average used for learned lateral profiles."""
    points = np.asarray(points, dtype=float)
    if len(points) < window:
        return points.copy()
    window = max(3, int(window) | 1)
    half = window // 2
    padded = np.pad(points, ((half, half), (0, 0)), mode='wrap')
    kernel = np.ones(window) / window
    moving = np.column_stack([
        np.convolve(padded[:, axis], kernel, mode='valid') for axis in range(2)
    ])
    detail = points - moving
    correction = (np.roll(detail, 1, axis=0) + detail + np.roll(detail, -1, axis=0)) / 3.0
    return moving + 0.35 * correction


def trajectory_markers(header, points, namespace, color):
    array = MarkerArray()
    delete = Marker(header=header, action=Marker.DELETEALL)
    array.markers.append(delete)
    for index, (x, y, scale) in enumerate(points):
        marker = Marker(header=header)
        marker.ns = namespace
        marker.id = index
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = max(0.025, float(scale) * 0.5)
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = 0.10
        marker.scale.z = max(0.05, float(scale))
        marker.color.a = 0.9
        marker.color.r, marker.color.g, marker.color.b = color
        array.markers.append(marker)
    return array
