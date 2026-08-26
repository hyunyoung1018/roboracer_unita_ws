"""
Conversion from numpy trajectories to f110_msgs/WpntArray and RViz markers.

Ported from ForzaETH race_stack (global_planner_utils.py). Kept byte-compatible
with their message layout so downstream consumers (controller, state machine,
sector tuner) work unchanged.
"""

import numpy as np
from f110_msgs.msg import Wpnt, WpntArray
from visualization_msgs.msg import Marker, MarkerArray

from .map_processing import conv_psi


def _sphere(frame_id: str, marker_id: int, x: float, y: float,
            rgba, scale: float = 0.05) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = Marker.SPHERE
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba
    marker.id = marker_id
    marker.pose.position.x = float(x)
    marker.pose.position.y = float(y)
    marker.pose.orientation.w = 1.0
    return marker


def create_centerline_markers(centerline: np.ndarray, frame_id: str = 'map'):
    """
    Build the centerline WpntArray + MarkerArray from
    [x_m, y_m, w_tr_right_m, w_tr_left_m].
    """
    markers = MarkerArray()
    wpnts = WpntArray()

    for i, row in enumerate(centerline):
        markers.markers.append(
            _sphere(frame_id, i, row[0], row[1], (0.0, 0.0, 1.0, 1.0)))

        wpnt = Wpnt()
        wpnt.id = i
        wpnt.x_m = float(row[0])
        wpnt.y_m = float(row[1])
        wpnt.d_right = float(row[2])
        wpnt.d_left = float(row[3])
        wpnts.wpnts.append(wpnt)

    return wpnts, markers


def add_centerline_heading(wpnts: WpntArray, stepsize: float = 0.1) -> None:
    """
    Fill s_m / psi_rad / kappa_radpm on centerline waypoints, in place.

    The centerline is emitted at a fixed step, so s is just index * stepsize.
    psi gets the same +pi/2 offset the optimizer output does, because
    trajectory_planning_helpers measures heading from +y.
    """
    import trajectory_planning_helpers as tph

    coords = np.array([[w.x_m, w.y_m] for w in wpnts.wpnts])
    if len(coords) < 3:
        return

    psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(
        path=coords,
        el_lengths=stepsize * np.ones(len(coords) - 1),
        is_closed=False)

    for i, (p, k) in enumerate(zip(psi, kappa)):
        wpnts.wpnts[i].s_m = i * stepsize
        wpnts.wpnts[i].psi_rad = float(p + np.pi / 2)
        wpnts.wpnts[i].kappa_radpm = float(k)


def create_trackbounds_markers(bound_r: np.ndarray, bound_l: np.ndarray,
                               frame_id: str = 'map') -> MarkerArray:
    markers = MarkerArray()
    marker_id = 0

    for pnt in bound_r:
        markers.markers.append(
            _sphere(frame_id, marker_id, pnt[0], pnt[1], (0.5, 0.0, 0.5, 1.0)))
        marker_id += 1
    for pnt in bound_l:
        markers.markers.append(
            _sphere(frame_id, marker_id, pnt[0], pnt[1], (0.5, 1.0, 0.0, 1.0)))
        marker_id += 1

    return markers


def create_wpnts_markers(trajectory: np.ndarray, d_right: np.ndarray, d_left: np.ndarray,
                         second_traj: bool = False, frame_id: str = 'map'):
    """
    Build the global trajectory WpntArray + MarkerArray.

    `trajectory` rows are [s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2].
    Marker height encodes speed normalized to the fastest point, so the raceline
    reads as a speed profile in RViz at a glance. `second_traj` colours the
    shortest path differently from the main raceline.
    """
    max_vx_mps = float(np.max(trajectory[:, 5]))

    global_wpnts = WpntArray()
    global_markers = MarkerArray()

    for i, pnt in enumerate(trajectory):
        wpnt = Wpnt()
        wpnt.id = i
        wpnt.s_m = float(pnt[0])
        wpnt.x_m = float(pnt[1])
        wpnt.y_m = float(pnt[2])
        wpnt.d_right = float(d_right[i])
        wpnt.d_left = float(d_left[i])
        wpnt.psi_rad = conv_psi(float(pnt[3]))
        wpnt.kappa_radpm = float(pnt[4])
        wpnt.vx_mps = float(pnt[5])
        wpnt.ax_mps2 = float(pnt[6])
        global_wpnts.wpnts.append(wpnt)

        height = wpnt.vx_mps / max_vx_mps if max_vx_mps > 0.0 else 0.0
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = Marker.CYLINDER
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = height
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0 if second_traj else 0.0
        marker.id = i
        marker.pose.position.x = float(pnt[1])
        marker.pose.position.y = float(pnt[2])
        marker.pose.position.z = height / 2
        marker.pose.orientation.w = 1.0
        global_markers.markers.append(marker)

    return global_wpnts, global_markers
