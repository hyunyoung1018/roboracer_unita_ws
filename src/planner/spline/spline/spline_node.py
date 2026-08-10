#!/usr/bin/env python3
"""Lightweight Frenet spline avoidance adapted from UNICORN's spliner node."""

import time

import numpy as np
import rclpy
from f110_msgs.msg import ObstacleArray, OTWpntArray, WpntArray, Wpnt
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from scipy.interpolate import CubicSpline
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

from frenet_conversion.frenet_converter import FrenetConverter


class SplineNode(Node):
    """Generate a local overtaking line around the nearest relevant obstacle."""

    def __init__(self):
        super().__init__('spline_node')
        self.obstacles = ObstacleArray()
        self.odom = None
        self.global_msg = None
        self.scaled_msg = None
        self.converter = None

        defaults = {
            'lookahead': 5.0,
            'evasion_distance': 0.2,
            'trajectory_threshold': 0.6,
            'boundary_margin': 0.20,
            'spline_resolution': 0.10,
            'measure': False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self._load_parameters()
        self.add_on_set_parameters_callback(self._parameter_cb)

        self.create_subscription(ObstacleArray, '/tracking/obstacles', self._obstacle_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom_frenet', self._odom_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self._global_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints_scaled', self._scaled_cb, 10)
        self.path_pub = self.create_publisher(OTWpntArray, '/planner/avoidance/otwpnts', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/planner/avoidance/markers', 10)
        self.latency_pub = self.create_publisher(Float32, '/planner/avoidance/latency', 10)
        self.create_timer(0.05, self._loop)

    def _load_parameters(self):
        self.lookahead = float(self.get_parameter('lookahead').value)
        self.evasion_distance = float(self.get_parameter('evasion_distance').value)
        self.trajectory_threshold = float(self.get_parameter('trajectory_threshold').value)
        self.boundary_margin = float(self.get_parameter('boundary_margin').value)
        self.resolution = float(self.get_parameter('spline_resolution').value)
        self.measure = bool(self.get_parameter('measure').value)

    def _parameter_cb(self, params):
        attributes = {
            'lookahead': 'lookahead',
            'evasion_distance': 'evasion_distance',
            'trajectory_threshold': 'trajectory_threshold',
            'boundary_margin': 'boundary_margin',
            'spline_resolution': 'resolution',
            'measure': 'measure',
        }
        for parameter in params:
            if parameter.name in attributes:
                setattr(self, attributes[parameter.name], parameter.value)
        return SetParametersResult(successful=True)

    def _obstacle_cb(self, msg):
        self.obstacles = msg

    def _odom_cb(self, msg):
        self.odom = msg

    def _global_cb(self, msg):
        if not msg.wpnts:
            return
        self.global_msg = msg
        if self.converter is None:
            x = np.asarray([w.x_m for w in msg.wpnts])
            y = np.asarray([w.y_m for w in msg.wpnts])
            psi = np.asarray([w.psi_rad for w in msg.wpnts])
            self.converter = FrenetConverter(x, y, psi)

    def _scaled_cb(self, msg):
        if msg.wpnts:
            self.scaled_msg = msg

    def _loop(self):
        if self.odom is None or self.global_msg is None or self.scaled_msg is None or self.converter is None:
            return
        started = time.perf_counter()
        result = self._plan()
        self.path_pub.publish(result)
        self.marker_pub.publish(self._markers(result))
        if self.measure:
            self.latency_pub.publish(Float32(data=float(time.perf_counter() - started)))

    def _bail(self, out, reason):
        """Return an empty path, but say why.

        Every exit from _plan below publishes an empty OTWpntArray, which the
        state machine reads as "no avoidance available" and silently keeps
        trailing. Without a reason on the way out, a planner that never manages
        to route around anything looks identical to one that is not running.
        """
        out.wpnts.clear()
        self.get_logger().info(f"no avoidance path: {reason}", throttle_duration_sec=2.0)
        return out

    def _plan(self):
        out = OTWpntArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        cur_s = self.odom.pose.pose.position.x
        max_s = self.global_msg.wpnts[-1].s_m
        if max_s <= 0.0:
            return out

        candidates = [
            obstacle for obstacle in self.obstacles.obstacles
            if (obstacle.s_center - cur_s) % max_s < self.lookahead
            and abs(obstacle.d_center) < self.trajectory_threshold
        ]
        if not candidates:
            return out  # nothing in range; not a failure, so not logged
        obstacle = min(candidates, key=lambda item: (item.s_center - cur_s) % max_s)
        reference = self.scaled_msg.wpnts
        s_values = np.asarray([w.s_m for w in reference])
        apex_s = obstacle.s_center
        ref_idx = int(np.argmin(np.abs(s_values - (apex_s % max_s))))
        ref = reference[ref_idx]

        left_apex = obstacle.d_left + self.evasion_distance
        right_apex = obstacle.d_right - self.evasion_distance
        left_room = ref.d_left - left_apex
        right_room = ref.d_right + right_apex
        if left_room >= self.boundary_margin and left_room >= right_room:
            side, apex_d = 'left', max(0.0, left_apex)
        elif right_room >= self.boundary_margin:
            side, apex_d = 'right', min(0.0, right_apex)
        else:
            return self._bail(
                out,
                f"no room either side of obstacle at s={apex_s:.2f}: "
                f"left {left_room:.2f} m, right {right_room:.2f} m, "
                f"need {self.boundary_margin:.2f} "
                f"(evasion_distance {self.evasion_distance:.2f})")

        speed = max(1.0, abs(self.odom.twist.twist.linear.x))
        scale = np.clip(1.0 + speed / max(1.0, max(w.vx_mps for w in reference)), 1.0, 1.5)
        offsets = np.asarray([-4.0, -3.0, -1.5, 0.0, 2.0, 3.0, 4.5]) * scale
        control_s = apex_s + offsets
        control_d = np.zeros_like(control_s)
        control_d[3] = apex_d
        spline = CubicSpline(control_s, control_d, bc_type='natural')
        sample_unwrapped = np.arange(control_s[0], control_s[-1], self.resolution)
        sample_d = np.clip(spline(sample_unwrapped), min(0.0, apex_d), max(0.0, apex_d))
        sample_s = sample_unwrapped % max_s
        xy = self.converter.get_cartesian(sample_s, sample_d)

        for idx, (s_m, d_m) in enumerate(zip(sample_s, sample_d)):
            waypoint_idx = int(np.argmin(np.abs(s_values - s_m)))
            base = reference[waypoint_idx]
            available = base.d_left if d_m >= 0.0 else base.d_right
            if abs(d_m) > max(0.0, available - self.boundary_margin):
                return self._bail(
                    out,
                    f"path leaves the track at s={s_m:.2f}: needs d={d_m:.2f} m, "
                    f"bound {available:.2f} m less {self.boundary_margin:.2f} margin")
            out.wpnts.append(Wpnt(
                id=idx, s_m=float(s_m), d_m=float(d_m),
                x_m=float(xy[0, idx]), y_m=float(xy[1, idx]),
                psi_rad=base.psi_rad, kappa_radpm=base.kappa_radpm,
                vx_mps=base.vx_mps, ax_mps2=base.ax_mps2,
                d_left=base.d_left, d_right=base.d_right,
            ))
        out.ot_side = side
        out.ot_line = side
        return out

    def _markers(self, path):
        markers = MarkerArray()
        if not path.wpnts:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.action = Marker.DELETEALL
            markers.markers.append(marker)
            return markers
        for waypoint in path.wpnts:
            marker = Marker()
            marker.header = path.header
            marker.ns = 'spline_path'
            marker.id = waypoint.id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = waypoint.x_m
            marker.pose.position.y = waypoint.y_m
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.08
            marker.color.a = 1.0
            marker.color.g = 1.0
            markers.markers.append(marker)
        return markers


def main(args=None):
    rclpy.init(args=args)
    node = SplineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
