#!/usr/bin/env python3
"""Multi-obstacle spline planner using the messages available in this workspace."""

import numpy as np
import rclpy
from f110_msgs.msg import ObstacleArray, OTWpntArray, Wpnt, WpntArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from std_msgs.msg import Bool, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from frenet_conversion.frenet_converter import FrenetConverter


def _path_heading_and_curvature(x, y):
    """Return heading and signed curvature for a sampled Cartesian path."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return np.zeros_like(x), np.zeros_like(x)

    edge_order = 2 if x.size >= 3 else 1
    dx = np.gradient(x, edge_order=edge_order)
    dy = np.gradient(y, edge_order=edge_order)
    heading = np.arctan2(dy, dx)
    if x.size < 3:
        return heading, np.zeros_like(x)

    ddx = np.gradient(dx, edge_order=edge_order)
    ddy = np.gradient(dy, edge_order=edge_order)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    curvature = np.divide(
        dx * ddy - dy * ddx,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-9,
    )
    return heading, curvature


class DynamicAvoidanceNode(Node):
    def __init__(self):
        super().__init__('dynamic_avoidance_node')
        self.obstacles = ObstacleArray()
        self.odom = None
        self.global_msg = None
        self.scaled_msg = None
        self.converter = None
        self.overtaking_allowed = False

        for name, default in (
            ('lookahead', 15.0), ('evasion_distance', 0.35),
            ('boundary_margin', 0.25), ('trajectory_threshold', 1.5),
            ('return_distance', 6.0), ('resolution', 0.10),
            ('require_overtaking_permission', False),
        ):
            self.declare_parameter(name, default)

        self.create_subscription(ObstacleArray, '/tracking/obstacles', self._obstacles_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom_frenet', self._odom_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self._global_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints_scaled', self._scaled_cb, 10)
        self.create_subscription(Bool, '/ot_section_check', self._permission_cb, 10)
        self.path_pub = self.create_publisher(OTWpntArray, '/planner/avoidance/otwpnts', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/planner/avoidance/markers', 10)
        self.merge_pub = self.create_publisher(Float32MultiArray, '/planner/avoidance/merger', 10)
        self.create_timer(0.05, self._loop)

    def _obstacles_cb(self, msg):
        self.obstacles = msg

    def _odom_cb(self, msg):
        self.odom = msg

    def _global_cb(self, msg):
        if not msg.wpnts:
            return
        self.global_msg = msg
        if self.converter is None:
            self.converter = FrenetConverter(
                np.asarray([w.x_m for w in msg.wpnts]),
                np.asarray([w.y_m for w in msg.wpnts]),
                np.asarray([w.psi_rad for w in msg.wpnts]),
            )

    def _scaled_cb(self, msg):
        if msg.wpnts:
            self.scaled_msg = msg

    def _permission_cb(self, msg):
        self.overtaking_allowed = msg.data

    def _loop(self):
        if any(value is None for value in (self.odom, self.global_msg, self.scaled_msg, self.converter)):
            return
        if self.get_parameter('require_overtaking_permission').value and not self.overtaking_allowed:
            self._publish_empty()
            return
        path, merge = self._plan()
        self.path_pub.publish(path)
        self.marker_pub.publish(self._markers(path))
        if merge is not None:
            self.merge_pub.publish(Float32MultiArray(data=merge))

    def _plan(self):
        result = OTWpntArray()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = 'map'
        cur_s = self.odom.pose.pose.position.x
        max_s = self.global_msg.wpnts[-1].s_m
        if max_s <= 0.0:
            return result, None

        lookahead = float(self.get_parameter('lookahead').value)
        threshold = float(self.get_parameter('trajectory_threshold').value)
        relevant = [o for o in self.obstacles.obstacles
                    if (o.s_center - cur_s) % max_s < lookahead
                    and abs(o.d_center - self.odom.pose.pose.position.y) < threshold]
        relevant.sort(key=lambda o: (o.s_center - cur_s) % max_s)
        if not relevant:
            return result, None

        reference = self.scaled_msg.wpnts
        ref_s = np.asarray([w.s_m for w in reference])
        evasion = float(self.get_parameter('evasion_distance').value)
        margin = float(self.get_parameter('boundary_margin').value)
        controls_s = [cur_s]
        controls_d = [self.odom.pose.pose.position.y]
        side_votes = []
        for obstacle in relevant:
            idx = int(np.argmin(np.abs(ref_s - obstacle.s_center % max_s)))
            base = reference[idx]
            left_d = obstacle.d_left + evasion
            right_d = obstacle.d_right - evasion
            left_room = base.d_left - left_d
            right_room = base.d_right + right_d
            if left_room >= margin and left_room >= right_room:
                apex = max(0.0, left_d)
                side_votes.append('left')
            elif right_room >= margin:
                apex = min(0.0, right_d)
                side_votes.append('right')
            else:
                continue
            unwrapped = cur_s + (obstacle.s_center - cur_s) % max_s
            if unwrapped > controls_s[-1] + 0.05:
                controls_s.append(unwrapped)
                controls_d.append(apex)

        if len(controls_s) == 1:
            return result, None
        end_s = controls_s[-1] + float(self.get_parameter('return_distance').value)
        controls_s.append(end_s)
        controls_d.append(0.0)
        if len(controls_s) == 2:
            return result, None

        spline = CubicSpline(np.asarray(controls_s), np.asarray(controls_d), bc_type='clamped')
        resolution = float(self.get_parameter('resolution').value)
        sample_unwrapped = np.arange(controls_s[0], controls_s[-1], resolution)
        sample_d = spline(sample_unwrapped)
        if len(sample_d) >= 7:
            window = min(31, len(sample_d) if len(sample_d) % 2 else len(sample_d) - 1)
            if window >= 5:
                sample_d = savgol_filter(sample_d, window, 2)
        sample_s = sample_unwrapped % max_s
        xy = self.converter.get_cartesian(sample_s, sample_d)
        path_psi, path_kappa = _path_heading_and_curvature(xy[0], xy[1])

        for i, (s_m, d_m) in enumerate(zip(sample_s, sample_d)):
            idx = int(np.argmin(np.abs(ref_s - s_m)))
            base = reference[idx]
            room = base.d_left if d_m >= 0.0 else base.d_right
            if abs(d_m) > max(0.0, room - margin):
                result.wpnts.clear()
                return result, None
            result.wpnts.append(Wpnt(
                id=i, s_m=float(s_m), d_m=float(d_m),
                x_m=float(xy[0, i]), y_m=float(xy[1, i]),
                d_left=base.d_left, d_right=base.d_right,
                psi_rad=float(path_psi[i]), kappa_radpm=float(path_kappa[i]),
                vx_mps=base.vx_mps, ax_mps2=base.ax_mps2,
            ))
        result.ot_side = max(set(side_votes), key=side_votes.count)
        result.ot_line = result.ot_side
        return result, [float(relevant[-1].s_end % max_s), float(end_s % max_s)]

    def _publish_empty(self):
        msg = OTWpntArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        self.path_pub.publish(msg)
        self.marker_pub.publish(self._markers(msg))

    def _markers(self, path):
        array = MarkerArray()
        if not path.wpnts:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.action = Marker.DELETEALL
            array.markers.append(marker)
            return array
        for waypoint in path.wpnts:
            marker = Marker()
            marker.header = path.header
            marker.ns = 'dynamic_spline_path'
            marker.id = waypoint.id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = waypoint.x_m
            marker.pose.position.y = waypoint.y_m
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.08
            marker.color.a = 1.0
            marker.color.r = 0.2
            marker.color.g = 0.8
            marker.color.b = 1.0
            array.markers.append(marker)
        return array


def main(args=None):
    rclpy.init(args=args)
    node = DynamicAvoidanceNode()
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
