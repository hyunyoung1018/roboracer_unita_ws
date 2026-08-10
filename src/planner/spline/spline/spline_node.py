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


class SplineNode(Node):
    """Generate a local overtaking line around the nearest relevant obstacle."""

    def __init__(self):
        super().__init__('spline_node')
        self.obstacles = ObstacleArray()
        self.odom = None
        self.global_msg = None
        self.scaled_msg = None
        self.converter = None

        self.last_path = None

        defaults = {
            'lookahead': 5.0,
            'evasion_distance': 0.4,
            'trajectory_threshold': 0.6,
            'boundary_margin': 0.20,
            'spline_resolution': 0.10,
            'rate_hz': 10.0,
            'path_hold_s': 0.3,
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
        # Rate is fixed at construction: a timer period cannot be changed at
        # runtime, and this one competes for the jetson with the detector and
        # the particle filter. 20 Hz replanned a static obstacle every 5 cm of
        # travel, for nothing.
        self.create_timer(1.0 / max(1.0, float(self.get_parameter('rate_hz').value)), self._loop)

    def _load_parameters(self):
        self.lookahead = float(self.get_parameter('lookahead').value)
        self.evasion_distance = float(self.get_parameter('evasion_distance').value)
        self.trajectory_threshold = float(self.get_parameter('trajectory_threshold').value)
        self.boundary_margin = float(self.get_parameter('boundary_margin').value)
        self.resolution = float(self.get_parameter('spline_resolution').value)
        self.path_hold_s = float(self.get_parameter('path_hold_s').value)
        self.measure = bool(self.get_parameter('measure').value)

    def _parameter_cb(self, params):
        attributes = {
            'lookahead': 'lookahead',
            'evasion_distance': 'evasion_distance',
            'trajectory_threshold': 'trajectory_threshold',
            'boundary_margin': 'boundary_margin',
            'spline_resolution': 'resolution',
            'path_hold_s': 'path_hold_s',
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

        # Hold the last good path across a dropped frame.
        #
        # Every bail publishes an empty path, and the state machine caches
        # whatever arrived last - so one bad frame in two erased the good one
        # before it could be acted on. Measured on test_213: the planner
        # alternated between a valid path and "no room" on the SAME stationary
        # obstacle, several times a second, and static avoidance never once
        # latched because have_path was false at almost every check.
        #
        # The held path keeps its ORIGINAL stamp. That is the point: it is not
        # claimed to be fresh, and the state machine's own latest_threshold
        # (0.25 s) still decides whether it is too old to use. This only stops
        # a gap from being filled with a positive assertion that there is
        # nothing there.
        if result.wpnts:
            self.last_path = result
        elif self.last_path is not None:
            age = self._age(self.last_path)
            if age <= self.path_hold_s:
                result = self.last_path
            else:
                self.last_path = None

        self.path_pub.publish(result)
        self.marker_pub.publish(self._markers(result))
        if self.measure:
            self.latency_pub.publish(Float32(data=float(time.perf_counter() - started)))

    def _age(self, msg):
        stamp = msg.header.stamp
        return (self.get_clock().now().nanoseconds * 1e-9
                - (stamp.sec + stamp.nanosec * 1e-9))

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
            if self.obstacles.obstacles:
                nearest = min(self.obstacles.obstacles,
                              key=lambda o: (o.s_center - cur_s) % max_s)
                self.get_logger().info(
                    f"{len(self.obstacles.obstacles)} obstacle(s), none in range: "
                    f"nearest is {(nearest.s_center - cur_s) % max_s:.2f} m ahead "
                    f"(lookahead {self.lookahead}) at d {nearest.d_center:.2f} "
                    f"(threshold {self.trajectory_threshold})",
                    throttle_duration_sec=2.0)
            return out
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
        # A negative left_apex (or positive right_apex) means the obstacle's
        # near edge is already further from the raceline than evasion_distance
        # - the line clears it as it is. Clamping that to 0.0 published the
        # raceline itself as an "avoidance path", which the state machine then
        # measured against the obstacle and refused, over and over. Say so and
        # publish nothing instead: there is nothing to avoid.
        if left_room >= self.boundary_margin and left_room >= right_room:
            if left_apex <= 0.0:
                return self._bail(
                    out,
                    f"raceline already clears the obstacle at s={apex_s:.2f} on the "
                    f"left (its edge is {-obstacle.d_left:.2f} m to the right of the "
                    f"line, evasion_distance {self.evasion_distance:.2f})")
            side, apex_d = 'left', left_apex
        elif right_room >= self.boundary_margin:
            if right_apex >= 0.0:
                return self._bail(
                    out,
                    f"raceline already clears the obstacle at s={apex_s:.2f} on the "
                    f"right (its edge is {obstacle.d_right:.2f} m to the left of the "
                    f"line, evasion_distance {self.evasion_distance:.2f})")
            side, apex_d = 'right', right_apex
        else:
            return self._bail(
                out,
                f"no room either side of obstacle at s={apex_s:.2f}: "
                f"left {left_room:.2f} m, right {right_room:.2f} m, "
                f"need {self.boundary_margin:.2f} "
                f"(obstacle {obstacle.d_left - obstacle.d_right:.2f} m wide "
                f"at d {obstacle.d_center:+.2f}, track "
                f"{ref.d_left + ref.d_right:.2f} m, "
                f"evasion_distance {self.evasion_distance:.2f})")

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

        # Heading and curvature OF THE SPLINE, not of the raceline underneath
        # it. Copying base.psi_rad / base.kappa_radpm told the controller the
        # evasion path was as straight as the line it departs from, and the
        # controller acts on both: curvature sets the L1 lookahead and scales
        # the speed down for a bend, and psi is what the heading-error speed
        # cut compares the car against. An evasion swerve read as straight is
        # taken too fast, with too long a lookahead.
        #
        # Both come from the cartesian samples, so they describe the path the
        # car will actually drive. psi is atan2(dy, dx), which is the same
        # convention the waypoints already carry - the raceline generator adds
        # pi/2 to tph's y-axis heading for exactly this (raceline/map_processing
        # conv_psi). Curvature is (x'y'' - y'x'') / (x'^2 + y'^2)^1.5, which is
        # invariant to the sample spacing, so the gradients need no scaling.
        px = np.asarray(xy[0], dtype=float)
        py = np.asarray(xy[1], dtype=float)
        if px.size >= 3:
            dx, dy = np.gradient(px), np.gradient(py)
            ddx, ddy = np.gradient(dx), np.gradient(dy)
            psi_path = np.arctan2(dy, dx)
            denom = np.power(dx * dx + dy * dy, 1.5)
            kappa_path = np.divide(dx * ddy - dy * ddx, denom,
                                   out=np.zeros_like(denom), where=denom > 1e-9)
        else:
            psi_path = np.zeros(px.size)
            kappa_path = np.zeros(px.size)

        for idx, (s_m, d_m) in enumerate(zip(sample_s, sample_d)):
            waypoint_idx = int(np.argmin(np.abs(s_values - s_m)))
            base = reference[waypoint_idx]
            available = base.d_left if d_m >= 0.0 else base.d_right
            if abs(d_m) > max(0.0, available - self.boundary_margin):
                # Where along the spline it fails matters more than that it
                # does. Near the apex means the obstacle sits too close to a
                # wall. Well past it means the path is still offset while the
                # track narrows underneath it - the excursion is too long for
                # this track, not too wide.
                past_apex = (s_m - apex_s + max_s / 2.0) % max_s - max_s / 2.0
                return self._bail(
                    out,
                    f"path leaves the track at s={s_m:.3f} "
                    f"({past_apex:+.2f} m from the apex at s={apex_s:.2f}): "
                    f"needs d={d_m:.3f} m, bound {available:.3f} m "
                    f"less {self.boundary_margin:.2f} margin")
            out.wpnts.append(Wpnt(
                id=idx, s_m=float(s_m), d_m=float(d_m),
                x_m=float(xy[0, idx]), y_m=float(xy[1, idx]),
                psi_rad=float(psi_path[idx]), kappa_radpm=float(kappa_path[idx]),
                vx_mps=base.vx_mps, ax_mps2=base.ax_mps2,
                d_left=base.d_left, d_right=base.d_right,
            ))
        out.ot_side = side
        out.ot_line = side
        self.get_logger().info(
            f"avoiding {side} around obstacle at s={apex_s:.2f}: "
            f"apex d={apex_d:.2f} m, {len(out.wpnts)} waypoints over "
            f"s {control_s[0]:.2f}..{control_s[-1]:.2f}",
            throttle_duration_sec=2.0)
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
