#!/usr/bin/env python3
"""Publish the RECOVERY path the state machine has always expected.

Nothing in this workspace published /planner/recovery/wpnts. The state machine
subscribes to it, RecoveryTransition and both obstacle transitions branch on it,
and the entire run logged

    [state_machine]: recovery_planner: not usable - nothing received on its
                     topic yet

every two seconds. So RECOVERY was structurally unreachable, and the fallback is
NonObstacleTransition returning (LOSTLINE, RACELINE): states.RacelineTracking
then emits

    s = int(cur_s / waypoints_dist + 0.5)
    [cur_gb_wpnts.list[(s + i) % n] for i in range(n_loc_wpnts)]

which is the raceline at the car's own s, taking no account of the car's d at
all. A car pushed off the line - by trailing a ghost, by an evasion, by anything
- is handed a path that starts wherever the raceline happens to be beside it and
cuts back. On the logged run the controller hit its 0.4 rad/tick steering rate
limit twice while doing 3.09 m/s.

What this node publishes is the missing middle: a path that starts AT the car,
at the car's own lateral offset, and blends onto the raceline over a settable
distance before continuing along it. The state machine keeps full control of
whether it is used - freshness, on-spline and free checks all still apply, and
it replans the velocity profile at recovery_planner's 0.5 safety factor.

Head-to-head only. time_trials.launch.xml starts no node from this package.
"""

import numpy as np
import rclpy
from f110_msgs.msg import Wpnt, WpntArray
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from frenet_conversion.frenet_converter import FrenetConverter


def heading_and_curvature(x, y):
    """Heading and signed curvature of a sampled Cartesian path."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return np.zeros_like(x), np.zeros_like(x)
    edge = 2 if x.size >= 3 else 1
    dx = np.gradient(x, edge_order=edge)
    dy = np.gradient(y, edge_order=edge)
    heading = np.arctan2(dy, dx)
    if x.size < 3:
        return heading, np.zeros_like(x)
    ddx = np.gradient(dx, edge_order=edge)
    ddy = np.gradient(dy, edge_order=edge)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    curvature = np.divide(
        dx * ddy - dy * ddx, denominator,
        out=np.zeros_like(denominator), where=denominator > 1e-9,
    )
    return heading, curvature


def blend_weights(distance, return_distance):
    """Raised cosine from 1 (hold the car's d) to 0 (on the raceline).

    Raised cosine rather than linear because its derivative is zero at both
    ends: the path leaves the car tangentially and arrives on the raceline
    tangentially, so neither join asks the controller for a step in curvature.
    """
    if return_distance <= 0.0:
        return np.zeros_like(distance)
    phase = np.clip(distance / return_distance, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * phase))


class RecoveryNode(Node):
    """Blend the car back onto the raceline from wherever it currently is."""

    PARAM_DEFAULTS = {
        'rate_hz': 20.0,
        # [Hz] Marker draw rate. Markers are also skipped entirely when no
        # viewer is subscribed; see _viz_due.
        'viz_rate_hz': 5.0,
        # [m] Distance over which the car's lateral offset is taken out. Long
        # enough that the manoeuvre is not a swerve at the speeds this arms at,
        # short enough that the car is back on the line before the next corner.
        'return_distance_m': 3.0,
        # [m] Total path length. NOT just return_distance_m: _check_free_frenet
        # treats a non-closed path as blocked whenever an obstacle sits beyond
        # its end (`gap > max_gap`), so a path shorter than the state machine's
        # interest_horizon_m reads as blocked exactly when there is something
        # ahead - which is when RECOVERY is wanted. Keep this at or above
        # interest_horizon_m (9.0 in h2h_state_machine_params.yaml).
        'path_length_m': 10.0,
        'resolution_m': 0.10,
        # [m] Below this lateral offset the car is already on the line and the
        # blend would be a no-op; the path is still published so the topic
        # stays fresh and the state machine's latest_threshold is satisfied.
        'min_offset_m': 0.02,
    }

    def __init__(self):
        super().__init__('recovery_node')
        for name, default in self.PARAM_DEFAULTS.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)

        self.current_s = None
        self.current_d = None
        self.scaled_msg = None
        self._scaled_s = None
        self.converter = None
        self.track_length = None

        self.path_pub = self.create_publisher(
            WpntArray, '/planner/recovery/wpnts', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/planner/recovery/markers', 10)

        self.create_subscription(
            Odometry, '/car_state/odom_frenet', self._frenet_cb, 10)
        self.create_subscription(
            WpntArray, '/global_waypoints', self._global_cb, 10)
        self.create_subscription(
            WpntArray, '/global_waypoints_scaled', self._scaled_cb, 10)
        self._viz_rate_hz = float(self.get_parameter('viz_rate_hz').value)
        self._last_viz_sec = 0.0
        self.create_timer(1.0 / max(float(self.rate_hz), 1.0), self._loop)
        self.get_logger().info(
            'Waiting for /car_state/odom_frenet and /global_waypoints_scaled')

    def _frenet_cb(self, msg):
        self.current_s = float(msg.pose.pose.position.x)
        self.current_d = float(msg.pose.pose.position.y)

    def _global_cb(self, msg):
        if not msg.wpnts:
            return
        self.track_length = float(msg.wpnts[-1].s_m)
        self.converter = FrenetConverter(
            np.asarray([w.x_m for w in msg.wpnts], dtype=float),
            np.asarray([w.y_m for w in msg.wpnts], dtype=float),
            np.asarray([w.psi_rad for w in msg.wpnts], dtype=float),
        )

    def _scaled_cb(self, msg):
        if msg.wpnts:
            self.scaled_msg = msg
            self._scaled_s = np.asarray(
                [w.s_m for w in msg.wpnts], dtype=float)

    def _reference_indices(self, s_values):
        """Nearest scaled-raceline index for each sampled s."""
        wrapped = np.asarray(s_values, dtype=float) % self.track_length
        return np.argmin(
            np.abs(self._scaled_s[None, :] - wrapped[:, None]), axis=1)

    def _loop(self):
        if any(value is None for value in (
                self.current_s, self.current_d, self.scaled_msg,
                self.converter, self.track_length)):
            return
        if self.track_length <= 0.0:
            return

        resolution = max(float(self.resolution_m), 0.01)
        length = max(float(self.path_length_m), 4.0 * resolution)
        count = int(np.ceil(length / resolution)) + 1
        travelled = np.arange(count, dtype=float) * resolution
        s_unwrapped = float(self.current_s) + travelled
        s_wrapped = s_unwrapped % self.track_length

        offset = float(self.current_d)
        if abs(offset) < float(self.min_offset_m):
            offset = 0.0
        d = offset * blend_weights(travelled, float(self.return_distance_m))

        xy = self.converter.get_cartesian(s_wrapped, d)
        if not np.all(np.isfinite(xy)):
            self.get_logger().warn(
                'recovery path is not finite; skipping this cycle',
                throttle_duration_sec=2.0)
            return
        psi, kappa = heading_and_curvature(xy[0], xy[1])
        indices = self._reference_indices(s_wrapped)

        message = WpntArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'map'
        for i, index in enumerate(indices):
            reference = self.scaled_msg.wpnts[int(index)]
            message.wpnts.append(Wpnt(
                id=i,
                s_m=float(s_wrapped[i]),
                d_m=float(d[i]),
                x_m=float(xy[0, i]),
                y_m=float(xy[1, i]),
                d_left=float(reference.d_left),
                d_right=float(reference.d_right),
                psi_rad=float(psi[i]),
                kappa_radpm=float(kappa[i]),
                # The state machine replans this at recovery_planner's 0.5
                # safety factor in recovery_wpnts_cb; seeding it from the
                # raceline keeps the profile sane if that replan is skipped.
                vx_mps=float(reference.vx_mps),
                ax_mps2=float(reference.ax_mps2),
            ))
        self.path_pub.publish(message)
        if self._viz_due():
            self.marker_pub.publish(self._markers(message))

    def _viz_due(self):
        """True when this tick should draw.

        Rate-limited, and skipped entirely when nothing is subscribed - with
        no viewer attached the markers cost nothing at all, which is how the
        car should normally race. Measured with py-spy on this node while
        rviz was closed: _markers was 13.05s of _loop's 37.96s, a third of
        everything it did, drawing for nobody. The same shape the state
        machine already uses.
        """
        if self._viz_rate_hz <= 0.0:
            return False
        if self.marker_pub.get_subscription_count() == 0:
            return False
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_viz_sec < 1.0 / self._viz_rate_hz:
            return False
        self._last_viz_sec = now
        return True

    def _markers(self, path):
        array = MarkerArray()
        array.markers.append(Marker(
            header=path.header, action=Marker.DELETEALL))
        for waypoint in path.wpnts[::5]:
            marker = Marker(header=path.header)
            marker.ns = 'recovery_path'
            marker.id = waypoint.id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = waypoint.x_m
            marker.pose.position.y = waypoint.y_m
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.07
            marker.color.a = 0.9
            marker.color.r = 1.0
            marker.color.g = 0.55
            array.markers.append(marker)
        return array


def main(args=None):
    rclpy.init(args=args)
    node = RecoveryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
