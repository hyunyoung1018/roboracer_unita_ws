#!/usr/bin/env python3
"""Accumulate one dynamic opponent's measured Frenet trajectory."""

from collections import deque
import math

import numpy as np
import rclpy
from f110_msgs.msg import (
    ObstacleArray,
    OpponentTrajectory,
    ProjOppPoint,
    ProjOppTraj,
    WpntArray,
)
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray

from frenet_conversion.frenet_converter import FrenetConverter
from .prediction_node import (
    circular_signed_delta,
    nearest_dynamic,
    periodic_interp,
    stamp_seconds,
    trajectory_markers,
)


class OpponentTrajectoryCollector(Node):
    """Collect stable per-distance samples and publish projected observations."""

    def __init__(self):
        super().__init__('opponent_trajectory')
        defaults = {
            'loop_rate': 25.0,
            'max_opponent_distance': 30.0,
            'max_measurement_age': 0.5,
            'min_sample_distance': 0.10,
            'max_sample_gap': 2.5,
            'min_publish_samples': 8,
            'max_samples': 400,
            'off_trajectory_threshold': 0.30,
            'off_trajectory_samples': 5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.ego_s = None
        self.track_length = None
        self.converter = None
        self.latest_obstacles = ObstacleArray()
        self.learned = None
        self.active_id = None
        self.last_s = None
        self.last_time = None
        self.traversed = 0.0
        self.off_count = 0
        self.samples = deque(maxlen=int(self.get_parameter('max_samples').value))

        self.projected_pub = self.create_publisher(
            ProjOppTraj, '/proj_opponent_trajectory', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/opponent_marker', 10)
        # Generic source name is remapped to /tracking/dynamic_obstacles.
        self.create_subscription(
            ObstacleArray, '/tracking/obstacles', self._obstacle_cb, 10)
        self.create_subscription(
            Odometry, '/car_state/odom_frenet', self._odom_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self._global_cb, 10)
        self.create_subscription(
            OpponentTrajectory, '/opponent_trajectory', self._learned_cb, 10)
        rate = float(self.get_parameter('loop_rate').value)
        self.create_timer(1.0 / max(rate, 1.0), self._sample)

    def _global_cb(self, msg):
        if not msg.wpnts:
            return
        self.track_length = float(msg.wpnts[-1].s_m)
        x = np.asarray([w.x_m for w in msg.wpnts], dtype=float)
        y = np.asarray([w.y_m for w in msg.wpnts], dtype=float)
        psi = np.asarray([w.psi_rad for w in msg.wpnts], dtype=float)
        self.converter = FrenetConverter(x, y, psi)

    def _odom_cb(self, msg):
        self.ego_s = float(msg.pose.pose.position.x)

    def _obstacle_cb(self, msg):
        self.latest_obstacles = msg

    def _learned_cb(self, msg):
        self.learned = msg

    def _reset_for_id(self, obstacle_id):
        self.active_id = int(obstacle_id)
        self.last_s = None
        self.last_time = None
        self.traversed = 0.0
        self.off_count = 0
        self.samples.clear()

    def _sample(self):
        if self.ego_s is None or not self.track_length or self.converter is None:
            return
        obstacle = nearest_dynamic(
            self.latest_obstacles.obstacles,
            self.ego_s,
            self.track_length,
            float(self.get_parameter('max_opponent_distance').value),
        )
        if obstacle is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        measurement_time = stamp_seconds(self.latest_obstacles.header.stamp) or now
        if now - measurement_time > float(self.get_parameter('max_measurement_age').value):
            return
        if self.active_id != obstacle.id:
            self._reset_for_id(obstacle.id)

        if self.last_s is None:
            ds = 0.0
        else:
            ds = circular_signed_delta(obstacle.s_center, self.last_s, self.track_length)
            if ds < -0.05 or ds > float(self.get_parameter('max_sample_gap').value):
                return
            if ds < float(self.get_parameter('min_sample_distance').value):
                return
        dt = None if self.last_time is None else measurement_time - self.last_time
        computed_vs = ds / dt if dt is not None and dt > 0.01 else obstacle.vs
        vs = float(obstacle.vs) if math.isfinite(obstacle.vs) else float(computed_vs)
        if dt is not None and dt > 0.01 and math.isfinite(computed_vs):
            vs = 0.5 * vs + 0.5 * float(computed_vs)

        point = ProjOppPoint()
        point.s = float(obstacle.s_center % self.track_length)
        point.d = float(obstacle.d_center)
        point.vs = max(0.0, vs)
        point.vd = float(obstacle.vd) if math.isfinite(obstacle.vd) else 0.0
        point.is_static = False
        point.is_visible = bool(obstacle.is_visible)
        point.time = float(measurement_time)
        point.s_var = max(0.0, float(obstacle.s_var))
        point.d_var = max(0.0, float(obstacle.d_var))
        point.vs_var = max(0.0, float(obstacle.vs_var))
        point.vd_var = max(0.0, float(obstacle.vd_var))
        self.samples.append(point)
        self.traversed += max(0.0, ds)
        self.last_s = float(obstacle.s_center)
        self.last_time = measurement_time

        if self.learned is not None and self.learned.oppwpnts:
            learned_s = [w.s_m for w in self.learned.oppwpnts]
            learned_d = [w.d_m for w in self.learned.oppwpnts]
            expected = float(periodic_interp(
                [point.s], learned_s, learned_d, self.track_length)[0])
            if abs(point.d - expected) > float(
                    self.get_parameter('off_trajectory_threshold').value):
                self.off_count += 1
            else:
                self.off_count = 0

        if len(self.samples) >= int(self.get_parameter('min_publish_samples').value):
            self._publish()

    def _publish(self):
        message = ProjOppTraj()
        message.lapcount = float(self.traversed / self.track_length)
        message.nrofpoints = float(len(self.samples))
        message.opp_is_on_trajectory = self.off_count < int(
            self.get_parameter('off_trajectory_samples').value)
        message.detections = list(self.samples)
        self.projected_pub.publish(message)

        points = list(self.samples)[-25:]
        sd = np.asarray([[p.s, p.d] for p in points])
        xy = self.converter.get_cartesian(sd[:, 0], sd[:, 1]).T
        header = self.latest_obstacles.header
        header.frame_id = 'map'
        self.marker_pub.publish(trajectory_markers(
            header,
            [(x, y, 0.12) for x, y in xy],
            'opponent_measurements',
            (0.0, 1.0, 1.0),
        ))


def main(args=None):
    rclpy.init(args=args)
    node = OpponentTrajectoryCollector()
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
