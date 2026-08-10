#!/usr/bin/env python3
"""Nearest-neighbour obstacle tracker in Frenet coordinates."""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import rclpy
from f110_msgs.msg import Obstacle, ObstacleArray, WpntArray
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray


def _circular_delta(a, b, length):
    return (a - b + length / 2.0) % length - length / 2.0


@dataclass
class Track:
    track_id: int
    obstacle: Obstacle
    stamp: float
    missed: int = 0
    s_history: List[float] = field(default_factory=list)
    d_history: List[float] = field(default_factory=list)


class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking')
        for name, default in (
            ('association_distance', 0.8), ('max_missed_frames', 8),
            ('velocity_smoothing', 0.65), ('static_speed_threshold', 0.15),
            ('static_min_samples', 6), ('publish_static', True), ('measure', False),
        ):
            self.declare_parameter(name, default)
        self.track_length = None
        self.tracks = []
        self.next_id = 1
        self.create_subscription(WpntArray, '/global_waypoints_scaled', self._path_cb, 10)
        self.create_subscription(ObstacleArray, '/detect/raw_obstacles', self._obstacles_cb, 10)
        self.obstacle_pub = self.create_publisher(ObstacleArray, '/tracking/obstacles', 5)
        self.raw_pub = self.create_publisher(ObstacleArray, '/tracking/raw_obstacles', 5)
        self.marker_pub = self.create_publisher(MarkerArray, '/tracking/static_dynamic_marker_pub', 5)
        self.latency_pub = self.create_publisher(Float32, '/tracking/latency', 5)
        self.get_logger().info('Waiting for /global_waypoints_scaled and /detect/raw_obstacles')

    def _path_cb(self, msg):
        if msg.wpnts:
            self.track_length = msg.wpnts[-1].s_m

    def _obstacles_cb(self, msg):
        if not self.track_length:
            return
        started = self.get_clock().now().nanoseconds / 1e9
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if stamp <= 0.0:
            stamp = started
        unmatched_tracks = set(range(len(self.tracks)))
        assignments = []

        for measurement in msg.obstacles:
            best, best_distance = None, float('inf')
            for index in unmatched_tracks:
                track = self.tracks[index]
                ds = _circular_delta(measurement.s_center, track.obstacle.s_center, self.track_length)
                dd = measurement.d_center - track.obstacle.d_center
                distance = float(np.hypot(ds, dd))
                if distance < best_distance:
                    best, best_distance = index, distance
            if best is not None and best_distance <= float(self.get_parameter('association_distance').value):
                unmatched_tracks.remove(best)
                assignments.append((self.tracks[best], measurement))
            else:
                assignments.append((self._new_track(measurement, stamp), measurement))

        for index in unmatched_tracks:
            self.tracks[index].missed += 1
        for track, measurement in assignments:
            self._update_track(track, measurement, stamp)
        max_missed = int(self.get_parameter('max_missed_frames').value)
        self.tracks = [track for track in self.tracks if track.missed <= max_missed]

        output = ObstacleArray()
        output.header = msg.header
        publish_static = bool(self.get_parameter('publish_static').value)
        # `missed <= max_missed`, not `missed == 0`. Publishing only what is
        # seen this instant made max_missed_frames do nothing downstream: one
        # dropped lidar frame and the obstacle vanished from /tracking/obstacles
        # entirely, so the state machine saw present/absent/present and its
        # avoidance hysteresis had nothing stable to hold on to. A track the
        # tracker still believes in is still reported; is_visible says whether
        # it was seen this frame, which is what that field is for.
        for track in self.tracks:
            if publish_static or not track.obstacle.is_static:
                output.obstacles.append(self._obstacle_for_output(track, stamp))
        self.obstacle_pub.publish(output)
        self.raw_pub.publish(msg)
        self.marker_pub.publish(self._markers(output))
        if self.get_parameter('measure').value:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - started
            self.latency_pub.publish(Float32(data=float(elapsed)))

    def _new_track(self, measurement, stamp):
        copied = self._copy_obstacle(measurement)
        copied.id = self.next_id
        track = Track(self.next_id, copied, stamp,
                      s_history=[copied.s_center], d_history=[copied.d_center])
        self.next_id += 1
        self.tracks.append(track)
        return track

    def _update_track(self, track, measurement, stamp):
        if track.stamp == stamp and len(track.s_history) == 1:
            track.missed = 0
            return
        dt = max(1e-3, stamp - track.stamp)
        ds = _circular_delta(measurement.s_center, track.obstacle.s_center, self.track_length)
        dd = measurement.d_center - track.obstacle.d_center
        alpha = float(self.get_parameter('velocity_smoothing').value)
        measured_vs, measured_vd = ds / dt, dd / dt
        previous_vs, previous_vd = track.obstacle.vs, track.obstacle.vd
        updated = self._copy_obstacle(measurement)
        updated.id = track.track_id
        updated.vs = alpha * previous_vs + (1.0 - alpha) * measured_vs
        updated.vd = alpha * previous_vd + (1.0 - alpha) * measured_vd
        track.obstacle = updated
        track.stamp = stamp
        track.missed = 0
        track.s_history.append(updated.s_center)
        track.d_history.append(updated.d_center)
        track.s_history = track.s_history[-20:]
        track.d_history = track.d_history[-20:]
        min_samples = int(self.get_parameter('static_min_samples').value)
        speed_limit = float(self.get_parameter('static_speed_threshold').value)
        # bool(), and it is load-bearing. `a and b` returns b when a is truthy,
        # and b here is a numpy comparison, so the result is numpy.bool_ - which
        # is not a subclass of bool, and rosidl asserts on the field's type. The
        # node died the moment any track collected static_min_samples of
        # history, which is a few frames after the first obstacle appears.
        updated.is_static = bool(
            len(track.s_history) >= min_samples
            and np.hypot(updated.vs, updated.vd) < speed_limit)
        updated.is_visible = True

    @staticmethod
    def _copy_obstacle(source):
        target = Obstacle()
        for field in (
            'id', 's_start', 's_end', 'd_right', 'd_left', 'is_actually_a_gap',
            's_center', 'd_center', 'size', 'vs', 'vd', 'is_static', 'is_visible',
        ):
            setattr(target, field, getattr(source, field))
        return target

    def _obstacle_for_output(self, track, stamp):
        """Publish retained tracks through short detector dropouts.

        A missed track is no longer a current LiDAR observation, so expose that
        distinction through is_visible.  Dynamic tracks are propagated from the
        last measurement with their filtered constant velocity; static tracks
        remain at their last measured position.
        """
        obstacle = self._copy_obstacle(track.obstacle)
        obstacle.is_visible = track.missed == 0
        if track.missed == 0 or obstacle.is_static:
            return obstacle

        dt = max(0.0, stamp - track.stamp)
        ds = obstacle.vs * dt
        dd = obstacle.vd * dt
        obstacle.s_center = float((obstacle.s_center + ds) % self.track_length)
        obstacle.s_start = float((obstacle.s_start + ds) % self.track_length)
        obstacle.s_end = float((obstacle.s_end + ds) % self.track_length)
        obstacle.d_center = float(obstacle.d_center + dd)
        obstacle.d_right = float(obstacle.d_right + dd)
        obstacle.d_left = float(obstacle.d_left + dd)
        return obstacle

    def _markers(self, obstacles):
        array = MarkerArray()
        if not obstacles.obstacles:
            marker = Marker()
            marker.header = obstacles.header
            marker.header.frame_id = 'map'
            marker.action = Marker.DELETEALL
            array.markers.append(marker)
            return array
        for obstacle in obstacles.obstacles:
            marker = Marker()
            marker.header = obstacles.header
            marker.header.frame_id = 'map'
            marker.ns = 'tracked_obstacles'
            marker.id = obstacle.id
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            # Frenet coordinates are intentionally shown as text-only metadata;
            # detector markers provide the Cartesian obstacle positions.
            marker.pose.position.z = 0.5
            marker.scale.z = 0.18
            marker.color.a = 1.0
            marker.color.r = 0.2 if obstacle.is_static else 1.0
            marker.color.g = 1.0 if obstacle.is_static else 0.2
            marker.text = f'id={obstacle.id} s={obstacle.s_center:.2f} d={obstacle.d_center:.2f} v={obstacle.vs:.2f}'
            array.markers.append(marker)
        return array


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
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
