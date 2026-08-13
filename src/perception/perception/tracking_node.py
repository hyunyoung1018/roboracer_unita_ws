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
    # Held half-extents from the centre, in metres: the running maximum of
    # everything this track has been measured at. See _hold_extents.
    extents: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking')
        for name, default in (
            ('association_distance', 0.8), ('max_missed_frames', 8),
            ('velocity_smoothing', 0.65), ('static_speed_threshold', 0.15),
            ('static_min_samples', 6), ('publish_static', True), ('measure', False),
            ('extent_max_m', 0.50), ('extent_decay_mps', 0.05),
            ('static_position_samples', 10),
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

    @staticmethod
    def _measured_extents(obstacle):
        """Half-extents from the centre: back, front, right, left."""
        return [
            abs(obstacle.s_center - obstacle.s_start),
            abs(obstacle.s_end - obstacle.s_center),
            abs(obstacle.d_center - obstacle.d_right),
            abs(obstacle.d_left - obstacle.d_center),
        ]

    def _filtered_center(self, track):
        """Median of the recent measured positions, in Frenet."""
        n = int(self.get_parameter('static_position_samples').value)
        s_history = track.s_history[-n:]
        d_history = track.d_history[-n:]
        if len(s_history) < 2:
            return s_history[-1], d_history[-1]
        # s wraps, so take the median of the offsets from the newest sample
        # rather than of the values themselves.
        reference = s_history[-1]
        offsets = [_circular_delta(s, reference, self.track_length) for s in s_history]
        s_center = (reference + float(np.median(offsets))) % self.track_length
        return float(s_center), float(np.median(d_history))

    def _hold_extents(self, track, measurement, dt):
        """Hold the largest extent this obstacle has ever shown.

        The lidar only ever sees the near faces, so every single frame
        UNDER-estimates the real footprint, and by a different amount each
        time as the viewing angle changes. Feeding that straight to the
        planner moved the obstacle's edges centimetres per frame, the apex
        moved with them, and the path never settled - which is the problem
        the detector was answering by modelling every obstacle as a fixed
        0.50 m cube regardless of what was measured.

        A fixed cube is the wrong place to solve it. 0.50 m is the rule's
        MAXIMUM, not the size of the thing in front of the car, and assuming
        it costs real passable gaps: on this track a centred 0.50 m obstacle
        can be driven around at 22% of positions, a 0.30 m one at 67%.

        Because each measurement is a lower bound, the running maximum over
        the track's life converges upward to the true extent and then stops
        moving. That is stable by construction rather than by assumption, and
        it is per-obstacle, so a small cone stays small.

        Two guards: a slow leak downward, so one bad frame or a mis-associated
        detection bleeds off instead of inflating the obstacle forever; and a
        hard ceiling at the rule's maximum, which is what that number is for.
        """
        held = track.extents
        measured = self._measured_extents(measurement)
        ceiling = float(self.get_parameter('extent_max_m').value) / 2.0
        leak = float(self.get_parameter('extent_decay_mps').value) * max(0.0, dt)
        for i in range(4):
            held[i] = min(ceiling, max(measured[i], held[i] - leak))
        return held

    @staticmethod
    def _apply_extents(obstacle, held, track_length):
        back, front, right, left = held
        obstacle.s_start = float((obstacle.s_center - back) % track_length)
        obstacle.s_end = float((obstacle.s_center + front) % track_length)
        obstacle.d_right = float(obstacle.d_center - right)
        obstacle.d_left = float(obstacle.d_center + left)
        # `size` is the bounding square's side, which is what the marker draws
        # and what anything reading it has always meant.
        obstacle.size = float(max(back + front, right + left))

    def _new_track(self, measurement, stamp):
        copied = self._copy_obstacle(measurement)
        copied.id = self.next_id
        track = Track(self.next_id, copied, stamp,
                      s_history=[copied.s_center], d_history=[copied.d_center],
                      extents=self._measured_extents(copied))
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

        track.s_history.append(measurement.s_center)
        track.d_history.append(measurement.d_center)
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

        # A static obstacle is not moving, so its position should not move
        # either - but the L-shape fit lands a few centimetres differently
        # every frame as the viewing angle changes. The planner puts the
        # avoidance apex at the obstacle's edge, so that jitter propagates
        # straight into the path, which is why the local path never looked
        # settled. Take the median of the recent measurements instead: it is
        # the same estimator the extents use, and it discards the occasional
        # bad fit rather than averaging it in.
        #
        # Only for static tracks. A moving obstacle's position is supposed to
        # change, and a median would lag it.
        if updated.is_static:
            updated.s_center, updated.d_center = self._filtered_center(track)

        self._apply_extents(updated, self._hold_extents(track, measurement, dt),
                            self.track_length)
        track.obstacle = updated
        track.stamp = stamp
        track.missed = 0

    @staticmethod
    def _copy_obstacle(source):
        target = Obstacle()
        for field in (
            'id', 's_start', 's_end', 'd_right', 'd_left', 'is_actually_a_gap',
            's_center', 'd_center', 'size', 'vs', 'vd', 'is_static', 'is_visible',
            # Carried so the marker can be drawn where the obstacle is. The
            # detector fills these; without them every label stacked at the
            # map origin.
            'x_m', 'y_m', 'theta',
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
        # DELETEALL at the front of EVERY array, not only the empty ones.
        #
        # Marker ids here are track ids, and track ids only ever go up.
        # Without a delete, a viewer keeps every marker this node has ever
        # published: each phantom detection that lived eight frames leaves a
        # cylinder and a label behind forever, and after a few minutes of
        # driving past walls that is thousands of them. The car does not slow
        # down for it, but the machine drawing them does, and so does anything
        # measuring the stack through that viewer.
        #
        # It also has to be in the same message as the markers, or the frame
        # blinks - see clearMarkers in detect.cpp.
        array = MarkerArray()
        clear = Marker()
        clear.header = obstacles.header
        clear.header.frame_id = 'map'
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        if not obstacles.obstacles:
            return array
        # Draw what the PLANNER is given, not what the detector last saw.
        #
        # /detect/obstacles_markers_new draws the raw per-frame fit: a square
        # of side max(width, height) at the L-shape centre. That is not what
        # anything downstream acts on. The planner reads THIS topic, whose
        # extents are the running maximum this track has ever shown and whose
        # position, for a static obstacle, is the median of recent frames. So
        # a box seen jittering in rviz says the detector is jittering - it
        # does not say the planner is.
        #
        # A cylinder, because the number that decides everything is the
        # LATERAL extent: the planner puts the avoidance apex evasion_distance
        # beyond this edge, and the state machine subtracts this width from
        # the room either side. Drawing it as a circle of that diameter is
        # drawing the quantity, without implying an orientation the tracker
        # does not have.
        for obstacle in obstacles.obstacles:
            lateral = max(1e-3, float(obstacle.d_left) - float(obstacle.d_right))

            body = Marker()
            body.header = obstacles.header
            body.header.frame_id = 'map'
            body.ns = 'tracked_obstacles'
            body.id = obstacle.id
            body.type = Marker.CYLINDER
            body.action = Marker.ADD
            body.pose.position.x = float(obstacle.x_m)
            body.pose.position.y = float(obstacle.y_m)
            body.pose.position.z = 0.15
            body.pose.orientation.w = 1.0
            body.scale.x = body.scale.y = lateral
            body.scale.z = 0.3
            # Green static, red dynamic; hollow while the track is coasting on
            # a missed detection rather than a current one.
            body.color.a = 0.6 if obstacle.is_visible else 0.25
            body.color.r = 0.2 if obstacle.is_static else 1.0
            body.color.g = 1.0 if obstacle.is_static else 0.2
            array.markers.append(body)

            label = Marker()
            label.header = obstacles.header
            label.header.frame_id = 'map'
            label.ns = 'tracked_obstacle_labels'
            label.id = obstacle.id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(obstacle.x_m)
            label.pose.position.y = float(obstacle.y_m)
            label.pose.position.z = 0.5
            label.pose.orientation.w = 1.0
            label.scale.z = 0.14
            label.color.a = 1.0
            label.color.r = label.color.g = label.color.b = 1.0
            label.text = (
                f'id={obstacle.id} '
                f's={obstacle.s_center:.2f} d={obstacle.d_center:+.2f} '
                f'w={lateral:.2f} v={obstacle.vs:.2f}'
                f'{"" if obstacle.is_visible else " (coasting)"}')
            array.markers.append(label)
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
