#!/usr/bin/env python3
"""LiDAR clustering and Frenet obstacle extraction."""

import math
import time

import numpy as np
import rclpy
from f110_msgs.msg import Obstacle, ObstacleArray, WpntArray
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from frenet_conversion.frenet_converter import FrenetConverter


def _rotate(points, quaternion):
    """Rotate Nx3 points by a ROS quaternion without an extra dependency."""
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    matrix = np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    return points @ matrix.T


class DetectNode(Node):
    def __init__(self):
        super().__init__('detect')
        for name, default in (
            ('cluster_distance', 0.25), ('min_points', 3),
            ('min_size', 0.04), ('max_size', 0.80),
            ('max_viewing_distance', 9.0), ('boundary_inflation', 0.10),
            ('measure', False),
        ):
            self.declare_parameter(name, default)

        self.converter = None
        self.waypoints = None
        self.track_length = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(WpntArray, '/global_waypoints_scaled', self._path_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
        self.obstacle_pub = self.create_publisher(ObstacleArray, '/detect/raw_obstacles', 5)
        self.marker_pub = self.create_publisher(MarkerArray, '/detect/obstacles_markers_new', 5)
        self.breakpoint_pub = self.create_publisher(MarkerArray, '/detect/breakpoints_markers', 5)
        self.latency_pub = self.create_publisher(Float32, '/detect/latency', 5)
        self._warned_tf = False
        self.get_logger().info('Waiting for /global_waypoints_scaled and /scan')

    def _path_cb(self, msg):
        if not msg.wpnts:
            return
        self.waypoints = msg.wpnts
        self.track_length = msg.wpnts[-1].s_m
        self.converter = FrenetConverter(
            np.asarray([w.x_m for w in msg.wpnts]),
            np.asarray([w.y_m for w in msg.wpnts]),
            np.asarray([w.psi_rad for w in msg.wpnts]),
        )

    def _scan_cb(self, scan):
        if self.converter is None:
            return
        started = time.perf_counter()
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', scan.header.frame_id or 'laser', rclpy.time.Time(),
                timeout=Duration(seconds=0.05))
        except TransformException as exc:
            if not self._warned_tf:
                self.get_logger().warning(f'Cannot transform scan into map: {exc}')
                self._warned_tf = True
            return
        self._warned_tf = False

        ranges = np.asarray(scan.ranges, dtype=float)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        valid = np.isfinite(ranges)
        valid &= ranges >= max(scan.range_min, 0.02)
        valid &= ranges <= min(scan.range_max, float(self.get_parameter('max_viewing_distance').value))
        local = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles), np.zeros_like(ranges)))
        rotated = _rotate(local, transform.transform.rotation)
        translation = transform.transform.translation
        world = rotated + np.asarray([translation.x, translation.y, translation.z])

        clusters = self._clusters(world[:, :2], valid)
        output = ObstacleArray()
        output.header.stamp = scan.header.stamp
        output.header.frame_id = 'map'
        markers = MarkerArray()
        obstacle_id = 0
        for cluster in clusters:
            obstacle = self._to_obstacle(cluster, obstacle_id)
            if obstacle is None:
                continue
            output.obstacles.append(obstacle)
            markers.markers.append(self._marker(cluster, obstacle_id, scan.header.stamp))
            obstacle_id += 1

        if not markers.markers:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = scan.header.stamp
            marker.action = Marker.DELETEALL
            markers.markers.append(marker)
        self.obstacle_pub.publish(output)
        self.marker_pub.publish(markers)
        if self.get_parameter('measure').value:
            self.latency_pub.publish(Float32(data=float(time.perf_counter() - started)))

    def _clusters(self, points, valid):
        threshold = float(self.get_parameter('cluster_distance').value)
        min_points = int(self.get_parameter('min_points').value)
        clusters, current = [], []
        previous = None
        for point, is_valid in zip(points, valid):
            if not is_valid:
                if len(current) >= min_points:
                    clusters.append(np.asarray(current))
                current, previous = [], None
                continue
            if previous is not None and np.linalg.norm(point - previous) > threshold:
                if len(current) >= min_points:
                    clusters.append(np.asarray(current))
                current = []
            current.append(point)
            previous = point
        if len(current) >= min_points:
            clusters.append(np.asarray(current))
        return clusters

    def _to_obstacle(self, cluster, obstacle_id):
        center = np.mean(cluster, axis=0)
        size = float(np.max(np.linalg.norm(cluster - center, axis=1)) * 2.0)
        if not float(self.get_parameter('min_size').value) <= size <= float(self.get_parameter('max_size').value):
            return None
        frenet = self.converter.get_frenet(cluster[:, 0], cluster[:, 1])
        s_values, d_values = np.asarray(frenet[0]), np.asarray(frenet[1])
        center_sd = self.converter.get_frenet(np.asarray([center[0]]), np.asarray([center[1]]))
        center_s, center_d = float(center_sd[0][0]), float(center_sd[1][0])
        waypoint_s = np.asarray([w.s_m for w in self.waypoints])
        ref = self.waypoints[int(np.argmin(np.abs(waypoint_s - center_s)))]
        inflation = float(self.get_parameter('boundary_inflation').value)
        if center_d > ref.d_left - inflation or center_d < -ref.d_right + inflation:
            return None

        # Unwrap around the start/finish seam before finding longitudinal bounds.
        delta = (s_values - center_s + self.track_length / 2.0) % self.track_length - self.track_length / 2.0
        return Obstacle(
            id=obstacle_id,
            s_start=float((center_s + np.min(delta)) % self.track_length),
            s_end=float((center_s + np.max(delta)) % self.track_length),
            d_right=float(np.min(d_values)), d_left=float(np.max(d_values)),
            s_center=center_s, d_center=center_d, size=size,
            vs=0.0, vd=0.0, is_static=False, is_visible=True,
        )

    @staticmethod
    def _marker(cluster, marker_id, stamp):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp
        marker.ns = 'detected_obstacles'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        center = np.mean(cluster, axis=0)
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.orientation.w = 1.0
        diameter = max(0.05, float(np.max(np.linalg.norm(cluster - center, axis=1)) * 2.0))
        marker.scale.x = marker.scale.y = diameter
        marker.scale.z = 0.15
        marker.color.a = 0.8
        marker.color.r = 1.0
        marker.color.g = 0.4
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = DetectNode()
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
