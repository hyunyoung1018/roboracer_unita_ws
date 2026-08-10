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
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from frenet_conversion.frenet_converter import FrenetConverter


def _rotate(points, quaternion):
    """Rotate Nx3 points by a ROS quaternion without an extra dependency."""
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    matrix = np.asarray([
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ],
        [
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ],
        [
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
    ])

    return points @ matrix.T


class DetectNode(Node):

    def __init__(self):
        super().__init__('detect')

        # ============================================================
        # Parameters
        # ============================================================
        for name, default in (
            # 기존 detector의 최소 clustering 거리.
            # Adaptive threshold가 이 값보다 작아지지 않도록 한다.
            ('cluster_distance', 0.25),

            # Adaptive clustering
            ('lambda_deg', 10.0),
            ('sigma', 0.03),
            ('min_2_points_dist', 0.01),
            ('new_cluster_threshold', 0.40),

            # 기존 detector와 동일하게 최소 3 point
            ('min_points', 3),

            # Cluster size filtering
            ('min_size', 0.04),
            ('max_size', 0.80),

            # Detection range
            ('max_viewing_distance', 9.0),

            # Track boundary
            ('boundary_inflation', 0.10),

            # 현재는 기존 detector의 center-only filtering과
            # 최대한 동일하게 사용한다.
            #
            # 이후 실차 검증 후:
            # 0.0 -> 0.3 -> 0.5 -> 0.7 순으로 올려볼 수 있다.
            ('min_inside_ratio', 0.0),

            # Profiling
            ('measure', False),
        ):
            self.declare_parameter(name, default)

        # ============================================================
        # Frenet / global path
        # ============================================================
        self.converter = None
        self.waypoints = None
        self._waypoint_s = None
        self.track_length = None

        # ============================================================
        # TF
        # ============================================================
        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=5.0)
        )

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        # ============================================================
        # Subscribers
        # ============================================================
        self.create_subscription(
            WpntArray,
            '/global_waypoints_scaled',
            self._path_cb,
            10,
        )

        self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_cb,
            qos_profile_sensor_data,
        )

        # ============================================================
        # Publishers
        # ============================================================
        self.obstacle_pub = self.create_publisher(
            ObstacleArray,
            '/detect/raw_obstacles',
            5,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/detect/obstacles_markers_new',
            5,
        )

        # 기존 perception topic 구조와 호환성을 위해 유지
        self.breakpoint_pub = self.create_publisher(
            MarkerArray,
            '/detect/breakpoints_markers',
            5,
        )

        self.latency_pub = self.create_publisher(
            Float32,
            '/detect/latency',
            5,
        )

        # ============================================================
        # Internal states
        # ============================================================
        self._warned_tf = False
        self._last_marker_count = 0

        self.get_logger().info(
            'Waiting for /global_waypoints_scaled and /scan'
        )

    # ================================================================
    # Global path callback
    # ================================================================

    def _path_cb(self, msg):
        """Build/update FrenetConverter from global waypoints."""

        if not msg.wpnts:
            return

        self.waypoints = msg.wpnts

        self._waypoint_s = np.asarray(
            [wp.s_m for wp in msg.wpnts],
            dtype=float,
        )

        self.track_length = float(
            msg.wpnts[-1].s_m
        )

        if self.track_length <= 0.0:
            self.get_logger().warning(
                'Invalid track length from /global_waypoints_scaled'
            )
            self.converter = None
            return

        self.converter = FrenetConverter(
            np.asarray(
                [wp.x_m for wp in msg.wpnts],
                dtype=float,
            ),
            np.asarray(
                [wp.y_m for wp in msg.wpnts],
                dtype=float,
            ),
            np.asarray(
                [wp.psi_rad for wp in msg.wpnts],
                dtype=float,
            ),
        )

    # ================================================================
    # LaserScan callback
    # ================================================================

    def _scan_cb(self, scan):
        """
        LaserScan
            -> map-frame points
            -> adaptive clustering
            -> Frenet obstacle
            -> ObstacleArray
        """

        if self.converter is None:
            return

        started = time.perf_counter()

        # ------------------------------------------------------------
        # LiDAR frame
        # ------------------------------------------------------------
        frame_id = (
            scan.header.frame_id
            or 'ego_racecar/laser'
        )

        # ------------------------------------------------------------
        # TF at the scan's own timestamp, falling back to the latest
        # ------------------------------------------------------------
        # Asking for the transform AT the scan's stamp is the correct thing:
        # it compensates for the car having moved between the sweep and now.
        # But it only succeeds while tf holds data bracketing that instant, and
        # on this jetson - scan at 40 Hz, the ekf already missing its own
        # deadlines - it frequently does not inside a 50 ms timeout. Returning
        # in that case stops detection completely, silently, because the
        # warning latches after one print. That is what "no obstacles at all"
        # was.
        #
        # So: try the stamp, and if tf cannot serve it, take the latest
        # transform instead. At the speed this branch runs, the difference is a
        # couple of centimetres of lateral error - against not detecting.
        scan_time = Time.from_msg(scan.header.stamp)
        if scan_time.nanoseconds == 0:
            scan_time = Time()

        transform = None
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', frame_id, scan_time, timeout=Duration(seconds=0.05))
            self._warned_tf = False
        except TransformException as stamped_exc:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'map', frame_id, Time(), timeout=Duration(seconds=0.05))
                self.get_logger().warning(
                    f'no tf at the scan stamp ({stamped_exc}); using the latest',
                    throttle_duration_sec=5.0)
                self._warned_tf = False
            except TransformException as latest_exc:
                # Throttled, not latched: before this it warned once and then
                # went quiet for good, so a permanent failure looked the same
                # as a single startup hiccup.
                self.get_logger().warning(
                    f'Cannot transform {frame_id} into map: {latest_exc}',
                    throttle_duration_sec=5.0)
                return

        # ------------------------------------------------------------
        # LaserScan -> numpy
        # ------------------------------------------------------------
        ranges = np.asarray(
            scan.ranges,
            dtype=float,
        )

        if ranges.size == 0:
            return

        angles = (
            scan.angle_min
            + np.arange(
                len(ranges),
                dtype=float,
            ) * scan.angle_increment
        )

        max_viewing_distance = float(
            self.get_parameter(
                'max_viewing_distance'
            ).value
        )

        # ------------------------------------------------------------
        # Valid LiDAR points
        # ------------------------------------------------------------
        valid = np.isfinite(ranges)

        valid &= ranges >= max(
            float(scan.range_min),
            0.02,
        )

        valid &= ranges <= min(
            float(scan.range_max),
            max_viewing_distance,
        )

        # ------------------------------------------------------------
        # Laser frame Cartesian coordinates
        # ------------------------------------------------------------
        local = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles),
            np.zeros_like(ranges),
        ))

        # ------------------------------------------------------------
        # Laser frame -> map frame
        # ------------------------------------------------------------
        rotated = _rotate(
            local,
            transform.transform.rotation,
        )

        translation = transform.transform.translation

        world = rotated + np.asarray([
            translation.x,
            translation.y,
            translation.z,
        ])

        # ------------------------------------------------------------
        # Adaptive clustering
        # ------------------------------------------------------------
        clusters = self._clusters(
            world[:, :2],
            ranges,
            valid,
            scan.angle_increment,
        )

        # ------------------------------------------------------------
        # Obstacle output
        # ------------------------------------------------------------
        output = ObstacleArray()

        output.header.stamp = scan.header.stamp
        output.header.frame_id = 'map'

        markers = MarkerArray()

        obstacle_id = 0

        for cluster in clusters:

            obstacle = self._to_obstacle(
                cluster,
                obstacle_id,
            )

            if obstacle is None:
                continue

            output.obstacles.append(
                obstacle
            )

            markers.markers.append(
                self._marker(
                    cluster,
                    obstacle_id,
                    scan.header.stamp,
                )
            )

            obstacle_id += 1

        # ------------------------------------------------------------
        # Delete stale RViz markers
        # ------------------------------------------------------------
        for stale_id in range(
            obstacle_id,
            self._last_marker_count,
        ):
            marker = Marker()

            marker.header.frame_id = 'map'
            marker.header.stamp = scan.header.stamp

            marker.ns = 'detected_obstacles'
            marker.id = stale_id

            marker.action = Marker.DELETE

            markers.markers.append(
                marker
            )

        self._last_marker_count = obstacle_id

        # ------------------------------------------------------------
        # Publish
        # ------------------------------------------------------------
        self.obstacle_pub.publish(
            output
        )

        self.marker_pub.publish(
            markers
        )

        # ------------------------------------------------------------
        # Latency
        # ------------------------------------------------------------
        if self.get_parameter(
            'measure'
        ).value:

            latency = (
                time.perf_counter()
                - started
            )

            self.latency_pub.publish(
                Float32(
                    data=float(latency)
                )
            )

    # ================================================================
    # Adaptive clustering
    # ================================================================

    def _clusters(
        self,
        points,
        ranges,
        valid,
        angle_increment,
    ):
        """
        Range-adaptive sequential LiDAR clustering.

        핵심 원칙:

        가까운 거리:
            기존 cluster_distance = 0.25 m 유지

        먼 거리:
            LiDAR angular resolution 때문에 점 간격이 커지는 경우
            adaptive threshold를 사용해 0.25 m보다 증가시킨다.

        따라서 adaptive threshold가 절대로 기존 0.25 m보다
        작아지지 않는다.
        """

        min_points = int(
            self.get_parameter(
                'min_points'
            ).value
        )

        lambda_angle = math.radians(
            float(
                self.get_parameter(
                    'lambda_deg'
                ).value
            )
        )

        sigma = float(
            self.get_parameter(
                'sigma'
            ).value
        )

        min_dist = float(
            self.get_parameter(
                'min_2_points_dist'
            ).value
        )

        max_cluster_gap = float(
            self.get_parameter(
                'new_cluster_threshold'
            ).value
        )

        fallback_threshold = float(
            self.get_parameter(
                'cluster_distance'
            ).value
        )

        d_phi = abs(
            float(angle_increment)
        )

        clusters = []
        current = []

        previous = None

        for i, (point, is_valid) in enumerate(
            zip(points, valid)
        ):

            # --------------------------------------------------------
            # Invalid beam
            # --------------------------------------------------------
            if not is_valid:

                if len(current) >= min_points:
                    clusters.append(
                        np.asarray(current)
                    )

                current = []
                previous = None

                continue

            # --------------------------------------------------------
            # First valid point
            # --------------------------------------------------------
            if previous is None:

                current.append(point)
                previous = point

                continue

            # --------------------------------------------------------
            # Adaptive threshold
            # --------------------------------------------------------
            adaptive_threshold = (
                fallback_threshold
            )

            if (
                d_phi > 0.0
                and lambda_angle > d_phi
            ):

                denominator = math.sin(
                    lambda_angle - d_phi
                )

                if abs(denominator) > 1e-9:

                    adaptive_threshold = (
                        float(ranges[i])
                        * math.sin(d_phi)
                        / denominator
                        + 3.0 * sigma
                    )

            # ========================================================
            # IMPORTANT
            #
            # 기존 detector의 cluster_distance보다
            # adaptive threshold가 절대로 작아지지 않도록 한다.
            #
            # 예:
            #
            # adaptive = 0.11 -> 실제 0.25
            # adaptive = 0.16 -> 실제 0.25
            # adaptive = 0.22 -> 실제 0.25
            # adaptive = 0.27 -> 실제 0.27
            # adaptive = 0.32 -> 실제 0.32
            #
            # ========================================================
            adaptive_threshold = max(
                adaptive_threshold,
                fallback_threshold,
                min_dist,
            )

            # --------------------------------------------------------
            # 너무 커져 서로 다른 물체가 합쳐지는 것을 방지
            # --------------------------------------------------------
            adaptive_threshold = min(
                adaptive_threshold,
                max_cluster_gap,
            )

            # --------------------------------------------------------
            # Distance between consecutive LiDAR points
            # --------------------------------------------------------
            distance = float(
                np.linalg.norm(
                    point - previous
                )
            )

            # --------------------------------------------------------
            # Breakpoint
            # --------------------------------------------------------
            if distance > adaptive_threshold:

                if len(current) >= min_points:
                    clusters.append(
                        np.asarray(current)
                    )

                current = []

            current.append(point)
            previous = point

        # ------------------------------------------------------------
        # Final cluster
        # ------------------------------------------------------------
        if len(current) >= min_points:

            clusters.append(
                np.asarray(current)
            )

        return clusters

    # ================================================================
    # Nearest global waypoint
    # ================================================================

    def _nearest_waypoint_index(
        self,
        s_value,
    ):
        """
        Find nearest waypoint in Frenet s,
        including start/finish seam wrapping.
        """

        delta = np.abs(
            (
                self._waypoint_s
                - float(s_value)
                + self.track_length / 2.0
            )
            % self.track_length
            - self.track_length / 2.0
        )

        return int(
            np.argmin(delta)
        )

    # ================================================================
    # Cluster -> Obstacle
    # ================================================================

    def _to_obstacle(
        self,
        cluster,
        obstacle_id,
    ):
        """Convert a map-frame LiDAR cluster to a Frenet obstacle."""

        if (
            self.converter is None
            or self.track_length is None
            or self.track_length <= 0.0
            or self._waypoint_s is None
            or self.waypoints is None
        ):
            return None

        if cluster.size == 0:
            return None

        # ------------------------------------------------------------
        # Cluster center
        # ------------------------------------------------------------
        center = np.mean(
            cluster,
            axis=0,
        )

        # ------------------------------------------------------------
        # Cluster size
        # ------------------------------------------------------------
        distance_from_center = np.linalg.norm(
            cluster - center,
            axis=1,
        )

        size = float(
            np.max(
                distance_from_center
            ) * 2.0
        )

        min_size = float(
            self.get_parameter(
                'min_size'
            ).value
        )

        max_size = float(
            self.get_parameter(
                'max_size'
            ).value
        )

        if (
            size < min_size
            or size > max_size
        ):
            return None

        # ------------------------------------------------------------
        # Entire cluster -> Frenet
        # ------------------------------------------------------------
        frenet = self.converter.get_frenet(
            cluster[:, 0],
            cluster[:, 1],
        )

        s_values = np.asarray(
            frenet[0],
            dtype=float,
        )

        d_values = np.asarray(
            frenet[1],
            dtype=float,
        )

        if (
            not np.all(
                np.isfinite(s_values)
            )
            or not np.all(
                np.isfinite(d_values)
            )
        ):
            return None

        # ------------------------------------------------------------
        # Center -> Frenet
        # ------------------------------------------------------------
        center_sd = self.converter.get_frenet(
            np.asarray(
                [center[0]],
                dtype=float,
            ),
            np.asarray(
                [center[1]],
                dtype=float,
            ),
        )

        center_s = float(
            center_sd[0][0]
        )

        center_d = float(
            center_sd[1][0]
        )

        if (
            not math.isfinite(center_s)
            or not math.isfinite(center_d)
        ):
            return None

        # ------------------------------------------------------------
        # Normalize Frenet s
        # ------------------------------------------------------------
        center_s %= self.track_length

        s_values = np.mod(
            s_values,
            self.track_length,
        )

        # ------------------------------------------------------------
        # Center boundary check
        # ------------------------------------------------------------
        center_index = (
            self._nearest_waypoint_index(
                center_s
            )
        )

        ref = self.waypoints[
            center_index
        ]

        inflation = float(
            self.get_parameter(
                'boundary_inflation'
            ).value
        )

        left_limit = (
            float(ref.d_left)
            - inflation
        )

        right_limit = (
            -float(ref.d_right)
            + inflation
        )

        # 잘못된 waypoint boundary
        if left_limit <= right_limit:
            return None

        # ------------------------------------------------------------
        # 기존 detector와 동일한 핵심 boundary filter
        #
        # 장애물 CENTER가 track 안에 있어야 한다.
        # ------------------------------------------------------------
        if (
            center_d > left_limit
            or center_d < right_limit
        ):
            return None

        # ------------------------------------------------------------
        # Entire cluster inside-ratio check
        # ------------------------------------------------------------
        #
        # 현재 default:
        #
        # min_inside_ratio = 0.0
        #
        # 따라서 실질적으로 기존 center-only detector와 동일하게
        # 동작한다.
        #
        # 추후 필요 시 0.3 -> 0.5 -> 0.7 순으로 강화 가능.
        # ------------------------------------------------------------
        min_inside_ratio = float(
            self.get_parameter(
                'min_inside_ratio'
            ).value
        )

        if min_inside_ratio > 0.0:

            inside_count = 0

            for (
                s_value,
                d_value,
            ) in zip(
                s_values,
                d_values,
            ):

                waypoint_index = (
                    self._nearest_waypoint_index(
                        s_value
                    )
                )

                wp = self.waypoints[
                    waypoint_index
                ]

                point_left_limit = (
                    float(wp.d_left)
                    - inflation
                )

                point_right_limit = (
                    -float(wp.d_right)
                    + inflation
                )

                if (
                    point_left_limit
                    <= point_right_limit
                ):
                    continue

                if (
                    point_right_limit
                    <= d_value
                    <= point_left_limit
                ):
                    inside_count += 1

            inside_ratio = (
                inside_count
                / max(
                    len(cluster),
                    1,
                )
            )

            if (
                inside_ratio
                < min_inside_ratio
            ):
                return None

        # ------------------------------------------------------------
        # Start / finish seam handling
        # ------------------------------------------------------------
        longitudinal_delta = (
            s_values
            - center_s
            + self.track_length / 2.0
        ) % self.track_length \
            - self.track_length / 2.0

        s_start = float(
            (
                center_s
                + np.min(
                    longitudinal_delta
                )
            )
            % self.track_length
        )

        s_end = float(
            (
                center_s
                + np.max(
                    longitudinal_delta
                )
            )
            % self.track_length
        )

        # ------------------------------------------------------------
        # Obstacle output
        #
        # Detection에서는 위치/크기만 계산.
        # 속도와 static/dynamic 판정은 tracking_node 담당.
        # ------------------------------------------------------------
        return Obstacle(
            id=obstacle_id,

            s_start=s_start,
            s_end=s_end,

            d_right=float(
                np.min(d_values)
            ),

            d_left=float(
                np.max(d_values)
            ),

            s_center=center_s,
            d_center=center_d,

            size=size,

            vs=0.0,
            vd=0.0,

            is_static=False,
            is_visible=True,
        )

    # ================================================================
    # RViz marker
    # ================================================================

    @staticmethod
    def _marker(
        cluster,
        marker_id,
        stamp,
    ):
        """Create RViz marker for detected obstacle."""

        marker = Marker()

        marker.header.frame_id = 'map'
        marker.header.stamp = stamp

        marker.ns = 'detected_obstacles'
        marker.id = marker_id

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        center = np.mean(
            cluster,
            axis=0,
        )

        marker.pose.position.x = float(
            center[0]
        )

        marker.pose.position.y = float(
            center[1]
        )

        marker.pose.position.z = 0.0

        marker.pose.orientation.w = 1.0

        diameter = max(
            0.05,
            float(
                np.max(
                    np.linalg.norm(
                        cluster - center,
                        axis=1,
                    )
                ) * 2.0
            ),
        )

        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = 0.15

        marker.color.a = 0.8
        marker.color.r = 1.0
        marker.color.g = 0.4
        marker.color.b = 0.0

        return marker


def main(args=None):
    rclpy.init(args=args)

    node = DetectNode()

    try:
        rclpy.spin(node)

    except (
        KeyboardInterrupt,
        ExternalShutdownException,
    ):
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()