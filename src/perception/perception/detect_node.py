#!/usr/bin/env python3
"""UNITA LiDAR obstacle detector with UNIST-style clustering/L-shape fitting."""

import math
import time

import numpy as np
import rclpy
from f110_msgs.msg import Obstacle, ObstacleArray, WpntArray
from frenet_conversion.frenet_converter import FrenetConverter
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def _rotate(points, quaternion):
    """Rotate Nx3 points by a ROS quaternion."""
    x, y, z, w = (
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    )

    matrix = np.asarray([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ], dtype=float)

    return points @ matrix.T


class DetectNode(Node):

    def __init__(self):
        super().__init__('detect')

        # ============================================================
        # Parameters
        #
        # UNIST detector parameters applicable without GridFilter.
        #
        # GridFilter 관련:
        #   filter_kernel_size
        #   map
        #
        # 등은 의도적으로 사용하지 않는다.
        # ============================================================
        for name, default in (
            ('rate_detect', 40.0),

            ('min_size_n', 4),

            ('min_size_m', 0.30),
            ('max_size_m', 0.60),

            ('lambda_deg', 10.0),
            ('sigma', 0.03),

            ('min_2_points_dist', 0.01),

            ('new_cluster_threshold_m', 0.40),

            ('max_viewing_distance', 9.0),

            ('boundaries_inflation', 0.10),

            # --------------------------------------------------------
            # UNITA-specific parameters
            # --------------------------------------------------------

            # Hokuyo 전체 /scan은 localization을 위해 그대로 유지하고,
            # detection에서만 차량 뒤쪽을 제외한다.
            ('detect_fov_deg', 220.0),

            # GridFilter를 사용하지 않으므로 기존 UNITA의
            # Frenet boundary filtering을 유지한다.
            #
            # 0.0:
            #   fitted obstacle center만 track 안에 있으면 인정.
            #
            # 필요하면 나중에:
            #   0.3 -> 0.5
            # 정도로 올릴 수 있다.
            ('min_inside_ratio', 0.0),

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
        # Latest LaserScan
        #
        # UNIST처럼 rate_detect timer에서 detection을 실행한다.
        # 단, 동일 LaserScan을 두 번 처리하지 않는다.
        # ============================================================
        self._latest_scan = None
        self._last_processed_scan_key = None

        # ============================================================
        # RViz marker state
        # ============================================================
        self._last_marker_count = 0
        self._last_breakpoint_marker_count = 0

        # ============================================================
        # TF
        #
        # 이 부분은 UNIST의 hard-coded "laser" 방식을 사용하지 않고
        # 현재 UNITA 방식을 그대로 유지한다.
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
        #
        # 기존 UNITA topic 모두 유지.
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
        # Detection timer
        # ============================================================
        rate_detect = float(
            self.get_parameter(
                'rate_detect'
            ).value
        )

        if rate_detect <= 0.0:
            self.get_logger().warning(
                'rate_detect must be > 0; using 40 Hz'
            )
            rate_detect = 40.0

        self.detect_timer = self.create_timer(
            1.0 / rate_detect,
            self._timer_cb,
        )

        self.get_logger().info(
            'Waiting for /global_waypoints_scaled and /scan '
            f'(detect rate: {rate_detect:.1f} Hz)'
        )

    # ================================================================
    # Global path
    # ================================================================

    def _path_cb(self, msg):

        if not msg.wpnts:
            return

        self.waypoints = msg.wpnts

        self._waypoint_s = np.asarray(
            [
                wp.s_m
                for wp in msg.wpnts
            ],
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
                [
                    wp.x_m
                    for wp in msg.wpnts
                ],
                dtype=float,
            ),
            np.asarray(
                [
                    wp.y_m
                    for wp in msg.wpnts
                ],
                dtype=float,
            ),
            np.asarray(
                [
                    wp.psi_rad
                    for wp in msg.wpnts
                ],
                dtype=float,
            ),
        )

    # ================================================================
    # LaserScan
    # ================================================================

    def _scan_cb(self, scan):
        """
        가장 최근 LaserScan만 저장.

        실제 detection은 rate_detect timer에서 실행한다.
        """

        self._latest_scan = scan

    # ================================================================
    # Detection timer
    # ================================================================

    def _timer_cb(self):

        if (
            self.converter is None
            or self._latest_scan is None
        ):
            return

        scan = self._latest_scan

        sec = int(
            scan.header.stamp.sec
        )

        nanosec = int(
            scan.header.stamp.nanosec
        )

        if (
            sec != 0
            or nanosec != 0
        ):
            scan_key = (
                sec,
                nanosec,
            )

        else:
            scan_key = id(
                scan
            )

        # 동일 LaserScan 재처리 방지
        if (
            scan_key
            == self._last_processed_scan_key
        ):
            return

        self._last_processed_scan_key = (
            scan_key
        )

        self._process_scan(
            scan
        )

    # ================================================================
    # Main detection pipeline
    # ================================================================

    def _process_scan(self, scan):

        started = time.perf_counter()

        # ------------------------------------------------------------
        # Laser frame
        #
        # UNITA 유지:
        #
        # 1순위:
        #   실제 scan.header.frame_id
        #
        # fallback:
        #   ego_racecar/laser
        #
        # UNIST의 hard-coded "laser"는 사용하지 않는다.
        # ------------------------------------------------------------
        frame_id = (
            scan.header.frame_id
            or 'ego_racecar/laser'
        )

        # ------------------------------------------------------------
        # TF
        #
        # UNITA의 기존 로직 유지.
        #
        # scan timestamp 기준 TF를 먼저 요청하고,
        # 실패하는 경우 latest TF를 사용한다.
        # ------------------------------------------------------------
        scan_time = Time.from_msg(
            scan.header.stamp
        )

        if scan_time.nanoseconds == 0:
            scan_time = Time()

        try:

            transform = (
                self.tf_buffer.lookup_transform(
                    'map',
                    frame_id,
                    scan_time,
                    timeout=Duration(
                        seconds=0.05
                    ),
                )
            )

        except TransformException as stamped_exc:

            try:

                transform = (
                    self.tf_buffer.lookup_transform(
                        'map',
                        frame_id,
                        Time(),
                        timeout=Duration(
                            seconds=0.05
                        ),
                    )
                )

                self.get_logger().warning(
                    f'no tf at the scan stamp '
                    f'({stamped_exc}); using the latest',
                    throttle_duration_sec=5.0,
                )

            except TransformException as latest_exc:

                self.get_logger().warning(
                    f'Cannot transform '
                    f'{frame_id} into map: '
                    f'{latest_exc}',
                    throttle_duration_sec=5.0,
                )

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
            float(
                scan.angle_min
            )
            + np.arange(
                ranges.size,
                dtype=float,
            )
            * float(
                scan.angle_increment
            )
        )

        # ------------------------------------------------------------
        # Valid points
        # ------------------------------------------------------------
        max_viewing_distance = float(
            self.get_parameter(
                'max_viewing_distance'
            ).value
        )

        valid = np.isfinite(
            ranges
        )

        valid &= (
            ranges
            >= max(
                float(
                    scan.range_min
                ),
                0.02,
            )
        )

        valid &= (
            ranges
            <= min(
                float(
                    scan.range_max
                ),
                max_viewing_distance,
            )
        )

        # ------------------------------------------------------------
        # UNITA detection FOV 유지
        #
        # /scan 자체를 줄이는 것이 아니라 detector 내부에서만
        # 차량 뒤쪽 point를 제외한다.
        # ------------------------------------------------------------
        fov = float(
            self.get_parameter(
                'detect_fov_deg'
            ).value
        )

        if (
            0.0
            < fov
            < 360.0
        ):
            valid &= (
                np.abs(
                    angles
                )
                <= math.radians(
                    fov
                )
                / 2.0
            )

        # invalid beam에서 inf / nan 연산 방지
        safe_ranges = np.where(
            valid,
            ranges,
            0.0,
        )

        # ------------------------------------------------------------
        # Laser coordinates
        # ------------------------------------------------------------
        local = np.column_stack((
            safe_ranges
            * np.cos(
                angles
            ),

            safe_ranges
            * np.sin(
                angles
            ),

            np.zeros_like(
                safe_ranges
            ),
        ))

        # ------------------------------------------------------------
        # Laser -> map
        # ------------------------------------------------------------
        rotated = _rotate(
            local,
            transform.transform.rotation,
        )

        translation = (
            transform.transform.translation
        )

        world = (
            rotated
            + np.asarray(
                [
                    translation.x,
                    translation.y,
                    translation.z,
                ],
                dtype=float,
            )
        )

        # L-shape fitting에서
        # LiDAR와 가장 가까운 rectangle corner를 찾기 위해 사용.
        sensor_xy = np.asarray(
            [
                translation.x,
                translation.y,
            ],
            dtype=float,
        )

        # ------------------------------------------------------------
        # UNIST adaptive clustering
        # ------------------------------------------------------------
        clusters = self._clusters(
            world[:, :2],
            ranges,
            valid,
            scan.angle_increment,
        )

        # ------------------------------------------------------------
        # Breakpoint visualization
        # ------------------------------------------------------------
        self.breakpoint_pub.publish(
            self._breakpoint_markers(
                clusters,
                scan.header.stamp,
            )
        )

        # ------------------------------------------------------------
        # Obstacle output
        # ------------------------------------------------------------
        output = ObstacleArray()

        output.header.stamp = (
            scan.header.stamp
        )

        output.header.frame_id = (
            'map'
        )

        markers = MarkerArray()

        obstacle_id = 0

        # ------------------------------------------------------------
        # Cluster -> L-shape -> Frenet obstacle
        # ------------------------------------------------------------
        for cluster in clusters:

            fitted = self._fit_l_shape(
                cluster,
                sensor_xy,
            )

            if fitted is None:
                continue

            center, size, theta = (
                fitted
            )

            obstacle = self._to_obstacle(
                cluster,
                center,
                size,
                obstacle_id,
            )

            if obstacle is None:
                continue

            output.obstacles.append(
                obstacle
            )

            markers.markers.append(
                self._obstacle_marker(
                    center,
                    size,
                    theta,
                    obstacle_id,
                    scan.header.stamp,
                )
            )

            obstacle_id += 1

        # ------------------------------------------------------------
        # Delete stale obstacle markers
        # ------------------------------------------------------------
        for stale_id in range(
            obstacle_id,
            self._last_marker_count,
        ):

            marker = Marker()

            marker.header.frame_id = (
                'map'
            )

            marker.header.stamp = (
                scan.header.stamp
            )

            marker.ns = (
                'detected_obstacles'
            )

            marker.id = (
                stale_id
            )

            marker.action = (
                Marker.DELETE
            )

            markers.markers.append(
                marker
            )

        self._last_marker_count = (
            obstacle_id
        )

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
        # Detection latency [seconds]
        # ------------------------------------------------------------
        if bool(
            self.get_parameter(
                'measure'
            ).value
        ):

            self.latency_pub.publish(
                Float32(
                    data=float(
                        time.perf_counter()
                        - started
                    )
                )
            )

    # ================================================================
    # UNIST adaptive clustering
    # ================================================================

    def _clusters(
        self,
        points,
        ranges,
        valid,
        angle_increment,
    ):
        """
        UNIST adaptive breakpoint clustering.

        GridFilter만 제외하고 핵심 clustering 구조를 그대로 가져온다.

        Adaptive breakpoint:

            d_max
            =
            range
            * sin(d_phi)
            / sin(lambda - d_phi)
            + 3 * sigma

        기존 UNITA와 달리:

            cluster_distance = 0.25

        하한을 강제로 적용하지 않는다.

        또한 d_max를 넘었다고 바로 cluster를 종료하는 것이 아니라
        기존 모든 cluster의 마지막 point를 검사하고,

            new_cluster_threshold_m

        이내라면 해당 cluster에 재결합한다.
        """

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        min_size_n = max(
            int(
                self.get_parameter(
                    'min_size_n'
                ).value
            ),
            1,
        )

        lambda_angle = (
            math.radians(
                float(
                    self.get_parameter(
                        'lambda_deg'
                    ).value
                )
            )
        )

        sigma = float(
            self.get_parameter(
                'sigma'
            ).value
        )

        reconnect_threshold = float(
            self.get_parameter(
                'new_cluster_threshold_m'
            ).value
        )

        # ------------------------------------------------------------
        # LiDAR angular resolution
        # ------------------------------------------------------------
        d_phi = abs(
            float(
                angle_increment
            )
        )

        denominator = math.sin(
            lambda_angle
            - d_phi
        )

        if (
            d_phi <= 0.0
            or abs(
                denominator
            )
            <= 1e-12
        ):

            self.get_logger().warning(
                'Invalid adaptive clustering geometry: '
                f'lambda='
                f'{math.degrees(lambda_angle):.3f} deg, '
                f'd_phi='
                f'{math.degrees(d_phi):.3f} deg',
                throttle_duration_sec=5.0,
            )

            return []

        # ------------------------------------------------------------
        # Original UNIST formula constant
        # ------------------------------------------------------------
        div_const = (
            math.sin(
                d_phi
            )
            / denominator
        )

        clusters = []

        # ------------------------------------------------------------
        # Scan points sequentially
        # ------------------------------------------------------------
        for i in range(
            points.shape[0]
        ):

            # --------------------------------------------------------
            # Invalid / out-of-FOV beam
            #
            # IMPORTANT:
            #
            # 기존 UNITA:
            #
            #   invalid point
            #      ->
            #   cluster 종료
            #
            # 변경 후:
            #
            #   invalid point
            #      ->
            #   그냥 skip
            #
            # 따라서 다음 valid point가 기존 cluster에
            # reconnect될 수 있다.
            # --------------------------------------------------------
            if not valid[i]:
                continue

            current_point = np.asarray(
                points[i],
                dtype=float,
            )

            # --------------------------------------------------------
            # First cluster
            # --------------------------------------------------------
            if not clusters:

                clusters.append(
                    [
                        current_point
                    ]
                )

                continue

            current_range = float(
                ranges[i]
            )

            # ========================================================
            # UNIST ORIGINAL ADAPTIVE BREAKPOINT FORMULA
            #
            # cluster_distance=0.25 하한 없음.
            # 0.40 clamp 없음.
            # ========================================================
            d_max = (
                current_range
                * div_const
                + 3.0
                * sigma
            )

            # --------------------------------------------------------
            # Current point vs active cluster
            # --------------------------------------------------------
            distance_to_active = float(
                np.linalg.norm(
                    current_point
                    - clusters[-1][-1]
                )
            )

            if (
                distance_to_active
                < d_max
            ):

                clusters[-1].append(
                    current_point
                )

                continue

            # ========================================================
            # UNIST CLUSTER RECONNECTION
            #
            # active cluster와 연결되지 않았다면
            # 모든 기존 cluster의 마지막 point를 검색.
            # ========================================================
            min_distance = (
                math.inf
            )

            min_cluster_index = (
                -1
            )

            for (
                cluster_index,
                cluster,
            ) in enumerate(
                clusters
            ):

                distance = float(
                    np.linalg.norm(
                        current_point
                        - cluster[-1]
                    )
                )

                if (
                    distance
                    < min_distance
                ):

                    min_distance = (
                        distance
                    )

                    min_cluster_index = (
                        cluster_index
                    )

            # --------------------------------------------------------
            # Existing cluster reconnect
            # --------------------------------------------------------
            if (
                min_cluster_index
                >= 0
                and min_distance
                < reconnect_threshold
            ):

                # UNIST와 동일하게 reconnect 대상 cluster를
                # list 맨 뒤(active cluster 위치)로 이동한다.
                cluster_to_move = (
                    clusters.pop(
                        min_cluster_index
                    )
                )

                clusters.append(
                    cluster_to_move
                )

                clusters[-1].append(
                    current_point
                )

            # --------------------------------------------------------
            # Completely new cluster
            # --------------------------------------------------------
            else:

                clusters.append(
                    [
                        current_point
                    ]
                )

        # ============================================================
        # min_size_n filtering
        # ============================================================
        filtered = [
            np.asarray(
                cluster,
                dtype=float,
            )
            for cluster in clusters
            if len(
                cluster
            )
            >= min_size_n
        ]

        return filtered

    # ================================================================
    # UNIST L-shape fitting
    # ================================================================

    def _fit_l_shape(
        self,
        cluster,
        sensor_xy,
    ):
        """
        Python port of UNIST fittingLShape().

        0 ~ 89 deg 사이에서 90개 orientation candidate를 검사한다.

        각 orientation에서 point가 rectangle의 두 변 중
        어느 쪽에 더 가까운지를 이용하여 score를 계산한다.

        최종적으로:

            center
            size
            theta

        를 반환한다.
        """

        if (
            cluster is None
            or len(
                cluster
            )
            == 0
        ):
            return None

        points = np.asarray(
            cluster,
            dtype=float,
        )

        if (
            points.ndim != 2
            or points.shape[1] != 2
        ):
            return None

        if not np.all(
            np.isfinite(
                points
            )
        ):
            return None

        # ------------------------------------------------------------
        # Candidate orientations
        #
        # UNIST:
        # 0 deg ~ 89 deg
        # 90 candidates
        # ------------------------------------------------------------
        num_candidates = 90

        candidate_angles = (
            np.linspace(
                0.0,
                math.pi / 2.0
                - math.pi / 180.0,
                num_candidates,
            )
        )

        min_dist = max(
            float(
                self.get_parameter(
                    'min_2_points_dist'
                ).value
            ),
            1e-6,
        )

        best_score = (
            -math.inf
        )

        theta_opt = 0.0

        # ------------------------------------------------------------
        # Search optimal rectangle orientation
        # ------------------------------------------------------------
        for theta in candidate_angles:

            theta = float(
                theta
            )

            c = math.cos(
                theta
            )

            s = math.sin(
                theta
            )

            # --------------------------------------------------------
            # Projection onto rotated axes
            # --------------------------------------------------------
            proj1 = (
                points[:, 0]
                * c
                + points[:, 1]
                * s
            )

            proj2 = (
                -points[:, 0]
                * s
                + points[:, 1]
                * c
            )

            max1 = float(
                np.max(
                    proj1
                )
            )

            min1 = float(
                np.min(
                    proj1
                )
            )

            max2 = float(
                np.max(
                    proj2
                )
            )

            min2 = float(
                np.min(
                    proj2
                )
            )

            # --------------------------------------------------------
            # Distance to rectangle side 1
            # --------------------------------------------------------
            d10 = (
                -proj1
                + max1
            )

            d11 = (
                proj1
                - min1
            )

            if (
                np.linalg.norm(
                    d10
                )
                > np.linalg.norm(
                    d11
                )
            ):
                d1 = d11

            else:
                d1 = d10

            # --------------------------------------------------------
            # Distance to rectangle side 2
            # --------------------------------------------------------
            d20 = (
                -proj2
                + max2
            )

            d21 = (
                proj2
                - min2
            )

            if (
                np.linalg.norm(
                    d20
                )
                > np.linalg.norm(
                    d21
                )
            ):
                d2 = d21

            else:
                d2 = d20

            # --------------------------------------------------------
            # L-shape score
            # --------------------------------------------------------
            distances = np.minimum(
                d1,
                d2,
            )

            distances = np.maximum(
                distances,
                min_dist,
            )

            score = float(
                np.sum(
                    1.0
                    / distances
                )
            )

            if (
                score
                > best_score
            ):

                best_score = (
                    score
                )

                theta_opt = (
                    theta
                )

        # ============================================================
        # Recompute rectangle using optimal orientation
        # ============================================================
        c = math.cos(
            theta_opt
        )

        s = math.sin(
            theta_opt
        )

        dist1 = (
            points[:, 0]
            * c
            + points[:, 1]
            * s
        )

        dist2 = (
            -points[:, 0]
            * s
            + points[:, 1]
            * c
        )

        max1 = float(
            np.max(
                dist1
            )
        )

        min1 = float(
            np.min(
                dist1
            )
        )

        max2 = float(
            np.max(
                dist2
            )
        )

        min2 = float(
            np.min(
                dist2
            )
        )

        # ------------------------------------------------------------
        # Sensor position in rotated coordinates
        # ------------------------------------------------------------
        sensor_rot = np.asarray([
            (
                float(
                    sensor_xy[0]
                )
                * c
                + float(
                    sensor_xy[1]
                )
                * s
            ),
            (
                -float(
                    sensor_xy[0]
                )
                * s
                + float(
                    sensor_xy[1]
                )
                * c
            ),
        ])

        # ------------------------------------------------------------
        # Four rectangle corners
        # ------------------------------------------------------------
        corners = np.asarray([
            [
                max1,
                max2,
            ],
            [
                max1,
                min2,
            ],
            [
                min1,
                max2,
            ],
            [
                min1,
                min2,
            ],
        ], dtype=float)

        # ------------------------------------------------------------
        # Corner closest to LiDAR
        # ------------------------------------------------------------
        closest_index = int(
            np.argmin(
                np.linalg.norm(
                    corners
                    - sensor_rot,
                    axis=1,
                )
            )
        )

        chosen_corner = (
            corners[
                closest_index
            ].copy()
        )

        # ------------------------------------------------------------
        # Rectangle size
        # ------------------------------------------------------------
        width = (
            max1
            - min1
        )

        height = (
            max2
            - min2
        )

        rect_size = max(
            width,
            height,
        )

        min_size_m = float(
            self.get_parameter(
                'min_size_m'
            ).value
        )

        max_size_m = float(
            self.get_parameter(
                'max_size_m'
            ).value
        )

        # ------------------------------------------------------------
        # UNIST behavior
        #
        # 작은 cluster를 reject하지 않고 min_size_m까지 확장한다.
        # ------------------------------------------------------------
        rect_size = max(
            rect_size,
            min_size_m,
        )

        # ------------------------------------------------------------
        # UNIST checkObstacles()
        # ------------------------------------------------------------
        if (
            rect_size
            > max_size_m
        ):
            return None

        # ============================================================
        # Estimate object center
        #
        # UNIST와 동일하게 obstacle을 square로 가정한다.
        # ============================================================
        center_rot = (
            chosen_corner
        )

        half = (
            rect_size
            / 2.0
        )

        # corner 0 = UR
        if closest_index == 0:

            center_rot[0] -= half
            center_rot[1] -= half

        # corner 1 = LR
        elif closest_index == 1:

            center_rot[0] -= half
            center_rot[1] += half

        # corner 2 = UL
        elif closest_index == 2:

            center_rot[0] += half
            center_rot[1] -= half

        # corner 3 = LL
        else:

            center_rot[0] += half
            center_rot[1] += half

        # ------------------------------------------------------------
        # Rotated frame -> map frame
        # ------------------------------------------------------------
        center = np.asarray([
            (
                c
                * center_rot[0]
                - s
                * center_rot[1]
            ),
            (
                s
                * center_rot[0]
                + c
                * center_rot[1]
            ),
        ], dtype=float)

        if not np.all(
            np.isfinite(
                center
            )
        ):
            return None

        return (
            center,
            float(
                rect_size
            ),
            float(
                theta_opt
            ),
        )

    # ================================================================
    # Nearest waypoint
    # ================================================================

    def _nearest_waypoint_index(
        self,
        s_value,
    ):

        delta = np.abs(
            (
                self._waypoint_s
                - float(
                    s_value
                )
                + self.track_length
                / 2.0
            )
            % self.track_length
            - self.track_length
            / 2.0
        )

        return int(
            np.argmin(
                delta
            )
        )

    # ================================================================
    # Fitted obstacle -> f110_msgs/Obstacle
    # ================================================================

    def _to_obstacle(
        self,
        cluster,
        fitted_center,
        fitted_size,
        obstacle_id,
    ):
        """
        기존 UNITA의 Frenet boundary filter와
        f110_msgs/Obstacle interface를 유지한다.
        """

        if (
            self.converter is None
            or self.track_length is None
            or self.track_length <= 0.0
            or self._waypoint_s is None
            or self.waypoints is None
        ):
            return None

        if (
            cluster is None
            or len(
                cluster
            )
            == 0
        ):
            return None

        center = np.asarray(
            fitted_center,
            dtype=float,
        )

        size = float(
            fitted_size
        )

        # ------------------------------------------------------------
        # Fitted center -> Frenet
        # ------------------------------------------------------------
        center_sd = (
            self.converter.get_frenet(
                np.asarray(
                    [
                        center[0]
                    ],
                    dtype=float,
                ),
                np.asarray(
                    [
                        center[1]
                    ],
                    dtype=float,
                ),
            )
        )

        center_s = float(
            center_sd[0][0]
        )

        center_d = float(
            center_sd[1][0]
        )

        if (
            not math.isfinite(
                center_s
            )
            or not math.isfinite(
                center_d
            )
        ):
            return None

        center_s %= (
            self.track_length
        )

        # ============================================================
        # UNITA Frenet boundary filter
        #
        # GridFilter를 제외했으므로 이 부분은 반드시 유지한다.
        # ============================================================
        center_index = (
            self._nearest_waypoint_index(
                center_s
            )
        )

        ref = (
            self.waypoints[
                center_index
            ]
        )

        inflation = float(
            self.get_parameter(
                'boundaries_inflation'
            ).value
        )

        left_limit = (
            float(
                ref.d_left
            )
            - inflation
        )

        right_limit = (
            -float(
                ref.d_right
            )
            + inflation
        )

        if (
            left_limit
            <= right_limit
        ):
            return None

        if (
            center_d
            > left_limit
            or center_d
            < right_limit
        ):
            return None

        # ============================================================
        # Optional cluster-inside-ratio filter
        #
        # default = 0.0
        #
        # 즉 현재 UNITA와 동일하게 center 기준으로만 사용.
        # ============================================================
        min_inside_ratio = float(
            self.get_parameter(
                'min_inside_ratio'
            ).value
        )

        if (
            min_inside_ratio
            > 0.0
        ):

            frenet = (
                self.converter.get_frenet(
                    cluster[:, 0],
                    cluster[:, 1],
                )
            )

            s_values = np.mod(
                np.asarray(
                    frenet[0],
                    dtype=float,
                ),
                self.track_length,
            )

            d_values = np.asarray(
                frenet[1],
                dtype=float,
            )

            if (
                not np.all(
                    np.isfinite(
                        s_values
                    )
                )
                or not np.all(
                    np.isfinite(
                        d_values
                    )
                )
            ):
                return None

            inside_count = 0

            for (
                s_value,
                d_value,
            ) in zip(
                s_values,
                d_values,
            ):

                wp = (
                    self.waypoints[
                        self._nearest_waypoint_index(
                            s_value
                        )
                    ]
                )

                point_left = (
                    float(
                        wp.d_left
                    )
                    - inflation
                )

                point_right = (
                    -float(
                        wp.d_right
                    )
                    + inflation
                )

                if (
                    point_left
                    <= point_right
                ):
                    continue

                if (
                    point_right
                    <= d_value
                    <= point_left
                ):
                    inside_count += 1

            inside_ratio = (
                inside_count
                / max(
                    len(
                        cluster
                    ),
                    1,
                )
            )

            if (
                inside_ratio
                < min_inside_ratio
            ):
                return None

        # ============================================================
        # Existing /detect/raw_obstacles message
        #
        # UNIST fitted-square semantics 사용.
        # ============================================================
        half = (
            size
            / 2.0
        )

        return Obstacle(
            id=obstacle_id,

            s_start=float(
                (
                    center_s
                    - half
                )
                % self.track_length
            ),

            s_end=float(
                (
                    center_s
                    + half
                )
                % self.track_length
            ),

            d_right=float(
                center_d
                - half
            ),

            d_left=float(
                center_d
                + half
            ),

            s_center=float(
                center_s
            ),

            d_center=float(
                center_d
            ),

            size=float(
                size
            ),

            # Detection에서는 위치와 크기만 구한다.
            # 실제 velocity / static 판단은 tracking_node에서 담당.
            vs=0.0,
            vd=0.0,

            is_static=False,
            is_visible=True,
        )

    # ================================================================
    # Breakpoint visualization
    # ================================================================

    def _breakpoint_markers(
        self,
        clusters,
        stamp,
    ):

        markers = MarkerArray()

        marker_id = 0

        for cluster in clusters:

            if (
                cluster is None
                or len(
                    cluster
                )
                == 0
            ):
                continue

            # Cluster 시작점과 끝점 표시
            for point in (
                cluster[0],
                cluster[-1],
            ):

                marker = Marker()

                marker.header.frame_id = (
                    'map'
                )

                marker.header.stamp = (
                    stamp
                )

                marker.ns = (
                    'detect_breakpoints'
                )

                marker.id = (
                    marker_id
                )

                marker.type = (
                    Marker.SPHERE
                )

                marker.action = (
                    Marker.ADD
                )

                marker.pose.position.x = float(
                    point[0]
                )

                marker.pose.position.y = float(
                    point[1]
                )

                marker.pose.position.z = 0.0

                marker.pose.orientation.w = 1.0

                marker.scale.x = 0.08
                marker.scale.y = 0.08
                marker.scale.z = 0.08

                marker.color.a = 0.7

                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0

                markers.markers.append(
                    marker
                )

                marker_id += 1

        # ------------------------------------------------------------
        # Delete stale breakpoint markers
        # ------------------------------------------------------------
        for stale_id in range(
            marker_id,
            self._last_breakpoint_marker_count,
        ):

            marker = Marker()

            marker.header.frame_id = (
                'map'
            )

            marker.header.stamp = (
                stamp
            )

            marker.ns = (
                'detect_breakpoints'
            )

            marker.id = (
                stale_id
            )

            marker.action = (
                Marker.DELETE
            )

            markers.markers.append(
                marker
            )

        self._last_breakpoint_marker_count = (
            marker_id
        )

        return markers

    # ================================================================
    # Obstacle visualization
    # ================================================================

    @staticmethod
    def _obstacle_marker(
        center,
        size,
        theta,
        marker_id,
        stamp,
    ):

        marker = Marker()

        marker.header.frame_id = (
            'map'
        )

        marker.header.stamp = (
            stamp
        )

        marker.ns = (
            'detected_obstacles'
        )

        marker.id = (
            marker_id
        )

        marker.type = (
            Marker.CUBE
        )

        marker.action = (
            Marker.ADD
        )

        marker.pose.position.x = float(
            center[0]
        )

        marker.pose.position.y = float(
            center[1]
        )

        marker.pose.position.z = 0.0

        # yaw = theta
        marker.pose.orientation.z = (
            math.sin(
                float(
                    theta
                )
                / 2.0
            )
        )

        marker.pose.orientation.w = (
            math.cos(
                float(
                    theta
                )
                / 2.0
            )
        )

        marker.scale.x = float(
            size
        )

        marker.scale.y = float(
            size
        )

        marker.scale.z = 0.15

        marker.color.a = 0.8

        marker.color.r = 1.0
        marker.color.g = 0.4
        marker.color.b = 0.0

        return marker


def main(args=None):

    rclpy.init(
        args=args
    )

    node = DetectNode()

    try:

        rclpy.spin(
            node
        )

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