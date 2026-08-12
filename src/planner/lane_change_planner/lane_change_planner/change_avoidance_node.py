#!/usr/bin/env python3
"""UNICORN-derived, prediction-gated lane-change avoidance planner."""

from copy import deepcopy
import json
import time

import numpy as np
import rclpy
from f110_msgs.msg import (
    BehaviorStrategy,
    ObstacleArray,
    OTWpntArray,
    PredictionArray,
    Wpnt,
    WpntArray,
)
from geometry_msgs.msg import Point
from grid_filter.grid_filter import GridFilter
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, Header, String
from visualization_msgs.msg import Marker, MarkerArray

from frenet_conversion.frenet_converter import FrenetConverter
from .smoothing import ccma_smooth


def _heading_curvature(xy):
    xy = np.asarray(xy, dtype=float)
    edge = 2 if len(xy) >= 3 else 1
    dx = np.gradient(xy[:, 0], edge_order=edge)
    dy = np.gradient(xy[:, 1], edge_order=edge)
    heading = np.arctan2(dy, dx)
    if len(xy) < 3:
        return heading, np.zeros(len(xy))
    ddx = np.gradient(dx, edge_order=edge)
    ddy = np.gradient(dy, edge_order=edge)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    curvature = np.divide(
        dx * ddy - dy * ddx,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-8,
    )
    return heading, curvature


class ChangeAvoidanceNode(Node):
    """Publish an overtaking path only when a fresh learned prediction is safe."""

    PARAM_DEFAULTS = {
        'rate_hz': 20.0,
        'lookahead': 15.0,
        'vehicle_width': 0.28,
        'safety_margin': 0.10,
        'lane_offset': 0.35,
        'back_to_raceline_before': 3.0,
        'back_to_raceline_after': 3.0,
        'obs_traj_thresh': 2.0,
        'spline_bound_mindist': 0.05,
        'pred_timeout': 0.5,
        'max_evasion_start_offset': 0.8,
        'path_resolution': 0.10,
        'side_switch_frames': 10,
        'map_erosion_kernel_size': 7,
        'clear_after_frames': 5,
        'empty_path_publish_hz': 1.0,
        'measure': False,
    }

    def __init__(self):
        super().__init__('change_avoidance_node')
        for name, default in self.PARAM_DEFAULTS.items():
            self.declare_parameter(name, default)
        self._read_parameters()
        self.add_on_set_parameters_callback(self._parameter_cb)

        self.current_s = None
        self.current_d = None
        self.global_msg = None
        self.scaled_msg = None
        self.updated_msg = None
        self.center_msg = None
        self.behavior = None
        self.converter = None
        self.track_length = None
        self.dynamic_obstacles = ObstacleArray()
        self.predicted_obstacles = ObstacleArray()
        self.predictions = PredictionArray()
        self.prediction_received_at = None
        self.force_trailing = True
        self.lanes = {}
        self.committed_side = None
        self.pending_side = None
        self.pending_count = 0
        self.no_path_count = 0
        self.last_empty_publish_at = None
        self._diagnostic_status = None
        self._diagnostic_payload = None
        self._diagnostic_subscriber_count = 0
        self._side_failure_status = 'NO_SAFE_SIDE'
        self._side_failure_detail = {}
        self.last_switch_time = self.get_clock().now().to_msg()

        self.path_pub = self.create_publisher(
            OTWpntArray, '/planner/avoidance/otwpnts', 10)
        self.merger_pub = self.create_publisher(
            Float32MultiArray, '/planner/avoidance/merger', 10)
        self.path_marker_pub = self.create_publisher(
            MarkerArray, '/planner/avoidance/markers_sqp', 10)
        self.lane_marker_pub = self.create_publisher(
            MarkerArray, '/planner/avoidance/lane_markers', 10)
        self.start_end_pub = self.create_publisher(
            MarkerArray, '/planner/avoidance/start_end_markers', 10)
        self.latency_pub = self.create_publisher(
            Float32, '/planner/avoidance/lane_change_latency', 10)
        # Deliberately volatile: the monitor reports live transitions only.
        self.diagnostic_pub = self.create_publisher(
            String, '/planner/avoidance/diagnostics', 10)

        # Keep the generic UNICORN input name; head_to_head remaps it to the
        # router's dynamic-only stream.
        self.create_subscription(
            ObstacleArray, '/tracking/obstacles', self._obstacle_cb, 10)
        self.create_subscription(
            ObstacleArray, '/opponent_prediction/obstacles', self._predicted_obstacle_cb, 10)
        self.create_subscription(
            PredictionArray, '/opponent_prediction/obstacles_pred', self._prediction_cb, 10)
        self.create_subscription(
            Bool, '/opponent_prediction/force_trailing', self._force_trailing_cb, 10)
        self.create_subscription(
            Odometry, '/car_state/odom_frenet', self._frenet_cb, 10)
        self.create_subscription(
            BehaviorStrategy, '/behavior_strategy', self._behavior_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self._global_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints_scaled', self._scaled_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints_updated', self._updated_cb, 10)
        self.create_subscription(WpntArray, '/centerline_waypoints', self._center_cb, 10)

        self.map_filter = GridFilter(
            self,
            map_topic='/map',
            kernel_size=self.map_erosion_kernel_size,
            unknown_is_occupied=True,
        )
        self.create_timer(1.0 / max(self.rate_hz, 1.0), self._loop)
        self.create_timer(0.2, self._publish_lane_markers)
        self.create_timer(0.5, self._republish_diagnostic_for_new_subscriber)

    def _read_parameters(self):
        for name in self.PARAM_DEFAULTS:
            setattr(self, name, self.get_parameter(name).value)

    def _parameter_cb(self, params):
        regenerate = False
        for parameter in params:
            if parameter.name in self.PARAM_DEFAULTS:
                if parameter.name == 'lane_offset' and parameter.value != self.lane_offset:
                    regenerate = True
                setattr(self, parameter.name, parameter.value)
        if any(p.name == 'map_erosion_kernel_size' for p in params):
            self.map_filter.set_erosion_kernel_size(self.map_erosion_kernel_size)
        if regenerate and self.center_msg is not None and self.converter is not None:
            self._generate_lanes()
        return SetParametersResult(successful=True)

    def _obstacle_cb(self, msg):
        self.dynamic_obstacles = msg

    def _predicted_obstacle_cb(self, msg):
        self.predicted_obstacles = msg

    def _prediction_cb(self, msg):
        self.predictions = msg
        if msg.predictions:
            self.prediction_received_at = self.get_clock().now()

    def _force_trailing_cb(self, msg):
        self.force_trailing = bool(msg.data)

    def _frenet_cb(self, msg):
        self.current_s = float(msg.pose.pose.position.x)
        self.current_d = float(msg.pose.pose.position.y)

    def _behavior_cb(self, msg):
        self.behavior = msg

    def _global_cb(self, msg):
        if not msg.wpnts:
            return
        self.global_msg = msg
        self.track_length = float(msg.wpnts[-1].s_m)
        x = np.asarray([wp.x_m for wp in msg.wpnts], dtype=float)
        y = np.asarray([wp.y_m for wp in msg.wpnts], dtype=float)
        psi = np.asarray([wp.psi_rad for wp in msg.wpnts], dtype=float)
        self.converter = FrenetConverter(x, y, psi)
        if self.center_msg is not None:
            self._generate_lanes()

    def _scaled_cb(self, msg):
        if msg.wpnts:
            self.scaled_msg = msg

    def _updated_cb(self, msg):
        if msg.wpnts:
            self.updated_msg = msg

    def _center_cb(self, msg):
        if msg.wpnts:
            self.center_msg = msg
            if self.converter is not None:
                self._generate_lanes()

    def _generate_lanes(self):
        center_xy = np.asarray([[w.x_m, w.y_m] for w in self.center_msg.wpnts], dtype=float)
        psi = np.asarray([w.psi_rad for w in self.center_msg.wpnts], dtype=float)
        # +d is left in the shared Frenet converter.
        left_xy = center_xy + self.lane_offset * np.column_stack((-np.sin(psi), np.cos(psi)))
        right_xy = center_xy - self.lane_offset * np.column_stack((-np.sin(psi), np.cos(psi)))
        self.lanes = {'center': center_xy, 'left': left_xy, 'right': right_xy}

    def _prediction_fresh(self):
        if self.prediction_received_at is None or not self.predictions.predictions:
            return False
        age = (self.get_clock().now() - self.prediction_received_at).nanoseconds * 1e-9
        return age <= float(self.pred_timeout)

    def _prediction_age(self):
        if self.prediction_received_at is None:
            return float('inf')
        return (
            self.get_clock().now() - self.prediction_received_at
        ).nanoseconds * 1e-9

    def _set_diagnostic(self, status, detail=None):
        """Publish a planner gate transition without loop-rate log spam."""
        if status == self._diagnostic_status:
            return
        self._diagnostic_status = status
        payload = {
            'source': 'planner',
            'status': status,
            'detail': detail or {},
        }
        self._diagnostic_payload = json.dumps(payload)
        self.diagnostic_pub.publish(String(data=self._diagnostic_payload))

    def _republish_diagnostic_for_new_subscriber(self):
        """Give a newly started volatile monitor the current state once."""
        count = self.diagnostic_pub.get_subscription_count()
        subscriber_added = count > self._diagnostic_subscriber_count
        has_payload = self._diagnostic_payload is not None
        if subscriber_added and has_payload:
            self.diagnostic_pub.publish(String(data=self._diagnostic_payload))
        self._diagnostic_subscriber_count = count

    def _reference_at(self, s):
        s_values = np.asarray([w.s_m for w in self.scaled_msg.wpnts])
        return self.scaled_msg.wpnts[int(np.argmin(np.abs(s_values - (s % self.track_length))))]

    def _side_geometry(self, obstacle):
        reference = self._reference_at(obstacle.s_center)
        half_width = 0.5 * float(self.vehicle_width)
        required_center_clearance = half_width + float(self.safety_margin)
        left_target = max(float(self.lane_offset), obstacle.d_left + required_center_clearance)
        right_target = min(-float(self.lane_offset), obstacle.d_right - required_center_clearance)
        left_bound_room = reference.d_left - left_target - half_width
        right_bound_room = reference.d_right + right_target - half_width
        return {
            'left': (left_target, left_bound_room),
            'right': (right_target, right_bound_room),
        }

    def _available_sides(self, obstacle):
        geometry = self._side_geometry(obstacle)
        available = {}
        for side, (target, room) in geometry.items():
            if room >= self.spline_bound_mindist:
                available[side] = (target, room)
        return available

    def _select_side(self, obstacles):
        if not obstacles:
            return None, None
        side_rooms = {'left': [], 'right': []}
        available_for_all = {'left': [], 'right': []}
        for obstacle in obstacles:
            geometry = self._side_geometry(obstacle)
            for side in side_rooms:
                side_rooms[side].append((
                    float(geometry[side][1]), int(obstacle.id),
                    float(obstacle.s_center), float(obstacle.d_center)))
            available = self._available_sides(obstacle)
            for side in list(available_for_all):
                if side not in available:
                    available_for_all.pop(side, None)
                else:
                    available_for_all[side].append(available[side])
        if not available_for_all:
            worst_left = min(side_rooms['left'], default=(float('-inf'), -1, 0.0, 0.0))
            worst_right = min(side_rooms['right'], default=(float('-inf'), -1, 0.0, 0.0))
            self._side_failure_status = 'NO_SAFE_SIDE'
            self._side_failure_detail = {
                'prediction_count': len(obstacles),
                'required_room_m': float(self.spline_bound_mindist),
                'min_left_room_m': round(worst_left[0], 3),
                'left_failure_id': worst_left[1],
                'min_right_room_m': round(worst_right[0], 3),
                'right_failure_id': worst_right[1],
            }
            return None, None
        raw_side = max(
            available_for_all,
            key=lambda side: min(room for _, room in available_for_all[side]),
        )
        target = max(v[0] for v in available_for_all[raw_side]) if raw_side == 'left' \
            else min(v[0] for v in available_for_all[raw_side])
        committed = self._apply_side_hysteresis(raw_side)
        if committed != raw_side:
            # The old side is no longer guaranteed safe for every predicted pose.
            self._side_failure_status = 'SIDE_SWITCH_PENDING'
            self._side_failure_detail = {
                'committed_side': committed,
                'requested_side': raw_side,
                'pending_frames': int(self.pending_count),
                'required_frames': int(self.side_switch_frames),
            }
            return None, None
        return committed, target

    def _apply_side_hysteresis(self, raw_side):
        if self.committed_side is None:
            self.committed_side = raw_side
            self.last_switch_time = self.get_clock().now().to_msg()
        elif raw_side == self.committed_side:
            self.pending_side = None
            self.pending_count = 0
        else:
            if raw_side == self.pending_side:
                self.pending_count += 1
            else:
                self.pending_side = raw_side
                self.pending_count = 1
            if self.pending_count >= int(self.side_switch_frames):
                self.committed_side = raw_side
                self.pending_side = None
                self.pending_count = 0
                self.last_switch_time = self.get_clock().now().to_msg()
        return self.committed_side

    def _considered_obstacles(self):
        if self.current_s is None or not self.track_length:
            return [], 'NOT_READY', {'missing': ['ego_s_or_track_length']}
        actual = [
            obs for obs in self.dynamic_obstacles.obstacles
            if not obs.is_static
            and (obs.s_center - self.current_s) % self.track_length < self.lookahead
            and abs(obs.d_center - self.current_d) < self.obs_traj_thresh
        ]
        if not actual:
            return [], 'NO_DYNAMIC_OBSTACLE', {
                'lookahead_m': float(self.lookahead),
                'lateral_limit_m': float(self.obs_traj_thresh),
            }
        nearest = min(actual, key=lambda obs: (obs.s_center - self.current_s) % self.track_length)
        if nearest.id != self.predictions.id:
            return [], 'PREDICTION_ID_MISMATCH', {
                'obstacle_id': int(nearest.id),
                'prediction_id': int(self.predictions.id),
            }
        predicted = [
            obs for obs in self.predicted_obstacles.obstacles
            if (obs.s_center - self.current_s) % self.track_length < self.lookahead
        ]
        return predicted or [nearest], None, None

    def _first_grid_failure(self, xy):
        """Return the first rejected sample so the failure is actionable."""
        for index, (x, y) in enumerate(np.asarray(xy, dtype=float)):
            if not self.map_filter.is_point_inside(x, y):
                pixel = self.map_filter.world_to_pixel(x, y)
                detail = {
                    'path_index': int(index),
                    'x_m': round(float(x), 3),
                    'y_m': round(float(y), 3),
                }
                if pixel is not None:
                    detail['pixel_x'] = int(pixel[0])
                    detail['pixel_y'] = int(pixel[1])
                return detail
        return None

    def _unwrap(self, s):
        forward = (float(s) - self.current_s) % self.track_length
        if forward > self.track_length - 1.0:
            forward = 0.0
        return self.current_s + forward

    def _plan(self, obstacles):
        side, target_d = self._select_side(obstacles)
        if side is None:
            self._set_diagnostic(
                self._side_failure_status, self._side_failure_detail)
            return None
        starts = [self._unwrap(obs.s_start) for obs in obstacles]
        ends = [self._unwrap(obs.s_end) for obs in obstacles]
        for index in range(len(ends)):
            if ends[index] < starts[index]:
                ends[index] += self.track_length
        obs_start = min(starts)
        obs_end = max(ends)
        path_start = self.current_s
        path_end = min(
            obs_end + self.back_to_raceline_after,
            path_start + 0.5 * self.track_length,
        )
        if path_end - path_start < self.path_resolution * 4:
            self._set_diagnostic('PATH_TOO_SHORT', {
                'length_m': round(path_end - path_start, 3),
            })
            return None

        count = max(5, int(np.ceil((path_end - path_start) / self.path_resolution)) + 1)
        s_unwrapped = np.linspace(path_start, path_end, count)
        d = np.zeros(count, dtype=float)
        ramp_in = max(float(self.back_to_raceline_before), self.path_resolution)
        ramp_out = max(float(self.back_to_raceline_after), self.path_resolution)
        for index, value in enumerate(s_unwrapped):
            if value < obs_start - ramp_in or value > obs_end + ramp_out:
                weight = 0.0
            elif value < obs_start:
                phase = (value - (obs_start - ramp_in)) / ramp_in
                weight = 0.5 * (1.0 - np.cos(np.pi * phase))
            elif value <= obs_end:
                weight = 1.0
            else:
                phase = (value - obs_end) / ramp_out
                weight = 0.5 * (1.0 + np.cos(np.pi * phase))
            d[index] = target_d * weight

        if abs(self.current_d - d[0]) > self.max_evasion_start_offset:
            self._set_diagnostic('START_OFFSET_TOO_LARGE', {
                'offset_m': round(abs(self.current_d - d[0]), 3),
                'limit_m': float(self.max_evasion_start_offset),
            })
            return None
        xy = self.converter.get_cartesian(s_unwrapped % self.track_length, d).T
        if len(xy) < 5 or not np.all(np.isfinite(xy)):
            self._set_diagnostic('INVALID_RAW_PATH', {'point_count': len(xy)})
            return None
        if not self.map_filter.ready:
            self._set_diagnostic('GRID_FILTER_NOT_READY')
            return None
        raw_failure = self._first_grid_failure(xy)
        if raw_failure is not None:
            self._set_diagnostic('RAW_GRID_REJECTED', raw_failure)
            return None

        smoothed_xy = ccma_smooth(xy)
        smoothed_sd = self.converter.get_frenet(smoothed_xy[:, 0], smoothed_xy[:, 1])
        raw_s = np.asarray(smoothed_sd[0])
        out_s = path_start + (raw_s - path_start) % self.track_length
        out_d = np.asarray(smoothed_sd[1])
        keep = np.concatenate([[True], np.diff(out_s) > 1e-4])
        smoothed_xy, out_s, out_d = smoothed_xy[keep], out_s[keep], out_d[keep]
        if len(out_s) < 5:
            self._set_diagnostic('INVALID_SMOOTHED_PATH', {
                'point_count': len(out_s),
            })
            return None
        smoothed_failure = self._first_grid_failure(smoothed_xy)
        if smoothed_failure is not None:
            self._set_diagnostic('SMOOTHED_GRID_REJECTED', smoothed_failure)
            return None

        heading, curvature = _heading_curvature(smoothed_xy)
        output = OTWpntArray()
        output.header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
        output.last_switch_time = self.last_switch_time
        output.side_switch = self.pending_side is not None
        output.ot_side = side
        output.ot_line = 'lane_change'
        for index, (s, lateral, point, psi, kappa) in enumerate(zip(
                out_s, out_d, smoothed_xy, heading, curvature)):
            output.wpnts.append(Wpnt(
                id=index,
                s_m=float(s % self.track_length),
                d_m=float(lateral),
                x_m=float(point[0]),
                y_m=float(point[1]),
                psi_rad=float(psi),
                kappa_radpm=float(kappa),
                vx_mps=0.0,
                ax_mps2=0.0,
            ))
        return output, obs_start, obs_end, path_end

    def _empty_path(self):
        return OTWpntArray(
            header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'))

    def _publish_empty_if_due(self):
        """Refresh an empty path at a low rate while no valid path exists."""
        now = self.get_clock().now()
        hz = max(0.01, float(self.empty_path_publish_hz))
        due = self.last_empty_publish_at is None or (
            now - self.last_empty_publish_at
        ).nanoseconds * 1e-9 >= 1.0 / hz
        if due:
            self.path_pub.publish(self._empty_path())
            self.last_empty_publish_at = now

    def _loop(self):
        started = time.perf_counter()
        required_inputs = {
            'ego_s': self.current_s,
            'converter': self.converter,
            'global_waypoints_scaled': self.scaled_msg,
            'global_waypoints_updated': self.updated_msg,
            'centerline_waypoints': self.center_msg,
            'behavior_strategy': self.behavior,
        }
        missing = [
            name for name, value in required_inputs.items()
            if value is None
        ]
        ready = not missing
        result = None
        obstacles = []
        if not ready:
            self._set_diagnostic('NOT_READY', {'missing': missing})
        elif not self.predictions.predictions:
            self._set_diagnostic('PREDICTION_MISSING')
        elif self.prediction_received_at is None:
            self._set_diagnostic('PREDICTION_MISSING')
        elif not self._prediction_fresh():
            self._set_diagnostic('PREDICTION_STALE', {
                'age_s': round(self._prediction_age(), 3),
                'limit_s': float(self.pred_timeout),
            })
        elif self.force_trailing:
            self._set_diagnostic('FORCE_TRAILING')
        else:
            obstacles, gate_status, gate_detail = self._considered_obstacles()
            if gate_status is not None:
                self._set_diagnostic(gate_status, gate_detail)
            if obstacles:
                result = self._plan(deepcopy(obstacles))

        if result is not None:
            path, obs_start, obs_end, path_end = result
            self.path_pub.publish(path)
            self.merger_pub.publish(Float32MultiArray(data=[
                float(obs_end % self.track_length),
                float(path_end % self.track_length),
            ]))
            self._publish_path_markers(path)
            self._publish_start_end_markers(self.current_s, obs_start, obs_end, path_end)
            self.no_path_count = 0
            self.last_empty_publish_at = None
            self._set_diagnostic('PATH_READY', {
                'side': path.ot_side,
                'point_count': len(path.wpnts),
            })
        else:
            self.no_path_count += 1
            if self.no_path_count == int(self.clear_after_frames):
                self._clear_path_markers()
                self.committed_side = None
                self.pending_side = None
                self.pending_count = 0
            if self.no_path_count >= int(self.clear_after_frames):
                self._publish_empty_if_due()
        if self.measure:
            self.latency_pub.publish(Float32(data=float(time.perf_counter() - started)))

    def _publish_lane_markers(self):
        if not self.lanes:
            return
        array = MarkerArray()
        specs = (
            ('center', self.lanes['center'], (1.0, 1.0, 1.0)),
            ('left', self.lanes['left'], (0.1, 0.9, 0.1)),
            ('right', self.lanes['right'], (0.9, 0.1, 0.1)),
        )
        for marker_id, (name, points, color) in enumerate(specs):
            marker = Marker(header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'))
            marker.ns = 'lane_change_' + name
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.03
            marker.color.a = 1.0
            marker.color.r, marker.color.g, marker.color.b = color
            marker.pose.orientation.w = 1.0
            marker.points = [Point(x=float(x), y=float(y), z=0.0) for x, y in points]
            array.markers.append(marker)
        self.lane_marker_pub.publish(array)

    def _publish_path_markers(self, path):
        array = MarkerArray()
        delete = Marker(header=path.header, action=Marker.DELETEALL)
        array.markers.append(delete)
        for index, waypoint in enumerate(path.wpnts):
            marker = Marker(header=path.header)
            marker.ns = 'lane_change_path'
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = waypoint.x_m
            marker.pose.position.y = waypoint.y_m
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.08
            marker.color.a = 1.0
            marker.color.r = 0.63
            marker.color.g = 0.13
            marker.color.b = 0.94
            array.markers.append(marker)
        self.path_marker_pub.publish(array)

    def _publish_start_end_markers(self, start, obs_start, obs_end, end):
        s = np.asarray([start, obs_start, obs_end, end]) % self.track_length
        xy = self.converter.get_cartesian(s, np.zeros(4)).T
        colors = ((0.0, 1.0, 1.0), (1.0, 1.0, 0.0),
                  (1.0, 0.5, 0.0), (1.0, 0.0, 1.0))
        array = MarkerArray()
        for index, (point, color) in enumerate(zip(xy, colors)):
            marker = Marker(header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'))
            marker.ns = 'lane_change_limits'
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(point[0])
            marker.pose.position.y = float(point[1])
            marker.pose.position.z = 0.15
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.3
            marker.color.a = 0.9
            marker.color.r, marker.color.g, marker.color.b = color
            array.markers.append(marker)
        self.start_end_pub.publish(array)

    def _clear_path_markers(self):
        for publisher in (self.path_marker_pub, self.start_end_pub):
            publisher.publish(MarkerArray(markers=[Marker(
                header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'),
                action=Marker.DELETEALL,
            )]))


def main(args=None):
    rclpy.init(args=args)
    node = ChangeAvoidanceNode()
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
