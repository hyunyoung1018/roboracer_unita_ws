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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
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
        # [m] How much of the opponent's predicted future the evasion corridor
        # has to span. See _considered_obstacles.
        'prediction_span_m': 3.0,
        # [m] Distance over which the published path is blended from the car's
        # current d onto the planned lateral profile. See _plan.
        'start_blend_m': 1.0,
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
        self._scaled_s = None
        self.updated_msg = None
        self.center_msg = None
        self.behavior = None
        self.converter = None
        self.track_length = None
        self.dynamic_obstacles = ObstacleArray()
        self.static_obstacles = ObstacleArray()
        self.predicted_obstacles = ObstacleArray()
        self._lane_marker_cache = None
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
        # The planner used to know only about the opponent, so a lane change
        # around it could be drawn straight through a static obstacle - nothing
        # in this node had ever heard of one. Absolute topic name, not the
        # remapped /tracking/obstacles: this is the router's static-only stream,
        # the same one the spline planner avoids on. Nothing publishes it in
        # time_trials, and this node is not launched there either, so an empty
        # array simply restores the previous behaviour.
        self.create_subscription(
            ObstacleArray, '/tracking/static_obstacles', self._static_obstacle_cb, 10)
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
        # "left" / "right" / "auto" for the car's current s, from the map's
        # ot_sectors.yaml by way of the state machine. Transient-local, so
        # starting after it has already published still gets the answer.
        self.preferred_side = 'auto'
        self.create_subscription(
            String, '/ot_preferred_side', self._preferred_side_cb,
            QoSProfile(depth=1,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))

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

    def _static_obstacle_cb(self, msg):
        self.static_obstacles = msg

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
            # Cached because _reference_at is the hottest call in the node:
            # _select_side asks for one reference waypoint per predicted pose,
            # and rebuilding this array inside it meant walking the whole
            # scaled raceline hundreds of times a second on the jetson.
            self._scaled_s = np.asarray(
                [w.s_m for w in msg.wpnts], dtype=float)

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
        # The marker geometry is derived from these, so it is stale now.
        self._lane_marker_cache = None

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
        s_values = self._scaled_s
        if s_values is None:
            s_values = np.asarray(
                [w.s_m for w in self.scaled_msg.wpnts], dtype=float)
        return self.scaled_msg.wpnts[
            int(np.argmin(np.abs(s_values - (s % self.track_length))))]

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

    def _available_sides(self, obstacle, geometry=None):
        if geometry is None:
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
            # One _side_geometry per obstacle. It was called twice - once here
            # and once inside _available_sides - and each call walked the whole
            # scaled raceline through _reference_at.
            geometry = self._side_geometry(obstacle)
            for side in side_rooms:
                side_rooms[side].append((
                    float(geometry[side][1]), int(obstacle.id),
                    float(obstacle.s_center), float(obstacle.d_center)))
            available = self._available_sides(obstacle, geometry)
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
        raw_side = self._policy_side(available_for_all)
        committed = self._apply_side_hysteresis(raw_side)
        if committed != raw_side:
            # A pending switch used to mean no path at all for
            # side_switch_frames loops - half a second of blackout at 20 Hz,
            # during which the state machine sees the planner stop publishing.
            # The committed side is what the hysteresis exists to hold, so hold
            # it: keep planning on it for as long as it is still safe for every
            # pose, and only give up when it is not.
            if committed not in available_for_all:
                self._side_failure_status = 'SIDE_SWITCH_PENDING'
                self._side_failure_detail = {
                    'committed_side': committed,
                    'requested_side': raw_side,
                    'pending_frames': int(self.pending_count),
                    'required_frames': int(self.side_switch_frames),
                }
                return None, None
            chosen = committed
        else:
            chosen = raw_side
        target = max(v[0] for v in available_for_all[chosen]) if chosen == 'left' \
            else min(v[0] for v in available_for_all[chosen])
        return chosen, target

    def _policy_side(self, available_for_all):
        """Which of the SAFE sides to take.

        The room comparison is the fallback, not the rule. It picks whichever
        side has the most room at its tightest point, which on a corner is the
        outside every time - the inside is shorter and is where an opponent
        running wide leaves a gap, and neither of those is anything this
        function can see.

        Worse, it re-decides every tick off a number that moves. The opponent's
        measured edge wanders about 10 cm frame to frame on this car, and when
        the two sides are within that of each other the answer flips, which is
        an overtake started and abandoned several times a second.
        side_hysteresis_m damps that; it does not decide it.

        So the map gets to say. ot_sectors.yaml carries preferred_side per
        overtaking sector and the state machine publishes the one for the car's
        current s - it already walks that table every tick for _check_ot_sector.
        "auto", and every sector that does not set it, is exactly the old
        behaviour.

        SAFETY IS NOT POLICY: this only ever chooses among sides that already
        passed _available_sides for every obstacle. A preference for a side
        that is not safe is ignored, and the car goes the other way rather than
        not at all.
        """
        by_room = max(
            available_for_all,
            key=lambda side: min(room for _, room in available_for_all[side]),
        )
        preferred = self.preferred_side
        if preferred in available_for_all and preferred != by_room:
            self.get_logger().info(
                f"taking the {preferred} side by sector policy; {by_room} has "
                f"more room here", throttle_duration_sec=2.0)
            return preferred
        if preferred not in ('auto', None) and preferred not in available_for_all:
            self.get_logger().info(
                f"sector policy asks for {preferred} but no side of every "
                f"obstacle clears there - taking {by_room}",
                throttle_duration_sec=2.0)
        return by_room

    def _preferred_side_cb(self, msg):
        side = str(msg.data).strip().lower()
        self.preferred_side = side if side in ('left', 'right') else 'auto'

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
        predicted = self._truncate_prediction(predicted)
        return predicted or [nearest], None, None

    def _truncate_prediction(self, predicted):
        """Keep only the near part of the opponent's predicted future.

        _select_side requires ONE side to be clear at EVERY pose it is given,
        and the predictor hands over n_time_steps * dt = 2 s of future. At
        3 m/s that is a 6 m corridor which has to be clear on the same side end
        to end, and the evasion offset it has to be clear BY is
        opponent_half + vehicle_half + safety_margin + spline_bound_mindist.
        Measured against the two maps in this workspace:

            corridor   test_213        map_no
            0 m        99 % of lap     100 %
            3 m        53 %             65 %
            6 m         2 %             10 %

        So at racing speed the manoeuvre was refused almost everywhere, and the
        refusal had nothing to do with the opponent - it was the track being
        21 m long and 1.05 to 1.85 m wide.

        The full span is not what the car needs. It replans at 20 Hz, so the
        corridor only has to be clear over the stretch the car will actually
        cover before the next decision, and it extends itself as the car moves.
        Committing 6 m ahead on a track this size is committing a third of a
        lap on one prediction.

        Truncating here rather than in _plan keeps obs_end - and therefore the
        path's ramp-out and its length - consistent with what was checked.
        """
        span = float(self.prediction_span_m)
        if span <= 0.0 or len(predicted) < 2:
            return predicted
        origin = float(predicted[0].s_center)
        kept = [
            obs for obs in predicted
            if (float(obs.s_center) - origin) % self.track_length <= span
        ]
        # Always keep at least the pose the opponent is at now plus one, so the
        # corridor is never degenerate.
        return kept if len(kept) >= 2 else predicted[:2]

    def _first_grid_failure(self, xy):
        """Return the first rejected sample so the failure is actionable.

        The verdict comes from one vectorised pass; the report - which costs a
        world_to_pixel and builds a dict - is only assembled for the one sample
        that actually failed. Previously every sample paid for both.
        """
        points = np.asarray(xy, dtype=float)
        index = self.map_filter.first_outside_index(points)
        if index is None:
            return None
        x, y = float(points[index, 0]), float(points[index, 1])
        detail = {
            'path_index': int(index),
            'x_m': round(x, 3),
            'y_m': round(y, 3),
        }
        pixel = self.map_filter.world_to_pixel(x, y)
        if pixel is not None:
            detail['pixel_x'] = int(pixel[0])
            detail['pixel_y'] = int(pixel[1])
        return detail

    def _unwrap(self, s):
        forward = (float(s) - self.current_s) % self.track_length
        if forward > self.track_length - 1.0:
            forward = 0.0
        return self.current_s + forward

    def _statics_in_span(self, obs_end):
        """Static obstacles the planned path would actually run over.

        The path leaves the raceline before the opponent and rejoins
        back_to_raceline_after past it, so anything static between the car and
        that rejoin point is on the manoeuvre whether this planner models it or
        not. Those get folded into the same side selection and corridor as the
        opponent; a static obstacle beyond the rejoin is left alone, because the
        path does not go there and the next replan will have moved the window.

        Note this can end in NO_SAFE_SIDE, i.e. no lane change at all. That is
        the point - refusing to pull out is the correct answer when the gap is
        blocked, and the state machine then keeps trailing.
        """
        if not self.static_obstacles.obstacles or not self.track_length:
            return []
        span_end = obs_end + float(self.back_to_raceline_after)
        kept = []
        for obs in self.static_obstacles.obstacles:
            centre = self._unwrap(obs.s_center)
            # _unwrap folds everything onto [current_s, current_s + L), and
            # collapses anything just behind the car to current_s exactly, so a
            # strictly positive forward distance separates "in front" from
            # "level with or behind".
            if self.path_resolution < centre - self.current_s and centre <= span_end:
                kept.append(obs)
        return kept

    def _plan(self, obstacles):
        starts = [self._unwrap(obs.s_start) for obs in obstacles]
        ends = [self._unwrap(obs.s_end) for obs in obstacles]
        for index in range(len(ends)):
            if ends[index] < starts[index]:
                ends[index] += self.track_length
        obs_start = min(starts)
        obs_end = max(ends)

        # Widen the problem to every obstacle the path will pass, then solve it
        # once. Done here rather than in _considered_obstacles because the span
        # that decides which statics matter is not known until the opponent's
        # own start and end are.
        statics = self._statics_in_span(obs_end)
        if statics:
            obstacles = list(obstacles) + statics
            for obs in statics:
                start = self._unwrap(obs.s_start)
                end = self._unwrap(obs.s_end)
                if end < start:
                    end += self.track_length
                obs_start = min(obs_start, start)
                obs_end = max(obs_end, end)

        side, target_d = self._select_side(obstacles)
        if side is None:
            if statics:
                self._side_failure_detail = dict(
                    self._side_failure_detail or {},
                    static_obstacle_count=len(statics))
            self._set_diagnostic(
                self._side_failure_status, self._side_failure_detail)
            return None
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

        # Start the path AT the car, not beside it. The ramp weight at index 0
        # is whatever the raised-cosine happens to be there, so if the car has
        # already passed obs_start - back_to_raceline_before, d[0] is a lateral
        # offset the car is not at. The only guard was max_evasion_start_offset,
        # which permitted a first waypoint up to 0.8 m to the side; the
        # controller then cut straight across to it, inside a corridor barely a
        # metre wide. Blending the discontinuity out over start_blend_m makes
        # d[0] exactly current_d and leaves the rest of the profile alone.
        blend_length = max(float(self.start_blend_m), self.path_resolution)
        offset = float(self.current_d) - float(d[0])
        if abs(offset) > 1e-6:
            decay = np.clip(
                1.0 - (s_unwrapped - path_start) / blend_length, 0.0, 1.0)
            d = d + offset * decay
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
        """Redraw the three lanes 5x a second - but they never move.

        The geometry only changes when _generate_lanes runs (a new centerline,
        or lane_offset retuned), so the Point lists are built there and cached.
        This used to rebuild three LINE_STRIPs over the whole track, one Python
        Point object per waypoint, five times a second, unconditionally.
        """
        if not self.lanes or self.lane_marker_pub.get_subscription_count() == 0:
            return
        if self._lane_marker_cache is None:
            self._lane_marker_cache = [
                self._lane_marker(marker_id, name, self.lanes[name], color)
                for marker_id, (name, color) in enumerate((
                    ('center', (1.0, 1.0, 1.0)),
                    ('left', (0.1, 0.9, 0.1)),
                    ('right', (0.9, 0.1, 0.1)),
                ))
            ]
        stamp = self.get_clock().now().to_msg()
        for marker in self._lane_marker_cache:
            marker.header.stamp = stamp
        self.lane_marker_pub.publish(MarkerArray(markers=self._lane_marker_cache))

    @staticmethod
    def _lane_marker(marker_id, name, points, color):
        marker = Marker(header=Header(frame_id='map'))
        marker.ns = 'lane_change_' + name
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = color
        marker.pose.orientation.w = 1.0
        marker.points = [Point(x=float(x), y=float(y), z=0.0) for x, y in points]
        return marker

    def _publish_path_markers(self, path):
        """Draw the path as ONE marker, and only when something is watching.

        This was a SPHERE per waypoint: 80 to 150 Marker messages built and
        serialised on every successful plan, at 20 Hz, whether or not RViz was
        open - a few thousand messages a second for a picture nobody was
        looking at. A single SPHERE_LIST carries the same points in one
        message, and the subscriber check makes a closed RViz free.

        The DELETEALL is kept: the old per-index markers have to be cleared
        once, and a viewer that connects later gets it on the next plan.
        """
        if self.path_marker_pub.get_subscription_count() == 0:
            return
        marker = Marker(header=path.header)
        marker.ns = 'lane_change_path'
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.08
        marker.color.a = 1.0
        marker.color.r = 0.63
        marker.color.g = 0.13
        marker.color.b = 0.94
        marker.points = [
            Point(x=float(w.x_m), y=float(w.y_m), z=0.0) for w in path.wpnts]
        self.path_marker_pub.publish(MarkerArray(markers=[
            Marker(header=path.header, action=Marker.DELETEALL),
            marker,
        ]))

    def _publish_start_end_markers(self, start, obs_start, obs_end, end):
        if self.start_end_pub.get_subscription_count() == 0:
            return
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
