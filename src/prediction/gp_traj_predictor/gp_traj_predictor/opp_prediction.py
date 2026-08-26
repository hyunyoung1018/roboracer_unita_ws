#!/usr/bin/env python3
"""Generate safe fallback or learned future trajectories for one opponent."""

import json

import numpy as np
import rclpy
from f110_msgs.msg import (
    Obstacle,
    ObstacleArray,
    OpponentTrajectory,
    Prediction,
    PredictionArray,
    WpntArray,
)
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Header, String
from visualization_msgs.msg import MarkerArray

from frenet_conversion.frenet_converter import FrenetConverter
from .prediction_node import nearest_dynamic, periodic_interp, trajectory_markers


def _circular_nearest_distances(query_s, reference_s, track_length):
    query_s = np.asarray(query_s, dtype=float) % track_length
    reference_s = np.asarray(reference_s, dtype=float) % track_length
    if len(reference_s) == 0:
        return np.full(len(query_s), np.inf)
    difference = np.abs(query_s[:, None] - reference_s[None, :])
    return np.min(np.minimum(difference, track_length - difference), axis=1)


def validate_learned_profile(
        profile_s, profile_d, trajectory, global_waypoints, track_length,
        opponent_width, boundary_margin, max_query_gap, max_d_variance):
    """Validate that every future point is observed, bounded and trustworthy."""
    profile_s = np.asarray(profile_s, dtype=float)
    profile_d = np.asarray(profile_d, dtype=float)
    if not trajectory or not global_waypoints or len(profile_s) != len(profile_d):
        return False, 'INVALID_LEARNED_TRAJECTORY', {}
    if not np.all(np.isfinite(profile_s + profile_d)):
        return False, 'INVALID_LEARNED_TRAJECTORY', {'reason': 'non_finite'}

    trajectory_s = np.asarray([point.s_m for point in trajectory], dtype=float)
    query_gap = _circular_nearest_distances(
        profile_s, trajectory_s, track_length)
    worst_gap_index = int(np.argmax(query_gap))
    if query_gap[worst_gap_index] > float(max_query_gap):
        return False, 'TRAJECTORY_UNOBSERVED', {
            'query_s_m': round(float(profile_s[worst_gap_index] % track_length), 3),
            'nearest_observation_m': round(float(query_gap[worst_gap_index]), 3),
            'limit_m': float(max_query_gap),
        }

    trajectory_var = np.asarray(
        [max(0.0, float(point.d_var)) for point in trajectory], dtype=float)
    query_var = periodic_interp(
        profile_s, trajectory_s, trajectory_var, track_length)
    worst_var_index = int(np.argmax(query_var))
    if query_var[worst_var_index] > float(max_d_variance):
        return False, 'TRAJECTORY_UNCERTAIN', {
            'query_s_m': round(float(profile_s[worst_var_index] % track_length), 3),
            'd_variance': round(float(query_var[worst_var_index]), 3),
            'limit': float(max_d_variance),
        }

    global_s = np.asarray([point.s_m for point in global_waypoints], dtype=float)
    half_width = 0.5 * float(opponent_width)
    clearance = half_width + float(boundary_margin)
    return profile_inside_track(
        profile_s, profile_d, global_waypoints, track_length,
        opponent_width, boundary_margin, 'TRAJECTORY_OUT_OF_BOUNDS')


def profile_inside_track(profile_s, profile_d, global_waypoints, track_length,
                         opponent_width, boundary_margin, status):
    """Check a predicted profile against the drivable corridor.

    Split out of validate_learned_profile because it is the only one of its
    three tests that needs no learned trajectory - it reads the global
    waypoints' own d_left/d_right. A constant-velocity prediction has no
    observation coverage and no GP variance to check, but it can and must
    still be checked against the track.
    """
    profile_s = np.asarray(profile_s, dtype=float)
    profile_d = np.asarray(profile_d, dtype=float)
    if not global_waypoints or len(profile_s) != len(profile_d):
        return False, 'INVALID_LEARNED_TRAJECTORY', {}
    if not np.all(np.isfinite(profile_s + profile_d)):
        return False, 'INVALID_LEARNED_TRAJECTORY', {'reason': 'non_finite'}
    global_s = np.asarray([point.s_m for point in global_waypoints], dtype=float)
    clearance = 0.5 * float(opponent_width) + float(boundary_margin)
    for index, (s, d) in enumerate(zip(profile_s, profile_d)):
        distance = np.abs(global_s - (s % track_length))
        distance = np.minimum(distance, track_length - distance)
        waypoint = global_waypoints[int(np.argmin(distance))]
        lower = -float(waypoint.d_right) + clearance
        upper = float(waypoint.d_left) - clearance
        if lower >= upper or d < lower - 1e-6 or d > upper + 1e-6:
            return False, status, {
                'query_s_m': round(float(s % track_length), 3),
                'd_m': round(float(d), 3),
                'right_limit_m': round(float(lower), 3),
                'left_limit_m': round(float(upper), 3),
                'point_index': index,
            }
    return True, None, None


class OpponentPredictor(Node):
    """Publish a learned prediction only after a complete, consistent trajectory."""

    def __init__(self):
        super().__init__('opp_prediction')
        defaults = {
            'loop_rate': 20.0,
            'n_time_steps': 20,
            'dt': 0.10,
            'max_opponent_distance': 8.0,
            'obstacle_timeout': 0.5,
            'trajectory_timeout': 2.0,
            'min_training_laps': 1.0,
            'learned_deviation_enter_threshold': 0.35,
            'learned_deviation_exit_threshold': 0.55,
            'learned_ready_confirm_frames': 3,
            'learned_reject_confirm_frames': 5,
            'opponent_width': 0.28,
            'trajectory_boundary_margin': 0.03,
            'max_trajectory_query_gap': 0.15,
            'max_trajectory_d_variance': 0.80,
            'speed_offset': 0.0,
            # --- constant-velocity authorization -------------------------------
            # force_trailing used to be `not learned`, so overtaking required the
            # whole GP chain to be green. On this workspace's 20-22 m tracks that
            # gate is close to unsatisfiable: the GP stamps every point it had to
            # clip at the track boundary with boundary_clipped_variance 1.0, and
            # the acceptance limit is 0.80, so one wall-hugging section of the
            # opponent's line refuses the whole prediction. That is why the car
            # has never overtaken.
            #
            # With this true the predictor may also authorize an overtake on the
            # fallback profile - constant velocity from the tracker's own vs,
            # blended onto the centreline. That is a weaker claim than a learned
            # trajectory, so it is gated on the things constant velocity actually
            # needs to hold, and the horizon is meant to be shortened to about a
            # second (see n_time_steps in the launch).
            #
            # False reproduces the original behaviour exactly.
            'authorize_on_fallback': False,
            # [m/s] Below this the opponent is not worth a lane change; it is
            # about to be reclassified STATIC by the router anyway, and static
            # avoidance is the correct planner for a stopped car.
            'fallback_min_opponent_speed_mps': 0.30,
            # [m/s] How far the ego may be LOSING ground and still be allowed
            # to pass, as a negative number read through `closing >= -this`.
            #
            # It was 0.25 the other way round - "ego must actually be closing" -
            # and that could never be satisfied. Trailing drives the ego at the
            # opponent's speed, so once the gap settles `ego_vs - opponent_vs`
            # is zero by construction and 0.25 shut the gate exactly when
            # trailing was working. Every authorization then failed
            # CONSTVEL_NOT_CLOSING, force_trailing stayed true, and the state
            # machine's own entry gate was vetoed on top of it.
            #
            # The claim it was making - "this pass can be completed" - is now
            # made properly by fallback_min_speed_advantage_mps below, against
            # a speed that trailing does not suppress. What is left here is the
            # narrower question this measurement CAN answer: is the car falling
            # behind right now, whatever the raceline says it could do. Matches
            # the state machine's own -0.5.
            'fallback_max_losing_mps': 0.5,
            # [m/s] Speed advantage the ego must be CAPABLE of before a
            # constant-velocity prediction may authorize a pass, measured
            # against the scaled raceline at the ego's s.
            #
            # Time spent alongside is about (car length + margin) / advantage,
            # so 1 m over the 2 s prediction horizon is a 0.5 m/s floor. 1.0
            # doubles it because 2 s is optimistic - the lane-change planner
            # validates only prediction_span_m of the opponent's future.
            #
            # Kept equal to the state machine's overtake_min_speed_advantage_mps
            # on purpose. This node vetoes through force_trailing, so a stricter
            # bar here would silently become the binding one.
            #'fallback_min_speed_advantage_mps': 0.5,
            'fallback_min_speed_advantage_mps': 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.ego_s = None
        self.ego_vs = None
        self.global_msg = None
        self.scaled_msg = None
        self.updated_msg = None
        self.center_msg = None
        self.track_length = None
        self.converter = None
        self.obstacles = ObstacleArray()
        self.obstacles_received_at = None
        self.trajectory = None
        self.trajectory_received_at = None
        self._diagnostic_status = None
        self._diagnostic_payload = None
        self._diagnostic_subscriber_count = 0
        self._learned_gate_open = False
        self._learned_ready_count = 0
        self._learned_reject_count = 0

        self.obstacle_pub = self.create_publisher(
            ObstacleArray, '/opponent_prediction/obstacles', 10)
        self.prediction_pub = self.create_publisher(
            PredictionArray, '/opponent_prediction/obstacles_pred', 10)
        self.force_trailing_pub = self.create_publisher(
            Bool, '/opponent_prediction/force_trailing', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/opponent_prediction_markerarray', 10)
        # Keep volatile QoS: this is live state, not latched configuration.
        self.diagnostic_pub = self.create_publisher(
            String, '/opponent_prediction/diagnostics', 10)

        # Generic source name is remapped to /tracking/dynamic_obstacles.
        self.create_subscription(
            ObstacleArray, '/tracking/obstacles', self._obstacle_cb, 10)
        self.create_subscription(
            Odometry, '/car_state/odom_frenet', self._odom_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self._global_cb, 10)
        # Speeds only, and the SCALED ones: speed_scaling.yaml is baked into
        # this topic, so it is what the car would actually drive here.
        # /global_waypoints above carries the unscaled optimiser output,
        # which would overstate the advantage by 1/scaling.
        self.create_subscription(
            WpntArray, '/global_waypoints_scaled', self._scaled_cb, 10)
        self.create_subscription(
            WpntArray, '/global_waypoints_updated', self._updated_cb, 10)
        self.create_subscription(
            WpntArray, '/centerline_waypoints', self._center_cb, 10)
        self.create_subscription(
            OpponentTrajectory, '/opponent_trajectory', self._trajectory_cb, 10)
        rate = float(self.get_parameter('loop_rate').value)
        self.create_timer(1.0 / max(rate, 1.0), self._loop)
        self.create_timer(0.5, self._republish_diagnostic_for_new_subscriber)

    def _obstacle_cb(self, msg):
        self.obstacles = msg
        self.obstacles_received_at = self.get_clock().now()

    def _odom_cb(self, msg):
        self.ego_s = float(msg.pose.pose.position.x)
        # Frenet forward speed, for the closing test in _fallback_authorized.
        self.ego_vs = float(msg.twist.twist.linear.x)

    def _global_cb(self, msg):
        if not msg.wpnts:
            return
        self.global_msg = msg
        self.track_length = float(msg.wpnts[-1].s_m)
        x = np.asarray([w.x_m for w in msg.wpnts], dtype=float)
        y = np.asarray([w.y_m for w in msg.wpnts], dtype=float)
        psi = np.asarray([w.psi_rad for w in msg.wpnts], dtype=float)
        self.converter = FrenetConverter(x, y, psi)

    def _scaled_cb(self, msg):
        if msg.wpnts:
            self.scaled_msg = msg

    def _raceline_speed_at(self, s_m):
        """Scaled raceline speed at ``s_m``, or None if it is not known yet."""
        msg = self.scaled_msg or self.global_msg
        if msg is None or not msg.wpnts or not self.track_length:
            return None
        wpnts = msg.wpnts
        spacing = self.track_length / len(wpnts)
        if spacing <= 0.0:
            return None
        index = int((float(s_m) % self.track_length) / spacing) % len(wpnts)
        return float(wpnts[index].vx_mps)

    def _updated_cb(self, msg):
        if msg.wpnts:
            self.updated_msg = msg

    def _center_cb(self, msg):
        if msg.wpnts:
            self.center_msg = msg

    def _trajectory_cb(self, msg):
        self.trajectory = msg
        self.trajectory_received_at = self.get_clock().now()

    def _age(self, received_at):
        if received_at is None:
            return float('inf')
        return (self.get_clock().now() - received_at).nanoseconds * 1e-9

    def _set_diagnostic(self, status, detail=None):
        """Publish only a diagnostic state transition, never every loop."""
        if status == self._diagnostic_status:
            return
        self._diagnostic_status = status
        payload = {
            'source': 'predictor',
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

    def _learned_status(self, obstacle):
        """Return learned readiness and the exact force-trailing reason."""
        if self.trajectory is None or not self.trajectory.oppwpnts:
            self._reset_learned_gate()
            return False, 'NO_TRAJECTORY', {'obstacle_id': int(obstacle.id)}
        timeout = float(self.get_parameter('trajectory_timeout').value)
        age = self._age(self.trajectory_received_at)
        if age > timeout:
            self._reset_learned_gate()
            return False, 'TRAJECTORY_STALE', {
                'age_s': round(age, 3),
                'limit_s': timeout,
            }
        minimum_laps = float(self.get_parameter('min_training_laps').value)
        lap_count = float(self.trajectory.lap_count)
        if lap_count < minimum_laps:
            self._reset_learned_gate()
            return False, 'TRAINING', {
                'lap_count': round(lap_count, 3),
                'required_laps': minimum_laps,
            }
        if not self.trajectory.opp_is_on_trajectory:
            self._reset_learned_gate()
            return False, 'OFF_TRAJECTORY', {
                'obstacle_id': int(obstacle.id),
            }
        s = [w.s_m for w in self.trajectory.oppwpnts]
        d = [w.d_m for w in self.trajectory.oppwpnts]
        expected = float(periodic_interp(
            [obstacle.s_center], s, d, self.track_length)[0])
        valid, invalid_status, invalid_detail = self._validate_learned_profile(
            [obstacle.s_center], [expected])
        if not valid:
            self._reset_learned_gate()
            return False, invalid_status, invalid_detail
        deviation = abs(float(obstacle.d_center) - expected)
        enter_threshold = float(self.get_parameter(
            'learned_deviation_enter_threshold').value)
        exit_threshold = float(self.get_parameter(
            'learned_deviation_exit_threshold').value)
        ready_frames = max(1, int(self.get_parameter(
            'learned_ready_confirm_frames').value))
        reject_frames = max(1, int(self.get_parameter(
            'learned_reject_confirm_frames').value))

        if self._learned_gate_open:
            if deviation >= exit_threshold:
                self._learned_reject_count += 1
            else:
                self._learned_reject_count = 0
            if self._learned_reject_count >= reject_frames:
                self._reset_learned_gate()
                return False, 'DEVIATION_TOO_LARGE', {
                    'deviation_m': round(deviation, 3),
                    'exit_limit_m': exit_threshold,
                    'reject_frames': reject_frames,
                }
            return True, 'LEARNED_READY', {
                'obstacle_id': int(obstacle.id),
                'lap_count': round(lap_count, 3),
                'deviation_m': round(deviation, 3),
            }

        if deviation <= enter_threshold:
            self._learned_ready_count += 1
        else:
            self._learned_ready_count = 0
        if self._learned_ready_count < ready_frames:
            status = 'LEARNED_CONFIRMING' \
                if deviation <= enter_threshold else 'DEVIATION_TOO_LARGE'
            return False, status, {
                'deviation_m': round(deviation, 3),
                'enter_limit_m': enter_threshold,
                'ready_frames': self._learned_ready_count,
                'required_frames': ready_frames,
            }
        self._learned_gate_open = True
        self._learned_ready_count = 0
        self._learned_reject_count = 0
        return True, 'LEARNED_READY', {
            'obstacle_id': int(obstacle.id),
            'lap_count': round(lap_count, 3),
            'deviation_m': round(deviation, 3),
        }

    def _fallback_authorized(self, obstacle, profile_s, profile_d):
        """Decide whether a constant-velocity prediction may authorize passing.

        The learned gate answered "do I know where the opponent will be over the
        next two seconds". Constant velocity cannot answer that, and pretending
        otherwise is how a prediction-free stack drives into a corner. What it
        can answer is "is this a confirmed, moving opponent that I am closing on,
        whose next second stays on the track" - so that is what is checked, and
        the horizon is expected to be short enough that the answer stays true.

        Being on this topic at all is already evidence: the router only puts the
        selected, DYNAMIC-confirmed opponent on /tracking/dynamic_obstacles, and
        stamps it with logical_opponent_id. An UNKNOWN or unselected track never
        reaches here, so there is no separate classification test to run.
        """
        speed = float(obstacle.vs)
        minimum = float(self.get_parameter(
            'fallback_min_opponent_speed_mps').value)
        if speed < minimum:
            return False, 'CONSTVEL_OPPONENT_TOO_SLOW', {
                'opponent_vs': round(speed, 3),
                'limit_mps': minimum,
            }

        if self.ego_vs is None:
            return False, 'CONSTVEL_NO_EGO_SPEED', {}

        # ONE: not falling behind right now. Narrow on purpose - this is the
        # only thing the measured speed can say while trailing is holding it
        # at the opponent's. It catches the ego crawling round a static
        # obstacle at the static planner's max_speed_mps, which the raceline
        # test below cannot see.
        closing = float(self.ego_vs) - speed
        max_losing = float(self.get_parameter('fallback_max_losing_mps').value)
        if closing < -max_losing:
            return False, 'CONSTVEL_NOT_CLOSING', {
                'closing_mps': round(closing, 3),
                'limit_mps': round(-max_losing, 3),
            }

        # TWO: a pass is completable at all. Against the scaled raceline,
        # because trailing suppresses the measured speed to the opponent's by
        # design and no positive bar on that difference can ever be met.
        raceline_speed = self._raceline_speed_at(self.ego_s) \
            if self.ego_s is not None else None
        if raceline_speed is None:
            # Fall back to the measured speed, which is never higher than the
            # raceline's - the conservative direction.
            raceline_speed = float(self.ego_vs)
        advantage = raceline_speed - speed
        min_advantage = float(
            self.get_parameter('fallback_min_speed_advantage_mps').value)
        if advantage < min_advantage:
            return False, 'CONSTVEL_NO_SPEED_ADVANTAGE', {
                'advantage_mps': round(advantage, 3),
                'raceline_mps': round(raceline_speed, 3),
                'opponent_vs': round(speed, 3),
                'limit_mps': min_advantage,
            }

        inside, _, detail = profile_inside_track(
            profile_s,
            profile_d,
            self.global_msg.wpnts if self.global_msg is not None else [],
            self.track_length,
            self.get_parameter('opponent_width').value,
            self.get_parameter('trajectory_boundary_margin').value,
            'CONSTVEL_OUT_OF_BOUNDS',
        )
        if not inside:
            return False, 'CONSTVEL_OUT_OF_BOUNDS', detail or {}

        return True, 'CONSTVEL_READY', {
            'obstacle_id': int(obstacle.id),
            'opponent_vs': round(speed, 3),
            'closing_mps': round(closing, 3),
            'horizon_s': round(
                int(self.get_parameter('n_time_steps').value)
                * float(self.get_parameter('dt').value), 2),
        }

    def _reset_learned_gate(self):
        self._learned_gate_open = False
        self._learned_ready_count = 0
        self._learned_reject_count = 0

    def _learned_ready(self, obstacle):
        """Compatibility wrapper retained for callers and unit tests."""
        return self._learned_status(obstacle)[0]

    def _fallback_profile(self, obstacle, count, dt):
        speed = max(0.0, float(obstacle.vs))
        steps = np.arange(count, dtype=float)
        s = obstacle.s_center + speed * dt * steps
        center_s = np.asarray([w.s_m for w in self.center_msg.wpnts])
        center_d = np.asarray([w.d_m for w in self.center_msg.wpnts])
        target_d = periodic_interp(s, center_s, center_d, self.track_length)
        weight = np.linspace(0.0, 1.0, count)
        d = (1.0 - weight) * obstacle.d_center + weight * target_d
        return s, d, np.full(count, speed), np.zeros(count)

    def _learned_profile(self, obstacle, count, dt):
        trajectory = self.trajectory.oppwpnts
        traj_s = np.asarray([w.s_m for w in trajectory])
        traj_d = np.asarray([w.d_m for w in trajectory])
        traj_vs = np.asarray([w.proj_vs_mps for w in trajectory])
        traj_vd = np.asarray([w.vd_mps for w in trajectory])
        s = np.zeros(count)
        d = np.zeros(count)
        speed = np.zeros(count)
        vd = np.zeros(count)
        current = float(obstacle.s_center)
        offset = float(self.get_parameter('speed_offset').value)
        for index in range(count):
            speed[index] = max(0.0, float(periodic_interp(
                [current], traj_s, traj_vs, self.track_length)[0]) + offset)
            d[index] = float(periodic_interp(
                [current], traj_s, traj_d, self.track_length)[0])
            vd[index] = float(periodic_interp(
                [current], traj_s, traj_vd, self.track_length)[0])
            s[index] = current
            current += speed[index] * dt
        return s, d, speed, vd

    def _validate_learned_profile(self, s, d):
        return validate_learned_profile(
            s,
            d,
            self.trajectory.oppwpnts if self.trajectory is not None else [],
            self.global_msg.wpnts if self.global_msg is not None else [],
            self.track_length,
            self.get_parameter('opponent_width').value,
            self.get_parameter('trajectory_boundary_margin').value,
            self.get_parameter('max_trajectory_query_gap').value,
            self.get_parameter('max_trajectory_d_variance').value,
        )

    def _loop(self):
        required_inputs = {
            'ego_s': self.ego_s,
            'track_length': self.track_length,
            'converter': self.converter,
            'global_waypoints_updated': self.updated_msg,
            'centerline_waypoints': self.center_msg,
        }
        missing = [
            name for name, value in required_inputs.items()
            if value is None
        ]
        if missing:
            self._set_diagnostic('NOT_READY', {'missing': missing})
            self._publish_empty()
            return
        obstacle_timeout = float(self.get_parameter('obstacle_timeout').value)
        obstacle_age = self._age(self.obstacles_received_at)
        if obstacle_age > obstacle_timeout:
            detail = {'limit_s': obstacle_timeout}
            if np.isfinite(obstacle_age):
                detail['age_s'] = round(obstacle_age, 3)
            else:
                detail['received'] = False
            self._set_diagnostic('OBSTACLE_STALE', detail)
            self._publish_empty()
            return
        obstacle = nearest_dynamic(
            self.obstacles.obstacles,
            self.ego_s,
            self.track_length,
            float(self.get_parameter('max_opponent_distance').value),
        )
        if obstacle is None:
            self._set_diagnostic('NO_DYNAMIC_OBSTACLE')
            self._publish_empty()
            return

        count = max(2, int(self.get_parameter('n_time_steps').value))
        dt = max(0.01, float(self.get_parameter('dt').value))
        learned, status, detail = self._learned_status(obstacle)
        if learned:
            s, d, speed, vd = self._learned_profile(obstacle, count, dt)
            valid, invalid_status, invalid_detail = self._validate_learned_profile(s, d)
            if not valid:
                learned = False
                status = invalid_status
                detail = invalid_detail
                self._reset_learned_gate()
        authorized = learned
        if not learned:
            s, d, speed, vd = self._fallback_profile(obstacle, count, dt)
            if bool(self.get_parameter('authorize_on_fallback').value):
                # Report the constant-velocity verdict rather than the learned
                # gate's reason: with the GP chain not launched that reason is
                # permanently NO_TRAJECTORY, which says nothing about whether
                # this particular overtake is allowed.
                authorized, status, detail = self._fallback_authorized(
                    obstacle, s, d)
        self._set_diagnostic(status, detail)
        self._publish_prediction(
            obstacle, s, d, speed, vd, dt, learned, authorized)

    def _publish_prediction(self, source, s, d, speed, vd, dt, learned,
                            authorized=None):
        # authorized defaults to learned so the original contract - and every
        # caller that predates constant-velocity mode - is unchanged.
        if authorized is None:
            authorized = learned
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
        xy = self.converter.get_cartesian(s % self.track_length, d).T
        reference_s = np.asarray([w.s_m for w in self.updated_msg.wpnts])
        reference_psi = np.asarray([w.psi_rad for w in self.updated_msg.wpnts])
        half_width = max(
            0.5 * float(self.get_parameter('opponent_width').value),
            0.5 * max(0.0, float(source.d_left - source.d_right)),
        )
        obstacle_array = ObstacleArray(header=header)
        prediction_array = PredictionArray(header=header, id=source.id, dt=float(dt))
        for index in range(len(s)):
            previous_s = s[index - 1] if index else s[index]
            predicted = Obstacle()
            predicted.id = int(source.id)
            predicted.s_start = float(previous_s % self.track_length)
            predicted.s_end = float(s[index] % self.track_length)
            predicted.s_center = float(s[index] % self.track_length)
            predicted.d_center = float(d[index])
            predicted.d_left = float(d[index] + half_width)
            predicted.d_right = float(d[index] - half_width)
            predicted.x_m = float(xy[index, 0])
            predicted.y_m = float(xy[index, 1])
            predicted.size = float(source.size)
            predicted.vs = float(speed[index])
            predicted.vd = float(vd[index])
            predicted.is_static = False
            predicted.is_visible = bool(source.is_visible)
            obstacle_array.obstacles.append(predicted)

            ref_index = int(np.argmin(np.abs(reference_s - (s[index] % self.track_length))))
            prediction_array.predictions.append(Prediction(
                header=header,
                id=index,
                pred_x=float(xy[index, 0]),
                pred_y=float(xy[index, 1]),
                pred_yaw=float(reference_psi[ref_index]),
                pred_vx=float(speed[index]),
                pred_s=float(s[index] % self.track_length),
                pred_d=float(d[index]),
                pred_epsi=0.0,
                pred_vs=float(speed[index]),
            ))
        self.obstacle_pub.publish(obstacle_array)
        self.prediction_pub.publish(prediction_array)
        self.force_trailing_pub.publish(Bool(data=not authorized))
        # Three states now, so the marker has to distinguish them: red is a
        # learned prediction, amber a constant-velocity one that is authorized
        # to pass, green one that is only good enough to trail behind.
        if learned:
            namespace, colour = 'learned_prediction', (1.0, 0.0, 0.0)
        elif authorized:
            namespace, colour = 'constvel_prediction', (1.0, 0.65, 0.0)
        else:
            namespace, colour = 'fallback_prediction', (0.0, 1.0, 0.0)
        self.marker_pub.publish(trajectory_markers(
            header, [(x, y, 0.15) for x, y in xy], namespace, colour))

    def _publish_empty(self):
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
        self.obstacle_pub.publish(ObstacleArray(header=header))
        self.prediction_pub.publish(PredictionArray(header=header))
        self.force_trailing_pub.publish(Bool(data=True))
        self.marker_pub.publish(trajectory_markers(
            header, [], 'opponent_prediction', (0.0, 1.0, 0.0)))


def main(args=None):
    rclpy.init(args=args)
    node = OpponentPredictor()
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
