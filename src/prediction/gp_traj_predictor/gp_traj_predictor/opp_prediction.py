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


class OpponentPredictor(Node):
    """Publish a learned prediction only after a complete, consistent trajectory."""

    def __init__(self):
        super().__init__('opp_prediction')
        defaults = {
            'loop_rate': 20.0,
            'n_time_steps': 20,
            'dt': 0.10,
            'max_opponent_distance': 30.0,
            'obstacle_timeout': 0.5,
            'trajectory_timeout': 2.0,
            'min_training_laps': 1.0,
            'learned_deviation_enter_threshold': 0.35,
            'learned_deviation_exit_threshold': 0.55,
            'learned_ready_confirm_frames': 3,
            'learned_reject_confirm_frames': 5,
            'opponent_width': 0.28,
            'speed_offset': 0.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.ego_s = None
        self.global_msg = None
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

    def _global_cb(self, msg):
        if not msg.wpnts:
            return
        self.global_msg = msg
        self.track_length = float(msg.wpnts[-1].s_m)
        x = np.asarray([w.x_m for w in msg.wpnts], dtype=float)
        y = np.asarray([w.y_m for w in msg.wpnts], dtype=float)
        psi = np.asarray([w.psi_rad for w in msg.wpnts], dtype=float)
        self.converter = FrenetConverter(x, y, psi)

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
        self._set_diagnostic(status, detail)
        if learned:
            s, d, speed, vd = self._learned_profile(obstacle, count, dt)
        else:
            s, d, speed, vd = self._fallback_profile(obstacle, count, dt)
        self._publish_prediction(obstacle, s, d, speed, vd, dt, learned)

    def _publish_prediction(self, source, s, d, speed, vd, dt, learned):
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
        self.force_trailing_pub.publish(Bool(data=not learned))
        self.marker_pub.publish(trajectory_markers(
            header,
            [(x, y, 0.15) for x, y in xy],
            'learned_prediction' if learned else 'fallback_prediction',
            (1.0, 0.0, 0.0) if learned else (0.0, 1.0, 0.0),
        ))

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
