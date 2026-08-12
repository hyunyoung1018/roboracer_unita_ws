#!/usr/bin/env python3
"""Head-to-head-only stable static/dynamic obstacle router."""

from copy import deepcopy
import json
import math

import rclpy
from f110_msgs.msg import ObstacleArray, WpntArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from .stable_classifier import (
    DYNAMIC,
    STATIC,
    StableObstacleClassifier,
    circular_delta,
    output_membership,
)


class StableObstacleRouter(Node):
    """Stabilize classifications without changing the legacy tracker."""

    PARAM_DEFAULTS = {
        # Provisional UNITA values. Tune from classification_debug on-car;
        # these are not copied from UNIST's 40 Hz vehicle settings.
        "min_std": 0.02,
        "max_std": 0.04,
        "min_nb_meas": 8,
        "history_size": 20,
        "static_confirm_count": 3,
        "dynamic_confirm_count": 2,
        "track_timeout_sec": 1.0,
        "single_dynamic_opponent": True,
        "dynamic_reid_timeout_sec": 1.0,
        "dynamic_reid_max_distance_m": 1.2,
        "dynamic_speed_hold_sec": 0.75,
        "dynamic_speed_valid_min_mps": 0.15,
        "debug": True,
    }

    def __init__(self):
        super().__init__("stable_obstacle_router")
        for name, default in self.PARAM_DEFAULTS.items():
            self.declare_parameter(name, default)

        self.classifier = StableObstacleClassifier(
            min_std=float(self.get_parameter("min_std").value),
            max_std=float(self.get_parameter("max_std").value),
            min_nb_meas=int(self.get_parameter("min_nb_meas").value),
            history_size=int(self.get_parameter("history_size").value),
            static_confirm_count=int(
                self.get_parameter("static_confirm_count").value),
            dynamic_confirm_count=int(
                self.get_parameter("dynamic_confirm_count").value),
            track_timeout_sec=float(
                self.get_parameter("track_timeout_sec").value),
        )
        self.debug_enabled = bool(self.get_parameter("debug").value)
        self.single_dynamic_opponent = bool(
            self.get_parameter("single_dynamic_opponent").value)
        self.dynamic_reid_timeout_sec = float(
            self.get_parameter("dynamic_reid_timeout_sec").value)
        self.dynamic_reid_max_distance_m = float(
            self.get_parameter("dynamic_reid_max_distance_m").value)
        self.dynamic_speed_hold_sec = float(
            self.get_parameter("dynamic_speed_hold_sec").value)
        self.dynamic_speed_valid_min_mps = float(
            self.get_parameter("dynamic_speed_valid_min_mps").value)

        self.active_dynamic_id = None
        self.last_dynamic_s = None
        self.last_dynamic_d = None
        self.last_dynamic_position_at = None
        self.last_dynamic_speed = None
        self.last_dynamic_speed_at = None
        self.retired_dynamic_ids = {}

        self.static_pub = self.create_publisher(
            ObstacleArray, "/tracking/static_obstacles", 10)
        self.dynamic_pub = self.create_publisher(
            ObstacleArray, "/tracking/dynamic_obstacles", 10)
        self.stable_pub = self.create_publisher(
            ObstacleArray, "/tracking/stable_obstacles", 10)
        self.debug_pub = self.create_publisher(
            String, "/tracking/classification_debug", 10)

        self.create_subscription(
            ObstacleArray, "/tracking/obstacles", self.obstacles_cb, 10)
        self.create_subscription(
            WpntArray, "/global_waypoints_scaled", self.waypoints_cb, 10)

    def waypoints_cb(self, msg):
        if msg.wpnts:
            self.classifier.track_length = float(msg.wpnts[-1].s_m)

    def _projected_dynamic_position(self, now, obstacles_by_id):
        active = obstacles_by_id.get(self.active_dynamic_id)
        if active is not None:
            return float(active.s_center), float(active.d_center)
        if self.last_dynamic_s is None or self.last_dynamic_d is None:
            return None
        elapsed = max(0.0, now - self.last_dynamic_position_at)
        speed = self.last_dynamic_speed or 0.0
        track_length = self.classifier.track_length
        projected_s = self.last_dynamic_s + speed * elapsed
        if track_length:
            projected_s %= track_length
        return projected_s, self.last_dynamic_d

    def _maybe_handoff_dynamic_id(self, obstacles, now):
        """Transfer the one opponent's class when the tracker reacquires it."""
        if not self.single_dynamic_opponent or self.active_dynamic_id is None:
            return None
        obstacles_by_id = {int(obs.id): obs for obs in obstacles}
        active = obstacles_by_id.get(self.active_dynamic_id)
        if active is not None and active.is_visible:
            return None
        reference = self._projected_dynamic_position(now, obstacles_by_id)
        if reference is None or self.last_dynamic_position_at is None:
            return None
        if now - self.last_dynamic_position_at > self.dynamic_reid_timeout_sec:
            return None

        candidates = []
        for obstacle in obstacles:
            obstacle_id = int(obstacle.id)
            if obstacle_id == self.active_dynamic_id or not obstacle.is_visible:
                continue
            history = self.classifier.tracks.get(obstacle_id)
            if history is not None and history.stable_class == STATIC:
                continue
            ds = circular_delta(
                float(obstacle.s_center), reference[0],
                self.classifier.track_length)
            dd = float(obstacle.d_center) - reference[1]
            candidates.append((math.hypot(ds, dd), obstacle_id))
        if not candidates:
            return None

        distance, new_id = min(candidates)
        if distance > self.dynamic_reid_max_distance_m:
            return None
        old_id = self.active_dynamic_id
        if self.classifier.transfer_track(old_id, new_id) is None:
            return None
        self.active_dynamic_id = new_id
        self.retired_dynamic_ids[old_id] = now + self.dynamic_reid_timeout_sec
        self.retired_dynamic_ids.pop(new_id, None)
        self.get_logger().warn(
            f'dynamic opponent tracker ID changed {old_id} -> {new_id}; '
            f'preserving DYNAMIC class and speed '
            f'(re-id distance {distance:.2f} m)')
        return new_id

    def _stabilize_dynamic_speed(self, obstacle, now):
        measured_speed = float(obstacle.vs)
        speed_is_valid = (
            math.isfinite(measured_speed)
            and measured_speed >= self.dynamic_speed_valid_min_mps
        )
        if obstacle.is_visible and speed_is_valid:
            self.last_dynamic_speed = measured_speed
            self.last_dynamic_speed_at = now
        elif (
            self.last_dynamic_speed is not None
            and self.last_dynamic_speed_at is not None
            and now - self.last_dynamic_speed_at <= self.dynamic_speed_hold_sec
        ):
            obstacle.vs = float(self.last_dynamic_speed)

        self.last_dynamic_s = float(obstacle.s_center)
        self.last_dynamic_d = float(obstacle.d_center)
        self.last_dynamic_position_at = now

    def obstacles_cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.retired_dynamic_ids = {
            obstacle_id: expires_at
            for obstacle_id, expires_at in self.retired_dynamic_ids.items()
            if expires_at >= now
        }
        handed_off_id = self._maybe_handoff_dynamic_id(msg.obstacles, now)
        static_msg = ObstacleArray(header=msg.header)
        dynamic_msg = ObstacleArray(header=msg.header)
        stable_msg = ObstacleArray(header=msg.header)
        debug_records = []

        for obstacle in msg.obstacles:
            obstacle_id = int(obstacle.id)
            if obstacle_id in self.retired_dynamic_ids:
                continue
            history = self.classifier.update(
                obstacle_id=obstacle.id,
                s=obstacle.s_center,
                d=obstacle.d_center,
                is_visible=obstacle.is_visible,
                now=now,
            )
            stable_obstacle = deepcopy(obstacle)
            stable_obstacle.is_static = history.stable_class == "STATIC"
            if history.stable_class == DYNAMIC:
                if self.active_dynamic_id is None:
                    self.active_dynamic_id = obstacle_id
                if obstacle_id == self.active_dynamic_id:
                    self._stabilize_dynamic_speed(stable_obstacle, now)
            stable_msg.obstacles.append(stable_obstacle)

            in_static, in_dynamic = output_membership(history.stable_class)
            if in_static:
                static_msg.obstacles.append(deepcopy(stable_obstacle))
            elif in_dynamic:
                dynamic_msg.obstacles.append(deepcopy(stable_obstacle))

            debug_records.append({
                "id": int(obstacle.id),
                "raw_is_static": bool(obstacle.is_static),
                "stable_class": history.stable_class,
                "std_s": round(float(history.std_s), 6),
                "std_d": round(float(history.std_d), 6),
                "sample_count": int(history.sample_count),
                "static_evidence": int(history.static_evidence),
                "dynamic_evidence": int(history.dynamic_evidence),
                "is_visible": bool(obstacle.is_visible),
                "routed_vs": round(float(stable_obstacle.vs), 4),
                "reidentified": obstacle_id == handed_off_id,
            })

        self.classifier.remove_stale(now)
        self.static_pub.publish(static_msg)
        self.dynamic_pub.publish(dynamic_msg)
        self.stable_pub.publish(stable_msg)
        if self.debug_enabled:
            self.debug_pub.publish(String(data=json.dumps(
                {"obstacles": debug_records},
                separators=(",", ":"),
                sort_keys=True,
            )))


def main(args=None):
    rclpy.init(args=args)
    node = StableObstacleRouter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
