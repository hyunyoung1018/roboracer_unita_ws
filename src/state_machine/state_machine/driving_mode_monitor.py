#!/usr/bin/env python3
"""Print the current high-level driving mode only when it changes.

This executable is used for head-to-head diagnosis.  ``Obstacle.is_static``
cannot represent the router's third state (UNKNOWN): both UNKNOWN and DYNAMIC
are encoded as ``False`` on the safety-facing obstacle stream.  Consume the
router's debug record as well so an unclassified/reacquired object is not
reported as a moving opponent.
"""

import json

import rclpy
from rclpy.node import Node

from f110_msgs.msg import BehaviorStrategy, ObstacleArray, WpntArray
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def obstacle_mode_for_class(prefix, stable_class):
    """Translate the router's three-state class without collapsing UNKNOWN."""
    action = "추월" if prefix == "OVERTAKE" else "트레일링"
    if stable_class == "UNKNOWN":
        return (
            f"OBSTACLE_{prefix}_UNKNOWN",
            f"장애물 {action} (정적/동적 분류 확인 중)",
        )
    if stable_class == "STATIC":
        if prefix == "OVERTAKE":
            return "STATIC_OBSTACLE_AVOIDANCE", "정적 장애물 회피"
        return "STATIC_OBSTACLE_TRAILING", "정적 장애물 트레일링"
    if stable_class == "DYNAMIC" and prefix == "OVERTAKE":
        return "DYNAMIC_OBSTACLE_OVERTAKE", "동적 장애물 추월"
    if stable_class == "DYNAMIC":
        return "DYNAMIC_OBSTACLE_TRAILING", "동적 장애물 트레일링"
    return f"OBSTACLE_{prefix}_UNKNOWN", f"장애물 {action} (분류값 오류)"


class DrivingModeMonitor(Node):
    """Monitor state-machine output in a quiet, dedicated terminal."""

    def __init__(self):
        super().__init__("driving_mode_monitor")

        self.last_mode = None
        self.obstacles = []
        self.stable_classes = {}
        self.current_s = None
        self.track_length = None

        self.declare_parameter(
            "obstacle_topic", "/tracking/stable_obstacles")
        self.declare_parameter(
            "classification_debug_topic", "/tracking/classification_debug")

        self.create_subscription(
            BehaviorStrategy, "/behavior_strategy", self.behavior_cb, 10)
        self.create_subscription(
            ObstacleArray,
            str(self.get_parameter("obstacle_topic").value),
            self.obstacles_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("classification_debug_topic").value),
            self.classification_debug_cb,
            10,
        )
        self.create_subscription(
            Odometry, "/car_state/odom_frenet", self.odom_frenet_cb, 10)
        self.create_subscription(
            WpntArray, "/global_waypoints_scaled", self.waypoints_cb, 10)

    def obstacles_cb(self, msg):
        self.obstacles = list(msg.obstacles)

    def classification_debug_cb(self, msg):
        """Cache the router's real three-state classification by tracker ID."""
        try:
            payload = json.loads(msg.data)
            records = payload.get("obstacles", [])
            self.stable_classes = {
                int(record["id"]): str(record["stable_class"])
                for record in records
                if "id" in record and "stable_class" in record
            }
        except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            self.get_logger().warn(
                f"invalid classification_debug message: {exc}",
                throttle_duration_sec=2.0,
            )

    def odom_frenet_cb(self, msg):
        self.current_s = float(msg.pose.pose.position.x)

    def waypoints_cb(self, msg):
        if msg.wpnts:
            # The final global waypoint carries the closed-loop track length.
            self.track_length = float(msg.wpnts[-1].s_m)

    def closest_obstacle(self):
        """Return closest tracked obstacle when targets are omitted."""
        if not self.obstacles:
            return None
        if (
            self.current_s is None
            or not self.track_length
            or self.track_length <= 0.0
        ):
            return self.obstacles[0]

        return min(
            self.obstacles,
            key=lambda obs: (
                float(obs.s_start) - self.current_s
            ) % self.track_length,
        )

    def obstacle_mode(self, prefix, target):
        if target is None:
            action = "추월" if prefix == "OVERTAKE" else "트레일링"
            return (
                f"OBSTACLE_{prefix}_UNKNOWN",
                f"장애물 {action} (종류 확인 중)",
            )

        # An ID absent from classification_debug has not yet been confirmed by
        # the stable router.  In particular, is_static=False alone must never
        # be called DYNAMIC because UNKNOWN uses the same wire representation.
        stable_class = self.stable_classes.get(int(target.id), "UNKNOWN")
        return obstacle_mode_for_class(prefix, stable_class)

    def classify(self, msg):
        state = msg.state
        if state == "RACELINE":
            return "NORMAL_DRIVING", "일반 주행"

        if state == "OVERTAKE":
            target = msg.overtaking_targets[0] if msg.overtaking_targets \
                else self.closest_obstacle()
            return self.obstacle_mode("OVERTAKE", target)

        if state in ("TRAILING", "ATTACK"):
            target = msg.trailing_targets[0] if msg.trailing_targets \
                else self.closest_obstacle()
            return self.obstacle_mode("TRAILING", target)

        modes = {
            "RECOVERY": ("RECOVERY", "레이싱라인 복귀"),
            "FTGONLY": ("FTG_ONLY", "FTG 긴급 회피"),
            "START": ("START", "스타트 주행"),
            "LOSTLINE": ("LOST_LINE", "레이싱라인 이탈"),
        }
        return modes.get(state, (state or "UNKNOWN", state or "상태 확인 중"))

    def behavior_cb(self, msg):
        mode, description = self.classify(msg)
        if mode == self.last_mode:
            return

        self.last_mode = mode
        self.get_logger().info(f"[DRIVING_MODE] {description} ({mode})")


def main(args=None):
    rclpy.init(args=args)
    node = DrivingModeMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
