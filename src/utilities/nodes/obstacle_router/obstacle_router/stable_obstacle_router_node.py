#!/usr/bin/env python3
"""Head-to-head-only stable static/dynamic obstacle router."""

from copy import deepcopy
import json

import rclpy
from f110_msgs.msg import ObstacleArray, WpntArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from .stable_classifier import StableObstacleClassifier, output_membership


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

    def obstacles_cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        static_msg = ObstacleArray(header=msg.header)
        dynamic_msg = ObstacleArray(header=msg.header)
        stable_msg = ObstacleArray(header=msg.header)
        debug_records = []

        for obstacle in msg.obstacles:
            history = self.classifier.update(
                obstacle_id=obstacle.id,
                s=obstacle.s_center,
                d=obstacle.d_center,
                is_visible=obstacle.is_visible,
                now=now,
            )
            stable_obstacle = deepcopy(obstacle)
            stable_obstacle.is_static = history.stable_class == "STATIC"
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
