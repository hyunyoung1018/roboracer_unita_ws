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


DIAGNOSTIC_DESCRIPTIONS = {
    "predictor": {
        "NOT_READY": "predictor 입력 준비 대기",
        "OBSTACLE_STALE": "동적 장애물 입력이 없거나 만료됨 (force_trailing)",
        "NO_DYNAMIC_OBSTACLE": "추적 중인 동적 장애물 없음 (force_trailing)",
        "NO_TRAJECTORY": "학습된 상대 궤적 없음 (force_trailing)",
        "TRAJECTORY_STALE": "상대 궤적이 만료됨 (force_trailing)",
        "TRAINING": "상대 궤적 학습 랩 수 부족 (force_trailing)",
        "OFF_TRAJECTORY": "상대가 학습 궤도를 벗어남 (force_trailing)",
        "DEVIATION_TOO_LARGE": "현재 상대 위치와 학습 궤적 편차가 큼 (force_trailing)",
        "INVALID_LEARNED_TRAJECTORY": "학습 예측값이 유효하지 않음 (force_trailing)",
        "TRAJECTORY_UNOBSERVED": "관측되지 않은 구간으로 예측이 확장됨 (force_trailing)",
        "TRAJECTORY_UNCERTAIN": "학습 예측 불확실도가 큼 (force_trailing)",
        "TRAJECTORY_OUT_OF_BOUNDS": "학습 예측이 트랙 경계를 벗어남 (force_trailing)",
        "LEARNED_CONFIRMING": "학습 예측 사용 조건 확인 중 (force_trailing)",
        "LEARNED_READY": "학습 예측 사용 가능 (force_trailing 해제)",
        "CONSTVEL_READY": "등속 예측으로 추월 허가 (force_trailing 해제)",
        "CONSTVEL_OPPONENT_TOO_SLOW": "상대가 너무 느림, 정적 회피 대상 (force_trailing)",
        "CONSTVEL_NOT_CLOSING": "상대에게 접근하고 있지 않음 (force_trailing)",
        "CONSTVEL_NO_EGO_SPEED": "자차 속도 미수신 (force_trailing)",
        "CONSTVEL_OUT_OF_BOUNDS": "등속 예측이 트랙 경계를 벗어남 (force_trailing)",
    },
    "planner": {
        "NOT_READY": "planner 입력 준비 대기",
        "PREDICTION_MISSING": "예측 경로 없음",
        "PREDICTION_STALE": "예측 경로 timestamp 만료",
        "FORCE_TRAILING": "predictor가 추월 경로 생성을 금지함",
        "NO_DYNAMIC_OBSTACLE": "lookahead 안에 회피할 동적 장애물 없음",
        "PREDICTION_ID_MISMATCH": "현재 장애물 ID와 예측 ID 불일치",
        "NO_SAFE_SIDE": "좌우 어느 쪽도 필요한 여유 폭을 만족하지 못함",
        "SIDE_SWITCH_PENDING": "안전한 추월 방향 전환 확인 중",
        "PATH_TOO_SHORT": "생성 가능한 회피 경로 길이가 너무 짧음",
        "START_OFFSET_TOO_LARGE": "차량 현재 위치와 회피 경로 시작점 차이가 큼",
        "INVALID_RAW_PATH": "원본 회피 경로 계산 결과가 유효하지 않음",
        "GRID_FILTER_NOT_READY": "GridFilter가 아직 맵을 받지 못함",
        "RAW_GRID_REJECTED": "GridFilter가 원본 회피 경로를 거부함",
        "INVALID_SMOOTHED_PATH": "평활화된 회피 경로가 너무 짧음",
        "SMOOTHED_GRID_REJECTED": "GridFilter가 평활화된 회피 경로를 거부함",
        "PATH_READY": "동적 장애물 추월 경로 생성 완료",
    },
}


def diagnostic_description(source, status):
    """Return a stable human-readable description for a diagnostic state."""
    return DIAGNOSTIC_DESCRIPTIONS.get(source, {}).get(
        status, "알 수 없는 진단 상태")


def diagnostic_detail_text(detail):
    """Format the small structured payload without hiding useful values."""
    if not isinstance(detail, dict) or not detail:
        return ""
    return ", ".join(
        f"{key}={value}" for key, value in detail.items())


class DrivingModeMonitor(Node):
    """Monitor state-machine output in a quiet, dedicated terminal."""

    def __init__(self):
        super().__init__("driving_mode_monitor")

        self.last_mode = None
        self.obstacles = []
        self.stable_classes = {}
        self.current_s = None
        self.track_length = None
        self.last_diagnostic_status = {}

        self.declare_parameter(
            "obstacle_topic", "/tracking/stable_obstacles")
        self.declare_parameter(
            "classification_debug_topic", "/tracking/classification_debug")
        self.declare_parameter(
            "prediction_diagnostic_topic", "/opponent_prediction/diagnostics")
        self.declare_parameter(
            "planner_diagnostic_topic", "/planner/avoidance/diagnostics")

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
        # Standard volatile QoS is intentional; diagnostics are live events.
        self.create_subscription(
            String,
            str(self.get_parameter("prediction_diagnostic_topic").value),
            self.diagnostic_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("planner_diagnostic_topic").value),
            self.diagnostic_cb,
            10,
        )

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

    def diagnostic_cb(self, msg):
        """Print a gate reason once, then wait for its state to change."""
        try:
            payload = json.loads(msg.data)
            source = str(payload["source"])
            status = str(payload["status"])
            detail = payload.get("detail", {})
        except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            self.get_logger().warn(
                f"invalid dynamic diagnostic message: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        if self.last_diagnostic_status.get(source) == status:
            return
        self.last_diagnostic_status[source] = status

        description = diagnostic_description(source, status)
        detail_text = diagnostic_detail_text(detail)
        message = f"[DYNAMIC_DIAG][{source.upper()}] {description} ({status})"
        if detail_text:
            message += f": {detail_text}"
        if status in ("LEARNED_READY", "PATH_READY"):
            self.get_logger().info(message)
        else:
            self.get_logger().warn(message)

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
