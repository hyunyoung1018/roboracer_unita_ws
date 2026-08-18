#!/usr/bin/env python3
"""Head-to-head-only stable static/dynamic obstacle router."""

from copy import deepcopy
import json
import math

import rclpy
from f110_msgs.msg import ObstacleArray, WpntArray
from nav_msgs.msg import Odometry
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
from .opponent_selector import (
    initial_candidate_key,
    inside_forward_window,
    inside_opponent_corridor,
    unique_reidentification_candidate,
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
        "dynamic_reid_max_lateral_m": 0.25,
        "dynamic_reid_ambiguity_margin_m": 0.20,
        "dynamic_speed_hold_sec": 0.75,
        "dynamic_speed_valid_min_mps": 0.15,
        "logical_opponent_id": 1000000,
        "opponent_width_m": 0.28,
        "opponent_boundary_margin_m": 0.03,
        "opponent_forward_min_m": 0.2,
        "opponent_forward_max_m": 8.0,
        # [m] Rear allowance for the opponent the router has ALREADY locked
        # onto. Signed, so a negative value is behind the ego car.
        #
        # The forward window exists so that a person standing behind the car
        # cannot become the opponent, and 0.2 m is right for acquiring one. It
        # is wrong for keeping one. During an overtake the opponent's forward
        # gap falls through 0.2 m and goes negative, it drops out of the
        # candidate list, /tracking/dynamic_obstacles empties, opp_prediction
        # reports NO_DYNAMIC_OBSTACLE and raises force_trailing, and the
        # lane-change planner withdraws its path - at the exact moment the car
        # is alongside and most needs it.
        #
        # -1.5 m covers the car's own length plus the opponent's. Beyond that
        # the opponent really is behind and released normally. This applies
        # only to the already-selected target, so acquisition is unchanged.
        "opponent_active_rear_m": -1.5,
        # Minimum observations before an UNKNOWN track may be published on
        # /tracking/stable_obstacles. See the include_stable comment.
        "min_unknown_samples": 4,
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
        self.dynamic_reid_max_lateral_m = float(
            self.get_parameter("dynamic_reid_max_lateral_m").value)
        self.dynamic_reid_ambiguity_margin_m = float(
            self.get_parameter("dynamic_reid_ambiguity_margin_m").value)
        self.dynamic_speed_hold_sec = float(
            self.get_parameter("dynamic_speed_hold_sec").value)
        self.dynamic_speed_valid_min_mps = float(
            self.get_parameter("dynamic_speed_valid_min_mps").value)
        self.logical_opponent_id = int(
            self.get_parameter("logical_opponent_id").value)
        self.opponent_width_m = float(
            self.get_parameter("opponent_width_m").value)
        self.opponent_boundary_margin_m = float(
            self.get_parameter("opponent_boundary_margin_m").value)
        self.opponent_forward_min_m = float(
            self.get_parameter("opponent_forward_min_m").value)
        self.opponent_forward_max_m = float(
            self.get_parameter("opponent_forward_max_m").value)
        self.opponent_active_rear_m = float(
            self.get_parameter("opponent_active_rear_m").value)
        self.min_unknown_samples = int(
            self.get_parameter("min_unknown_samples").value)

        self.active_dynamic_id = None
        self.ego_s = None
        self.waypoints = []
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
        self.create_subscription(
            Odometry, "/car_state/odom_frenet", self.odom_cb, 10)

    def waypoints_cb(self, msg):
        if msg.wpnts:
            self.classifier.track_length = float(msg.wpnts[-1].s_m)
            self.waypoints = list(msg.wpnts)

    def odom_cb(self, msg):
        self.ego_s = float(msg.pose.pose.position.x)

    def _inside_corridor(self, obstacle):
        return inside_opponent_corridor(
            obstacle,
            self.waypoints,
            self.classifier.track_length,
            self.opponent_width_m,
            self.opponent_boundary_margin_m,
        )

    def _inside_forward_window(self, obstacle):
        """Acquisition window, widened rearwards for the locked target.

        Keeping the opponent through the moment the car draws level with it is
        a different question from picking one out in the first place, so the
        two get different minimums.
        """
        minimum = self.opponent_forward_min_m
        if (
            self.active_dynamic_id is not None
            and int(obstacle.id) == self.active_dynamic_id
        ):
            minimum = min(minimum, self.opponent_active_rear_m)
        return inside_forward_window(
            obstacle,
            self.ego_s,
            self.classifier.track_length,
            minimum,
            self.opponent_forward_max_m,
        )

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
            # An ID this router just handed the opponent AWAY from may not be
            # handed it straight back. The retirement was already being written
            # on every handoff, and obstacles_cb's classification loop already
            # honours it - but this loop, which runs first and is the only
            # thing that actually moves the lock, did not read it.
            #
            # The consequence on the car was an ID ping-pong: 7->8->7->8 with
            # gaps as short as one detection period, 39 handoffs and 67 tracker
            # IDs created in 75 seconds. Every flip wrote the OTHER cluster's
            # s into last_dynamic_s, so the next sample went backwards, and
            # opponent_trajectory rejected it - 35 BACKWARD_JUMPs, which is why
            # `traversed` never accumulated, lap_count never reached
            # min_training_laps, and the predictor sat in TRAINING with
            # force_trailing raised for the whole run. No avoidance path was
            # ever planned.
            if obstacle_id in self.retired_dynamic_ids:
                continue
            if not self._inside_corridor(obstacle):
                continue
            if not self._inside_forward_window(obstacle):
                continue
            history = self.classifier.tracks.get(obstacle_id)
            if history is not None and history.stable_class == STATIC:
                continue
            # A track that independently accumulated enough evidence to become
            # DYNAMIC is a second moving object (for example the operator), not
            # a just-created replacement ID for the locked opponent.
            if history is not None and history.stable_class == DYNAMIC:
                continue
            ds = circular_delta(
                float(obstacle.s_center), reference[0],
                self.classifier.track_length)
            dd = float(obstacle.d_center) - reference[1]
            if abs(dd) > self.dynamic_reid_max_lateral_m:
                continue
            candidates.append((math.hypot(ds, dd), obstacle_id))
        if not candidates:
            return None

        selected = unique_reidentification_candidate(
            candidates,
            self.dynamic_reid_max_distance_m,
            self.dynamic_reid_ambiguity_margin_m,
        )
        if selected is None:
            return None
        distance, new_id = selected
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

    def _active_expired(self, now):
        if self.active_dynamic_id is None:
            return False
        return (
            self.last_dynamic_position_at is None
            or now - self.last_dynamic_position_at > self.dynamic_reid_timeout_sec
        )

    def _select_dynamic(self, rows, now):
        """Return one locked physical tracker target, or ``None``.

        A retained active track wins even when a person is momentarily closer.
        Re-identification is handled before classification.  If the previous
        target is in its re-identification grace period, publish no new target
        instead of silently switching the GP to a different moving object.
        """
        candidates = [
            row for row in rows
            if row[1].stable_class == DYNAMIC
            and row[3]
            and row[4]
        ]
        by_id = {int(row[0].id): row for row in candidates}
        selected = by_id.get(self.active_dynamic_id)
        if selected is not None:
            return selected

        if self._active_expired(now):
            old_id = self.active_dynamic_id
            self.active_dynamic_id = None
            self.get_logger().warn(
                f'dynamic opponent tracker {old_id} expired; '
                'waiting to acquire one replacement target')

        if self.active_dynamic_id is not None:
            return None

        visible = [row for row in candidates if row[0].is_visible]
        if not visible:
            return None
        selected = min(
            visible,
            key=lambda row: initial_candidate_key(
                row[0], self.ego_s, self.classifier.track_length),
        )
        self.active_dynamic_id = int(selected[0].id)
        self.get_logger().warn(
            f'acquired dynamic opponent tracker {self.active_dynamic_id} as '
            f'logical opponent {self.logical_opponent_id}')
        return selected

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
        rows = []

        for obstacle in msg.obstacles:
            obstacle_id = int(obstacle.id)
            if obstacle_id in self.retired_dynamic_ids:
                continue
            raw_is_static = bool(obstacle.is_static)
            history = self.classifier.update(
                obstacle_id=obstacle.id,
                s=obstacle.s_center,
                d=obstacle.d_center,
                is_visible=obstacle.is_visible,
                now=now,
            )
            stable_obstacle = deepcopy(obstacle)
            stable_obstacle.is_static = history.stable_class == "STATIC"
            corridor_ok = self._inside_corridor(stable_obstacle)
            forward_ok = self._inside_forward_window(stable_obstacle)
            rows.append((
                stable_obstacle, history, obstacle_id == handed_off_id,
                corridor_ok, forward_ok, raw_is_static,
            ))

        self.classifier.remove_stale(now)
        selected = self._select_dynamic(rows, now)
        selected_tracker_id = int(selected[0].id) if selected is not None else None

        for (
            stable_obstacle, history, reidentified, corridor_ok, forward_ok,
            raw_is_static,
        ) in rows:
            obstacle_id = int(stable_obstacle.id)
            in_static, in_dynamic = output_membership(history.stable_class)
            is_selected = in_dynamic and obstacle_id == selected_tracker_id
            selected_output = None
            if is_selected:
                selected_output = deepcopy(stable_obstacle)
                self._stabilize_dynamic_speed(selected_output, now)
                selected_output.id = self.logical_opponent_id

            # Reject an object whose expected opponent footprint does not fit in
            # the track. This router only runs in head-to-head, so time-trials'
            # detector/tracker/static spline path is unchanged.
            #
            # UNKNOWN also has to have been SEEN. /tracking/stable_obstacles is
            # what the state machine trails on, and it picks the nearest thing
            # ahead regardless of class - so a track that has existed for one
            # frame, with a velocity computed from a single difference, became
            # the trailing target. The controller logged 32 opponent changes in
            # 75 s across 17 raw IDs, reporting speeds of 3.05, 0.00 and
            # -0.08 m/s for what was supposed to be one car.
            #
            # A track needs a few frames before anything about it is
            # trustworthy. This is deliberately well below the classifier's own
            # min_nb_meas (8): the point is to drop one-frame noise, not to
            # hide a real object from the trailing logic while it is being
            # classified. Confirmed statics and the selected opponent are
            # unaffected.
            unknown_is_settled = (
                history.stable_class == "UNKNOWN"
                and corridor_ok
                and history.sample_count >= self.min_unknown_samples
            )
            include_stable = in_static or unknown_is_settled or is_selected
            if include_stable:
                output_obstacle = deepcopy(
                    selected_output if is_selected else stable_obstacle)
                stable_msg.obstacles.append(output_obstacle)

            # Not yet classified goes out as static.
            #
            # UNKNOWN used to reach neither output. It reached stable_msg, so
            # the state machine saw the obstacle and trailed on it, while
            # h2h_spline_node - which reads static_obstacles and is the only
            # thing that draws an avoidance path - had an empty list. The car
            # knew something was in the way, slowed for it, and nobody planned
            # a way round: "static_avoidance_planner: published an empty path"
            # with an obstacle plainly there.
            #
            # Static is the safe default for "do not know yet". Being wrong
            # that way costs an avoidance that was not needed. Being wrong the
            # old way costs the path entirely.
            #
            # This is not rare. The classifier is asymmetric on purpose -
            # STATIC needs std_s AND std_d under min_std, DYNAMIC needs either
            # over max_std - so noise falls towards DYNAMIC and everything
            # between the two thresholds sits in UNKNOWN. A cone measured from
            # a moving car wanders a few centimetres over the 20-sample window
            # simply because the visible face of it changes, which lands it
            # right in that band.
            #
            # The selected opponent still wins: is_selected is only ever true
            # for a confirmed DYNAMIC track, so a real opponent cannot be
            # routed here by mistake.
            if in_static or unknown_is_settled:
                static_msg.obstacles.append(deepcopy(stable_obstacle))
            elif is_selected:
                dynamic_msg.obstacles.append(deepcopy(selected_output))

            debug_records.append({
                "id": self.logical_opponent_id if is_selected else obstacle_id,
                "tracker_id": obstacle_id,
                "raw_is_static": raw_is_static,
                "stable_class": history.stable_class,
                "std_s": round(float(history.std_s), 6),
                "std_d": round(float(history.std_d), 6),
                "sample_count": int(history.sample_count),
                "static_evidence": int(history.static_evidence),
                "dynamic_evidence": int(history.dynamic_evidence),
                "is_visible": bool(stable_obstacle.is_visible),
                "routed_vs": round(float(stable_obstacle.vs), 4),
                "reidentified": reidentified,
                "inside_opponent_corridor": corridor_ok,
                "inside_forward_window": forward_ok,
                "selected_opponent": is_selected,
            })

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
