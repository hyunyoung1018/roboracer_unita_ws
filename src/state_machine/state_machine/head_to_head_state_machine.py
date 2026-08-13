#!/usr/bin/env python3
"""Head-to-head wrapper for the legacy UNITA state machine.

Everything head-to-head needs that the shared state machine gets wrong is
corrected here, by overriding, rather than in state_machine_node.py. That file
is what time_trials.launch.xml runs for static obstacle avoidance and it stays
byte-for-byte as it was.
"""

from copy import deepcopy

import numpy as np
import rclpy

from .state_machine_node import StateMachine


def nearest_ahead(obstacles, current_s, track_length, horizon):
    """Return the closest obstacle ahead for the trailing-target fallback."""
    if not obstacles or track_length is None or track_length <= 0.0:
        return None, None
    candidates = []
    for obstacle in obstacles:
        gap = (
            float(obstacle.s_center) - float(current_s)
        ) % float(track_length)
        if gap <= float(horizon):
            candidates.append((gap, obstacle))
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: item[0])


class HeadToHeadStateMachine(StateMachine):
    """Head-to-head-only safety fixes around the preserved shared state machine.

    The shared state machine negates this parameter during construction, while
    its runtime callback assigns it directly. Keeping the correction here
    preserves time-trials behavior and scopes it to head-to-head launches.

    The wrapper also uses the measured 0.28 m vehicle width for clearance tests
    (the shared 0.40 m value is a raceline-state tolerance, not the body width)
    and holds a blocked-raceline decision briefly.  That prevents centimetre
    scale obstacle-edge noise from alternating RACELINE/TRAILING every frame.
    """

    def __init__(self):
        super().__init__()
        self.use_force_trailing = bool(self.params.use_force_trailing)
        self.clearance_vehicle_width_m = float(
            self._get_or_declare("clearance_vehicle_width_m", 0.28))
        self.trailing_block_hold_sec = float(
            self._get_or_declare("trailing_block_hold_sec", 0.4))
        self.trailing_target_hold_sec = float(
            self._get_or_declare("trailing_target_hold_sec", 0.5))
        # [m] Floor on the "am I back on the raceline" lateral test. The shared
        # transitions all call _check_close_to_raceline(0.05), which paired with
        # a heading test that never ran (see the override below) made the whole
        # condition "within 5 cm of the line". A car that has just completed an
        # evasion is not within 5 cm of anything, so it stayed in
        # RECOVERY/TRAILING long after it had returned. 0.15 is inside the
        # 0.40 m gb_ego_width_m that the no-argument form uses and outside the
        # noise on cur_d.
        self.raceline_return_tolerance_m = float(
            self._get_or_declare("raceline_return_tolerance_m", 0.15))
        self._last_gb_blocked_at = None
        self._last_trailing_target = None
        self._last_trailing_target_at = None

    def _get_or_declare(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    # ------------------------------------------------------------------ #
    # CONDITION FIXES                                                     #
    # ------------------------------------------------------------------ #
    def _check_close_to_raceline(self, threshold_m=None) -> bool:
        """Raise the floor on the lateral tolerance, never lower it."""
        if threshold_m is not None:
            threshold_m = max(
                float(threshold_m), self.raceline_return_tolerance_m)
        return super()._check_close_to_raceline(threshold_m)

    def _check_close_to_raceline_heading(self, threshold_deg=None) -> bool:
        """Actually compare headings.

        The shared version compares ``abs(cur_d)`` - a lateral distance in
        metres - against ``deg2rad(threshold_deg)`` whenever a threshold is
        passed, and every call site passes 20. So the heading was never looked
        at: the test silently degraded to ``abs(cur_d) < 0.349``, duplicating
        the lateral check next to it. The no-argument branch does compare
        headings but subtracts two angles without wrapping, so it reads a car
        sitting at the +/-pi seam as 360 degrees off.
        """
        index = int(self.cur_s / self.waypoints_dist) % self.num_glb_wpnts
        reference_psi = float(self.cur_gb_wpnts.list[index].psi_rad)
        limit = np.deg2rad(20.0 if threshold_deg is None else float(threshold_deg))
        error = float(self.current_position[2]) - reference_psi
        error = (error + np.pi) % (2.0 * np.pi) - np.pi
        return bool(abs(error) < limit)

    def _check_getting_closer(self, threshold_m=3.0) -> bool:
        """Compare against the nearest obstacle ahead, and honour the horizon.

        The shared version reads ``obstacles_in_interest[0]``, which is not the
        nearest obstacle - the list is built by walking the ObstacleArray in
        publication order, so index 0 is whichever track the tracker created
        first. With a static box on the track that is a ``vs`` of zero, the test
        passes unconditionally and the gate does nothing; with the opponent
        first it behaves as intended. Which one it is depends on the order two
        objects happened to be detected in.

        ``threshold_m`` was also accepted and then ignored, so the 10 m the
        dynamic branch asks for and the 7 m the static branch asks for were the
        same test.
        """
        _, target = nearest_ahead(
            self.cur_obstacles_in_interest,
            self.cur_s,
            self.track_length,
            threshold_m,
        )
        if target is None:
            return False
        return bool(self.cur_vs - float(target.vs) > -0.5)

    def _check_overtaking_mode(self) -> bool:
        """Let the predictor veto ENTERING an overtake.

        ``/opponent_prediction/force_trailing`` is subscribed by the shared
        state machine, stored, and never read. The head-to-head wrapper already
        repairs the inverted ``use_force_trailing`` flag that feeds it; this is
        the missing consumer. Without it the state machine can enter and hold
        OVERTAKE on a cached avoidance path while the predictor is saying the
        opponent's future is not known well enough to pass it.

        Entry only. Sustainability is deliberately left alone: force_trailing
        goes true exactly when the opponent leaves the predictor's forward
        window, which is when the car is alongside it - the worst possible
        moment to abandon a half-finished manoeuvre. Leaving OVERTAKE there is
        the job of the TTL and of the free checks, which still run.
        """
        allowed = super()._check_overtaking_mode()
        if allowed and self.force_trailing:
            self.get_logger().info(
                "overtake entry vetoed: predictor asked for force_trailing",
                throttle_duration_sec=2.0)
            return False
        return allowed

    def _check_static_overtaking_mode(self) -> bool:
        """Clear the static/dynamic latch when neither branch arms.

        ``static_overtaking_mode`` decides which planner's cache OVERTAKE
        slices, and the shared code only ever clears it on a successful
        *dynamic* check. Once static avoidance has armed once, a later loop
        where both checks fail leaves it true, and the next OVERTAKE reads the
        static cache for what may be a dynamic opponent.
        """
        allowed = super()._check_static_overtaking_mode()
        if not allowed:
            self.static_overtaking_mode = False
        return allowed

    def _remember_trailing_target(self, target):
        if target is None or not bool(target.is_visible):
            return
        self._last_trailing_target = deepcopy(target)
        self._last_trailing_target_at = self.now_sec()

    def _held_trailing_target(self):
        """Propagate the one opponent briefly through a detector dropout."""
        if (
            self._last_trailing_target is None
            or self._last_trailing_target_at is None
        ):
            return None
        age = self.now_sec() - self._last_trailing_target_at
        if age > self.trailing_target_hold_sec:
            return None
        target = deepcopy(self._last_trailing_target)
        if self.track_length and self.track_length > 0.0:
            ds = max(0.0, float(target.vs)) * max(0.0, age)
            target.s_start = (float(target.s_start) + ds) % self.track_length
            target.s_end = (float(target.s_end) + ds) % self.track_length
            target.s_center = (float(target.s_center) + ds) % self.track_length
        target.is_visible = False
        return target

    def _nearest_interest_target(self):
        gap, target = nearest_ahead(
            self.cur_obstacles_in_interest,
            self.cur_s,
            self.track_length,
            self.interest_horizon_m,
        )
        return gap, target

    def _ot_path_clears_prediction(self, wpnts_data) -> bool:
        """Check an overtaking path against EVERY predicted opponent pose.

        The shared check only looks at the slice of the prediction the car is
        expected to arrive at:

            clip_vs   = max(relative_vs, overtake_min_closing_mps)   # 2.5
            ttc       = (gap - length) / clip_vs
            start_idx = int(ttc / prediction_dt)

        Two things go wrong with that. The closing speed is floored at 2.5 m/s
        whether or not the car is closing that fast, so a car creeping up at
        0.5 m/s is checked against where the opponent will be five times too
        early. And when ``ttc`` exceeds the predictor's two-second horizon,
        ``start_idx`` clamps to the end of the array, the slice is empty, no
        pose is examined at all and the path is reported free by default. From
        a distance the path is therefore always free, OVERTAKE arms, and the
        real check only bites once the car is close - which reads from outside
        as the car pulling out and then aborting.

        Checking every pose is both simpler and correct here, because it is the
        same guarantee the lane-change planner makes: its lateral profile holds
        the evasion offset across the opponent's whole predicted span, so a path
        that is genuinely safe passes this, and one that was only ever safe at
        an optimistic arrival time does not.
        """
        if not wpnts_data.is_init or not len(self.obstacles_prediction):
            return True
        if self.obstacles_prediction_id is None:
            return True
        obstacle = next(
            (obs for obs in self.cur_obstacles_in_interest
             if int(obs.id) == int(self.obstacles_prediction_id)),
            None,
        )
        if obstacle is None or not self.max_s:
            return True

        gap = (float(obstacle.s_center) - self.cur_s) % self.max_s
        reference = float(wpnts_data.free_scaling_reference_distance_m)
        scaling = 1.0 if reference <= 0.0 else float(
            np.clip(gap / reference, 0.0, 1.0))
        required = float(wpnts_data.lateral_width_m) * scaling
        occupied = (
            self._lateral_half_width(obstacle)
            + getattr(self, "clearance_vehicle_width_m", self.gb_ego_width_m) / 2.0
        )

        path_s = wpnts_data.array[:, 2]
        path_d = wpnts_data.array[:, 3]
        pred_s = np.asarray(
            [p.pred_s for p in self.obstacles_prediction], dtype=float)
        pred_d = np.asarray(
            [p.pred_d for p in self.obstacles_prediction], dtype=float)
        nearest = np.argmin(np.abs(path_s[None, :] - pred_s[:, None]), axis=1)
        worst = float(np.min(np.abs(path_d[nearest] - pred_d) - occupied))
        if worst >= required:
            return True

        self.get_logger().info(
            f"{wpnts_data.name}: blocked over the full prediction - worst "
            f"clearance {worst:.3f} m, need {required:.3f} m "
            f"({len(pred_s)} poses checked)",
            throttle_duration_sec=1.0)
        wpnts_data.closest_target = obstacle
        wpnts_data.closest_gap = gap
        return False

    def _check_free_frenet(self, wpnts_data):
        """Use physical body width and debounce only the global-path result."""
        configured_width = self.gb_ego_width_m
        clearance_width = getattr(
            self, "clearance_vehicle_width_m", configured_width)
        self.gb_ego_width_m = clearance_width
        try:
            is_free = super()._check_free_frenet(wpnts_data)
        finally:
            # Keep the legacy value for close-to-raceline state semantics.
            self.gb_ego_width_m = configured_width

        # An overtaking path the shared check called free may only have been
        # free because the arrival-time window it examined was empty.
        if is_free and wpnts_data.is_ot_wpnts:
            is_free = self._ot_path_clears_prediction(wpnts_data)

        if not wpnts_data.is_gb_track_wpnts:
            return is_free

        now = self.now_sec()
        gap, fallback_target = self._nearest_interest_target()
        if not is_free:
            self._last_gb_blocked_at = now
            if (
                wpnts_data.closest_target is None
                and fallback_target is not None
            ):
                wpnts_data.closest_target = fallback_target
                wpnts_data.closest_gap = gap
            self._remember_trailing_target(wpnts_data.closest_target)
            return False

        # Hold the previous blocked decision through one short detector/ID
        # dropout.  The held target is propagated at its last credible speed
        # and its timestamp is never refreshed by another invisible frame, so
        # this cannot turn into a permanent ghost opponent.
        held_target = self._held_trailing_target()
        target = fallback_target or held_target
        if (
            target is not None
            and getattr(self, "_last_gb_blocked_at", None) is not None
            and now - self._last_gb_blocked_at <= self.trailing_block_hold_sec
        ):
            wpnts_data.closest_target = target
            wpnts_data.closest_gap = (
                float(target.s_start) - self.cur_s
            ) % self.track_length
            if fallback_target is not None:
                self._remember_trailing_target(fallback_target)
            return False
        return True

    def get_farthest_target(self, local_wpnts_src):
        """Never publish TRAILING without a usable opponent target."""
        targets, selected_src = super().get_farthest_target(local_wpnts_src)
        if targets:
            self._remember_trailing_target(targets[0])
            return targets, selected_src

        _, target = self._nearest_interest_target()
        if target is not None:
            self._remember_trailing_target(target)
            return [target], selected_src

        held_target = self._held_trailing_target()
        if held_target is not None:
            return [held_target], selected_src
        return [], selected_src


def main(args=None):
    rclpy.init(args=args)
    state_machine = HeadToHeadStateMachine()
    try:
        rclpy.spin(state_machine)
    except KeyboardInterrupt:
        pass
    finally:
        state_machine.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
