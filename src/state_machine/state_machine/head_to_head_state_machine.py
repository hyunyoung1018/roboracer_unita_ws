#!/usr/bin/env python3
"""Head-to-head wrapper for the legacy UNITA state machine."""

from copy import deepcopy

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
        self._last_gb_blocked_at = None
        self._last_trailing_target = None
        self._last_trailing_target_at = None

    def _get_or_declare(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

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
