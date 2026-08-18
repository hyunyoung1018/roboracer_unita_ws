#!/usr/bin/env python3
"""Head-to-head spline planner: the shared one, told where the opponent is.

This exists as a subclass rather than as edits to :mod:`spline.spline_node`
because time_trials.launch.xml runs that node for static obstacle avoidance and
must keep behaving exactly as it does today. Nothing here has any effect unless
/tracking/dynamic_obstacles is publishing, and nothing publishes it in time
trials - h2h_tracking_node is what produces it, by splitting its own output on
the tracker's speed-based is_static.

ONE THING CHANGES: which side of a static obstacle the path goes.

The shared planner picks the side with more room to the TRACK BOUNDARY. Its
only input is /tracking/static_obstacles, so it has never heard of the
opponent, and the two of them tend to want the same side - the opponent
overtakes into the wider gap for the same reason the planner does. When they
coincide the path is drawn through the opponent, the state machine measures it
against every obstacle it knows about, and refuses. From outside the car that
is "sometimes it goes around the box and sometimes it stops behind it",
decided by which way the opponent happened to go.

The seam is _tightest_bounds, which answers "how much lateral room is there
over this stretch". A car sitting in the gap is a straight answer to that
question, so the opponent is folded in as a wall and everything downstream -
left_room/right_room, the boundary_margin test, the side comparison and its
hysteresis - keeps working unchanged on top of it.

WHAT IT DELIBERATELY DOES NOT DO: make this a dynamic-obstacle planner. The
opponent never becomes something to plan a path around, never joins a corridor
and never moves the apex. It only takes room away, over the stretch the
corridor holds its offset. An opponent the car is trailing, several metres
before the obstacle, is outside that stretch and changes nothing - which is
what leaves the state machine's own distant-opponent handling
(_blocked_only_by_distant_dynamics) free to do its job.
"""

import rclpy
from f110_msgs.msg import ObstacleArray
from rclpy.executors import ExternalShutdownException

from spline.spline_node import SplineNode


class HeadToHeadSplineNode(SplineNode):
    """Choose the side of a static obstacle that the opponent is not on."""

    PARAM_DEFAULTS = {
        # [m] How far either side of the corridor's hold region an opponent
        # still counts. The path ramps off the raceline before the obstacle and
        # back on after it, so a car level with the obstacle is not the only one
        # in the way. Below about a metre an opponent drawing alongside stops
        # being seen; well above it, an opponent the car is merely trailing
        # starts taking a side away several metres early.
        'opponent_bound_span_m': 1.5,
        # [m] Extra clearance to the opponent's near edge, on top of the
        # boundary_margin the path already keeps off a wall. A wall does not
        # move and this does, and its measured edges come from a 12 cm
        # detection pillar believed to be a car, so the edge itself is an
        # estimate.
        'opponent_extra_clearance_m': 0.10,
        # [s] Ignore the opponent list once it is older than this. Without it a
        # detector dropout, or simply no moving track to report, would leave the
        # last known opponent taking a side away for the rest of the run.
        'opponent_timeout_sec': 0.5,
        # Set false to get the shared planner's side choice back without
        # relaunching, which is the A/B for this whole file.
        'opponent_side_avoidance': True,
    }

    def __init__(self):
        super().__init__()
        for name, default in self.PARAM_DEFAULTS.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, default)

        self.opponents = ObstacleArray()
        self.opponents_stamp = None
        # Absolute topic name, not the remapped /tracking/obstacles the base
        # reads: head_to_head.launch.xml points that at h2h_tracking_node's
        # static-only stream, and this is the other half of the same split.
        self.create_subscription(
            ObstacleArray, '/tracking/dynamic_obstacles', self._opponent_cb, 10)
        self.get_logger().info(
            'head-to-head spline: side choice avoids the opponent '
            f'({self._param("opponent_bound_span_m"):.2f} m either side of the '
            'corridor)')

    def _param(self, name):
        return self.get_parameter(name).value

    def _opponent_cb(self, msg):
        self.opponents = msg
        self.opponents_stamp = self.now_sec()

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _live_opponents(self):
        """The opponents worth believing, or an empty list."""
        if not self._param('opponent_side_avoidance'):
            return []
        if self.opponents_stamp is None or not self.opponents.obstacles:
            return []
        if self.now_sec() - self.opponents_stamp > float(
                self._param('opponent_timeout_sec')):
            return []
        return list(self.opponents.obstacles)

    def _tightest_bounds(self, s_values, reference, start_s, end_s, max_s):
        """The shared bounds, with the opponent counted as a wall.

        Both returned values are POSITIVE half-widths measured out from the
        raceline - that is the convention the caller uses, in
        ``bound_left - left_apex`` and ``bound_right + right_apex``. An
        obstacle's own d_left/d_right are SIGNED offsets, so an opponent on the
        left has a positive d_right (its near edge) and one on the right has a
        negative d_left. An opponent straddling the raceline takes room from
        both sides and drives them to zero, which is the honest answer: there
        is no way past it here, and the shared planner then bails with its own
        "no room either side" rather than publishing a path into it.
        """
        bound_left, bound_right = super()._tightest_bounds(
            s_values, reference, start_s, end_s, max_s)
        opponents = self._live_opponents()
        if not opponents:
            return bound_left, bound_right

        span = float(self._param('opponent_bound_span_m'))
        clearance = float(self._param('opponent_extra_clearance_m'))
        length = max(0.0, end_s - start_s)
        limited_left, limited_right = bound_left, bound_right

        for opponent in opponents:
            # start_s is unwrapped and may run past max_s, so compare on a
            # signed delta rather than a plain modulo - an opponent a little
            # BEFORE the hold region is one the path ramps past, and the
            # unsigned form would put it most of a lap away.
            delta = (float(opponent.s_center) - start_s) % max_s
            if delta > 0.5 * max_s:
                delta -= max_s
            if delta < -span or delta > length + span:
                continue
            if float(opponent.d_left) > 0.0:
                limited_left = min(
                    limited_left, float(opponent.d_right) - clearance)
            if float(opponent.d_right) < 0.0:
                limited_right = min(
                    limited_right, -float(opponent.d_left) - clearance)

        limited_left = max(0.0, limited_left)
        limited_right = max(0.0, limited_right)
        if limited_left < bound_left or limited_right < bound_right:
            self.get_logger().info(
                'opponent narrows the corridor at '
                f's={start_s % max_s:.2f}..{end_s % max_s:.2f}: '
                f'left {bound_left:.2f} -> {limited_left:.2f} m, '
                f'right {bound_right:.2f} -> {limited_right:.2f} m',
                throttle_duration_sec=2.0)
        return limited_left, limited_right


def main(args=None):
    rclpy.init(args=args)
    node = HeadToHeadSplineNode()
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
