#!/usr/bin/env python3
"""Lightweight Frenet spline obstacle avoidance."""

import time

import numpy as np
import rclpy
from f110_msgs.msg import ObstacleArray, OTWpntArray, WpntArray, Wpnt
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from scipy.interpolate import CubicSpline
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

from frenet_conversion.frenet_converter import FrenetConverter


def _path_heading_and_curvature(x, y):
    """Return heading and signed curvature for a sampled Cartesian path."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return np.zeros_like(x), np.zeros_like(x)

    edge_order = 2 if x.size >= 3 else 1
    dx = np.gradient(x, edge_order=edge_order)
    dy = np.gradient(y, edge_order=edge_order)
    heading = np.arctan2(dy, dx)
    if x.size < 3:
        return heading, np.zeros_like(x)

    ddx = np.gradient(dx, edge_order=edge_order)
    ddy = np.gradient(dy, edge_order=edge_order)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    curvature = np.divide(
        dx * ddy - dy * ddx,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-9,
    )
    return heading, curvature


class SplineNode(Node):
    """Generate a local avoidance line around every obstacle in the way.

    The path is a single corridor: it leaves the raceline before the first
    obstacle it has to pass, holds one lateral offset that clears all of them,
    and returns after the last. Earlier it apexed on the NEAREST obstacle alone
    and was back on the raceline 1.2 * scale metres later, so a second obstacle
    anywhere in the return leg sat on the published path - which the state
    machine then measured and refused, correctly, leaving the car trailing two
    obstacles it had the room to drive around.
    """

    def __init__(self):
        super().__init__('spline_node')
        self.obstacles = ObstacleArray()
        self.odom = None
        self.global_msg = None
        self.scaled_msg = None
        self.converter = None

        self.last_path = None
        self.last_side = None

        defaults = {
            'lookahead': 3.0,
            'evasion_distance': 0.3,
            'trajectory_threshold': 0.6,
            'boundary_margin': 0.20,
            'spline_resolution': 0.10,
            'rate_hz': 10.0,
            'path_hold_s': 0.3,
            'side_hysteresis_m': 0.10,
            'raceline_clearance_m': 0.50,
            # [m] How far past the first obstacle the corridor may reach to
            # pick up further ones. Beyond this the path is planned for what it
            # can hold and the next replan takes over - a corridor longer than
            # this holds an offset across more track than the boundary check
            # can be expected to admit on a 1.0 to 1.9 m wide map.
            # Corridors at all. False plans one obstacle at a time.
            #
            # The only thing kept from the run of corridor patches that
            # followed b1db364: they fixed each other in circles and the car
            # ended up worse than it started, so the file is back to the last
            # state that was confirmed working on the car. This switch stays
            # because both launches pass it, and because being able to turn the
            # feature off without a rebuild is what that week was missing.
            'corridor_enabled': True,
            'corridor_max_len_m': 8.0,
            # [m] How far past the PREVIOUS member the next obstacle may sit
            # and still join the corridor, scaled by speed the same way the
            # return leg is. 3.0 is what the code did implicitly before it was
            # a parameter - it used retreat[-1], the end of the return leg -
            # so the default changes nothing and raising it is the knob.
            #
            # It is a reachability limit before it is a tuning one. Shifting
            # 0.5 m sideways at 3 m/s needs roughly 6*dd*v^2/L^2 of lateral
            # grip, which is 6.8 m/s^2 over 2 m and 1.7 over 4 - so obstacles
            # closer together than about 3 m cannot be passed at different
            # offsets anyway, whatever this says.
            'corridor_link_gap_m': 3.0,
            # [m] How far the next obstacle's d may differ FROM THE LEADER'S
            # and still join. Signed, so opposite sides fall out on their own:
            # -0.10 and -0.15 differ by 0.05 and belong together, -0.10 and
            # +0.15 differ by 0.25 and do not.
            #
            # This is the legible knob on top of the geometric test below,
            # which asks the exact question - does one offset still fit - and
            # stays as the backstop.
            'corridor_link_d_tol_m': 0.20,
            # [m] Spacing of the control points that hold the offset between
            # the first and last obstacle. Without them a cubic through two
            # equal knots bulges between them; the samples are clipped to the
            # planned offset anyway, but flat control points keep the curvature
            # - and therefore the speed the controller picks - honest.
            'corridor_point_spacing_m': 1.0,
            'retreat_clearance_m': 0.5,
            'retreat_floor_m': 1.0,
            'min_path_clearance_m': 0.20,
            'measure': False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self._load_parameters()
        self.add_on_set_parameters_callback(self._parameter_cb)

        self.create_subscription(ObstacleArray, '/tracking/obstacles', self._obstacle_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom_frenet', self._odom_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self._global_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints_scaled', self._scaled_cb, 10)
        self.path_pub = self.create_publisher(OTWpntArray, '/planner/avoidance/otwpnts', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/planner/avoidance/markers', 10)
        self.latency_pub = self.create_publisher(Float32, '/planner/avoidance/latency', 10)
        # Rate is fixed at construction: a timer period cannot be changed at
        # runtime, and this one competes for the jetson with the detector and
        # the particle filter. 20 Hz replanned a static obstacle every 5 cm of
        # travel, for nothing.
        self.create_timer(1.0 / max(1.0, float(self.get_parameter('rate_hz').value)), self._loop)

    def _load_parameters(self):
        self.lookahead = float(self.get_parameter('lookahead').value)
        self.evasion_distance = float(self.get_parameter('evasion_distance').value)
        self.trajectory_threshold = float(self.get_parameter('trajectory_threshold').value)
        self.boundary_margin = float(self.get_parameter('boundary_margin').value)
        self.resolution = float(self.get_parameter('spline_resolution').value)
        self.path_hold_s = float(self.get_parameter('path_hold_s').value)
        self.side_hysteresis_m = float(self.get_parameter('side_hysteresis_m').value)
        self.raceline_clearance_m = float(
            self.get_parameter('raceline_clearance_m').value)
        self.corridor_enabled = bool(
            self.get_parameter('corridor_enabled').value)
        self.corridor_max_len_m = float(
            self.get_parameter('corridor_max_len_m').value)
        self.corridor_link_gap_m = float(
            self.get_parameter('corridor_link_gap_m').value)
        self.corridor_link_d_tol_m = float(
            self.get_parameter('corridor_link_d_tol_m').value)
        self.retreat_clearance_m = float(
            self.get_parameter('retreat_clearance_m').value)
        self.retreat_floor_m = float(
            self.get_parameter('retreat_floor_m').value)
        self.min_path_clearance_m = float(
            self.get_parameter('min_path_clearance_m').value)
        self.corridor_point_spacing_m = max(
            0.2, float(self.get_parameter('corridor_point_spacing_m').value))
        self.measure = bool(self.get_parameter('measure').value)

    def _parameter_cb(self, params):
        attributes = {
            'lookahead': 'lookahead',
            'evasion_distance': 'evasion_distance',
            'trajectory_threshold': 'trajectory_threshold',
            'boundary_margin': 'boundary_margin',
            'spline_resolution': 'resolution',
            'path_hold_s': 'path_hold_s',
            'side_hysteresis_m': 'side_hysteresis_m',
            'raceline_clearance_m': 'raceline_clearance_m',
            'corridor_enabled': 'corridor_enabled',
            'corridor_max_len_m': 'corridor_max_len_m',
            'corridor_link_gap_m': 'corridor_link_gap_m',
            'corridor_link_d_tol_m': 'corridor_link_d_tol_m',
            'corridor_point_spacing_m': 'corridor_point_spacing_m',
            'retreat_clearance_m': 'retreat_clearance_m',
            'retreat_floor_m': 'retreat_floor_m',
            'min_path_clearance_m': 'min_path_clearance_m',
            'measure': 'measure',
        }
        for parameter in params:
            if parameter.name in attributes:
                setattr(self, attributes[parameter.name], parameter.value)
        # _load_parameters floors this at 0.2 and a live set went around that.
        # The control-point loop advances by exactly this much, so 0 hangs the
        # node in a while loop that never terminates.
        self.corridor_point_spacing_m = max(0.2, float(self.corridor_point_spacing_m))
        return SetParametersResult(successful=True)

    def _obstacle_cb(self, msg):
        self.obstacles = msg

    def _odom_cb(self, msg):
        self.odom = msg

    def _global_cb(self, msg):
        if not msg.wpnts:
            return
        self.global_msg = msg
        if self.converter is None:
            x = np.asarray([w.x_m for w in msg.wpnts])
            y = np.asarray([w.y_m for w in msg.wpnts])
            psi = np.asarray([w.psi_rad for w in msg.wpnts])
            self.converter = FrenetConverter(x, y, psi)

    def _scaled_cb(self, msg):
        if msg.wpnts:
            self.scaled_msg = msg

    def _loop(self):
        if self.odom is None or self.global_msg is None or self.scaled_msg is None or self.converter is None:
            return
        started = time.perf_counter()
        result = self._plan()

        # Hold the last good path across a dropped frame.
        #
        # Every bail publishes an empty path, and the state machine caches
        # whatever arrived last - so one bad frame in two erased the good one
        # before it could be acted on. Measured on test_213: the planner
        # alternated between a valid path and "no room" on the SAME stationary
        # obstacle, several times a second, and static avoidance never once
        # latched because have_path was false at almost every check.
        #
        # The held path keeps its ORIGINAL stamp. That is the point: it is not
        # claimed to be fresh, and the state machine's own latest_threshold
        # (0.25 s) still decides whether it is too old to use. This only stops
        # a gap from being filled with a positive assertion that there is
        # nothing there.
        if result.wpnts:
            self.last_path = result
        elif self.last_path is not None:
            age = self._age(self.last_path)
            if age <= self.path_hold_s:
                result = self.last_path
            else:
                self.last_path = None

        self.path_pub.publish(result)
        self.marker_pub.publish(self._markers(result))
        if self.measure:
            self.latency_pub.publish(Float32(data=float(time.perf_counter() - started)))

    def _age(self, msg):
        stamp = msg.header.stamp
        return (self.get_clock().now().nanoseconds * 1e-9
                - (stamp.sec + stamp.nanosec * 1e-9))

    def _bail(self, out, reason):
        """Return an empty path, but say why.

        Every exit from _plan below publishes an empty OTWpntArray, which the
        state machine reads as "no avoidance available" and silently keeps
        trailing. Without a reason on the way out, a planner that never manages
        to route around anything looks identical to one that is not running.
        """
        out.wpnts.clear()
        self.get_logger().info(f"no avoidance path: {reason}", throttle_duration_sec=2.0)
        return out

    def _raceline_clears(self, obstacle, cur_d=0.0):
        """Will the car get past this obstacle without a plan for it?

        The near edge has to be raceline_clearance_m clear of everywhere the
        car is about to be. That number is tuned to equal what the state
        machine calls a blocked raceline (gb_ego_width_m/2 + global_tracking's
        lateral_width_m), so a False here is the same judgement the state
        machine makes when it starts looking for a way around.

        cur_d is why this takes an argument. The question used to be asked at
        d=0 alone, and the car is not at d=0 when it matters: it is on the
        tail of the last avoidance, coming back to the line. Measured, the
        third obstacle of a row: edges +0.32..+0.73, so d_right cleared the
        0.30 threshold by two centimetres and the planner answered "nothing to
        avoid" and published nothing. The car was at d=+0.41 on the return leg
        from the obstacle before it, which leaves the body 23 cm INSIDE that
        obstacle, and it drove into it every lap while a path that looked
        perfectly good in RViz - the previous obstacle's, whose tail happens
        to run over the top of this one - was on screen.

        So the strip that has to be clear runs from where the car is to the
        raceline it is heading back to, widened by the clearance at both ends.
        With cur_d = 0 this is exactly the old test.
        """
        lo = min(0.0, cur_d) - self.raceline_clearance_m
        hi = max(0.0, cur_d) + self.raceline_clearance_m
        return obstacle.d_right >= hi or obstacle.d_left <= lo

    def _tightest_bounds(self, s_values, reference, start_s, end_s, max_s):
        """Narrowest track bounds over an s range, sampled at 0.25 m."""
        length = max(0.0, end_s - start_s)
        count = int(np.ceil(length / 0.25)) + 1
        samples = (np.linspace(start_s, end_s, count) if count > 1
                   else np.asarray([start_s])) % max_s
        indices = np.argmin(
            np.abs(s_values[None, :] - samples[:, None]), axis=1)
        return (min(reference[int(i)].d_left for i in indices),
                min(reference[int(i)].d_right for i in indices))

    def _plan(self):
        out = OTWpntArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        cur_s = self.odom.pose.pose.position.x
        # The car's CURRENT lateral offset, and the reason it is here at all.
        #
        # /car_state/odom_frenet carries s in position.x and d in position.y.
        # The planner read only s, so every path it built started on the
        # raceline no matter where the car actually was - and while a corridor
        # is being driven the car is a long way off it. Replanning at 10 Hz
        # then handed a displaced car a path that says "be at d=0 here", it
        # steered back towards the line, and the obstacle it had just passed
        # with the front wheels was still beside the rear ones.
        cur_d = float(self.odom.pose.pose.position.y)
        max_s = self.global_msg.wpnts[-1].s_m
        if max_s <= 0.0:
            return out

        # Two different questions, two different filters.
        #
        # `in_range` is everything ahead, whatever its d. `candidates` is the
        # subset near enough to the line to be worth planning around, and it
        # is what picks the leader.
        #
        # The corridor has to be drawn against `in_range`. It holds one offset
        # from the first obstacle to the last, so anything sitting in between
        # decides whether that is drivable - including something out at d=0.7
        # that this planner would never bother to avoid on its own. Growing
        # the corridor over `candidates` meant a zigzag's middle obstacle, off
        # at the far side near the 0.6 m threshold, was invisible: the first
        # and third linked up and the corridor closed over it. Worse, its
        # measured d wanders a few centimetres between frames, so it crossed
        # the threshold back and forth and the corridor formed and collapsed
        # several times a second - a different path each time, and the car
        # dithering while the two fought.
        in_range = [
            obstacle for obstacle in self.obstacles.obstacles
            if (obstacle.s_center - cur_s) % max_s < self.lookahead
        ]
        # Measured from the strip the car is on, not from the raceline.
        #
        # Same mistake as _raceline_clears had. Coming off an avoidance at
        # d=+0.4, an obstacle at d=+0.62 is 22 cm away sideways, and this
        # dropped it as "none in range, at d 0.62 (threshold 0.6)" - the
        # planner never even considered the thing the car was about to hit.
        # With cur_d = 0 this is exactly abs(d_center) < trajectory_threshold.
        strip_lo, strip_hi = min(0.0, cur_d), max(0.0, cur_d)

        def off_strip(d):
            if strip_lo <= d <= strip_hi:
                return 0.0
            return min(abs(d - strip_lo), abs(d - strip_hi))

        candidates = [
            obstacle for obstacle in in_range
            if off_strip(obstacle.d_center) < self.trajectory_threshold
        ]
        if not candidates:
            if self.obstacles.obstacles:
                nearest = min(self.obstacles.obstacles,
                              key=lambda o: (o.s_center - cur_s) % max_s)
                self.get_logger().info(
                    f"{len(self.obstacles.obstacles)} obstacle(s), none in range: "
                    f"nearest is {(nearest.s_center - cur_s) % max_s:.2f} m ahead "
                    f"(lookahead {self.lookahead}) at d {nearest.d_center:.2f}, "
                    f"{off_strip(nearest.d_center):.2f} m off the strip from the "
                    f"car at d={cur_d:+.2f} to the raceline "
                    f"(threshold {self.trajectory_threshold})",
                    throttle_duration_sec=2.0)
            self.last_side = None
            return out
        reference = self.scaled_msg.wpnts
        s_values = np.asarray([w.s_m for w in reference])

        def ahead(item):
            return (item.s_center - cur_s) % max_s

        candidates.sort(key=ahead)

        # Only obstacles the RACELINE does not already clear are worth a path,
        # and the FIRST such one leads the corridor.
        #
        # This used to be decided after the side was chosen and on the nearest
        # candidate alone: if that one happened to sit far enough off the line
        # the planner answered "nothing to avoid" and published nothing - even
        # with a second obstacle two metres later sitting on the line. The state
        # machine then asked for an avoidance that never came and the car
        # trailed forever. raceline_clearance_m is tuned to equal what the state
        # machine calls a blocked raceline, so the two agree about WHETHER an
        # obstacle is in the way; they also have to agree about WHICH one.
        blocking = [o for o in candidates if not self._raceline_clears(o, cur_d)]
        if not blocking:
            nearest = candidates[0]
            return self._bail(
                out,
                f"nothing to plan for: all {len(candidates)} obstacle(s) in "
                f"range clear the strip from the car at d={cur_d:+.2f} back to "
                f"the raceline. Nearest is at s={nearest.s_center:.2f} with "
                f"edges {nearest.d_right:+.2f}..{nearest.d_left:+.2f} m, and "
                f"this planner only moves for something inside "
                f"{self.raceline_clearance_m:.2f} m of it")

        # How far the path reaches, which the side choice below needs: an offset
        # is only usable if the rest of what this path drives through is clear.
        #
        # Control points, in metres of s relative to the first and last obstacle
        # the corridor covers. Only the hold region is offset laterally; the
        # approach and the return hold the path on the raceline.
        #
        # Was [-4.0, -3.0, -1.5, 0.0, 2.0, 3.0, 4.5]. The return leg is
        # shortened because it became the only remaining reason avoidance
        # failed on map_no - every refusal read
        #
        #   path leaves the track at s=3.755 (+1.30 m from the apex)
        #   path leaves the track at s=4.947 (+0.80 m from the apex)
        #   path leaves the track at s=4.367 (+0.50 m from the apex)
        #
        # i.e. the car cleared the obstacle and then ran out of track while
        # still displaced. With the first return point at +2.0 the apex is a
        # local maximum of a cubic that decays slowly: measured 0.393 m of
        # offset still remaining 1.30 m past an apex of 0.47 m, 84% of it.
        # Bringing that point to +1.2 is what actually shortens the swerve;
        # the outer two follow so the tail does not kink.
        #
        # The approach side is left long on purpose. There is time to move
        # over before the obstacle and none to spare after it, so the two
        # sides do not have to be symmetric.
        #
        # Do not shorten this much further without watching the speed: the
        # controller reads the path's curvature and cuts speed for it, and a
        # steeper return raises curvature at the point where the car is
        # already committed.
        speed = max(1.0, abs(self.odom.twist.twist.linear.x))
        scale = np.clip(1.0 + speed / max(1.0, max(w.vx_mps for w in reference)), 1.0, 1.5)
        approach = np.asarray([-4.0, -3.0, -1.5]) * scale
        retreat = np.asarray([1.2, 2.0, 3.0]) * scale

        # Everything the path has to get past, not just the first thing in the
        # way.
        #
        # The return leg is why this matters. It brings the path back to d=0
        # within retreat[0] of the last obstacle it knows about and holds it
        # there to the end, so an obstacle still near the raceline inside that
        # stretch is ON the published path. The state machine measures exactly
        # that and refuses - "path is at d=0.000, free -0.09" in its logs - and
        # no amount of choosing the apex better changes it, because the apex is
        # not where the collision is.
        #
        # It is also what separated the two failures seen on the car. Two
        # obstacles in a LINE stopped it; two staggered left/right did not.
        # Staggered ones get placed near the walls, where a return leg at d=0
        # clears them; in-line ones sit where the first one was, which is the
        # one place the return leg is guaranteed to go.
        #
        # So the corridor grows while the return leg would land on something:
        # take the next blocking obstacle within retreat[-1] of the current last
        # member, and repeat. What comes out is one offset held from the first
        # obstacle to the last, with the return after all of them.
        # Growth walks EVERY obstacle from the leader onwards, in s order.
        #
        # It used to walk `blocking` - the ones the raceline does not clear -
        # and that is what breaks a zigzag. Right, left, right: the middle one
        # is off to the far side, the raceline clears it, so it never enters
        # `blocking`, growth steps straight from the first to the third, finds
        # them 5 cm apart in d and merges them. The corridor then closes over
        # the top of an obstacle nobody looked at.
        #
        # The leader still has to be something worth avoiding, so it comes
        # from `blocking`. Everything after it is taken in order from
        # `candidates`, and the first one that fails the d test stops the
        # corridor - which for the second obstacle means there is no corridor.
        obstacle = blocking[0]
        leader_ahead = ahead(blocking[0])
        chain = sorted(
            (o for o in in_range if ahead(o) >= leader_ahead - 1e-3), key=ahead)
        gaps = [ahead(o) for o in chain]
        apex_s = obstacle.s_center
        leader_gap = gaps[0]

        def corridor_geometry(members):
            """Offset, rooms and bounds for a corridor covering `members`.

            Unwrapped forward from the leader so the control points stay
            monotonic across the s=0 seam. Room is taken at the TIGHTEST point
            of the hold region, not at the leader: the offset is held over the
            whole of it, so the narrowest track anywhere in there is what
            decides. One member means one sample, i.e. the old check.
            """
            member_s = [apex_s + (o.s_center - apex_s) % max_s for o in members]
            left = max(o.d_left for o in members) + self.evasion_distance
            right = min(o.d_right for o in members) - self.evasion_distance
            bound_left, bound_right = self._tightest_bounds(
                s_values, reference, member_s[0], member_s[-1], max_s)
            return (member_s, left, right,
                    bound_left - left, bound_right + right,
                    bound_left, bound_right)

        group_idx = [0]
        while self.corridor_enabled:
            last = group_idx[-1]
            # Strictly the next one in s. `chain` is sorted, so that is the
            # very next index - no skipping, no searching. Two obstacles at
            # the same s get tested too: if they sit on opposite sides, the d
            # test refuses the corridor, which is the right answer anyway.
            following = last + 1
            if following >= len(chain):
                break
            if gaps[following] > gaps[last] + self.corridor_link_gap_m * scale:
                break
            # Same side and near enough in d to share one offset. Measured
            # against the LEADER, so a corridor cannot drift across the track
            # one small step at a time.
            d_off = chain[following].d_center - obstacle.d_center
            if abs(d_off) > self.corridor_link_d_tol_m:
                self.get_logger().info(
                    f"not linking the obstacle at "
                    f"s={chain[following].s_center:.2f} into the corridor: "
                    f"its d={chain[following].d_center:+.2f} is {abs(d_off):.2f} m "
                    f"from the leader's {obstacle.d_center:+.2f}, over the "
                    f"{self.corridor_link_d_tol_m:.2f} m link tolerance - "
                    f"avoiding it on its own instead",
                    throttle_duration_sec=2.0)
                break
            if gaps[following] - gaps[0] > self.corridor_max_len_m:
                self.get_logger().info(
                    f"obstacle at s={chain[following].s_center:.2f} is "
                    f"{gaps[following] - gaps[0]:.2f} m past the first one, over "
                    f"the {self.corridor_max_len_m:.2f} m corridor limit - "
                    f"planning for the first {len(group_idx)} and letting the "
                    f"next replan pick up the rest",
                    throttle_duration_sec=2.0)
                break
            # Only widen the corridor if the wider one is still drivable.
            #
            # Obstacles on OPPOSITE sides cannot share an offset - covering both
            # means holding max(d_left)+evasion or min(d_right)-evasion, which on
            # a track this wide is off it. Merging them anyway turned the case
            # that already worked (staggered left/right, where the return leg
            # passes between them) into a refusal. So stop the corridor here and
            # publish the shorter one: the return leg clears an obstacle that is
            # far enough off the line, occupied_by refuses the side if it does
            # not, and the next replan gets the rest.
            trial = [chain[i] for i in group_idx + [following]]
            _, trial_left, trial_right, trial_lroom, trial_rroom, _, _ = \
                corridor_geometry(trial)
            if max(trial_lroom, trial_rroom) < self.boundary_margin:
                self.get_logger().info(
                    f"not widening the corridor to the obstacle at "
                    f"s={chain[following].s_center:.2f}: covering it too would "
                    f"need d={trial_left:+.2f} or {trial_right:+.2f}, leaving "
                    f"{trial_lroom:.2f}/{trial_rroom:.2f} m against a "
                    f"{self.boundary_margin:.2f} m margin - planning for the "
                    f"first {len(group_idx)}",
                    throttle_duration_sec=2.0)
                break
            group_idx.append(following)
        group = [chain[i] for i in group_idx]

        (member_s, left_apex, right_apex, left_room, right_room,
         bound_left, bound_right) = corridor_geometry(group)
        first_s, last_s = member_s[0], member_s[-1]

        # End the path BEFORE the obstacle the corridor refused to cover.
        #
        # This is what stopped the car on a zigzag. The corridor correctly
        # refused to link over the middle obstacle, so the plan was the leader
        # alone - but the return leg still ran its full 3 m back to the
        # raceline, straight across the middle obstacle's s. The path was
        # published, and the state machine's own clearance check, which looks
        # at every obstacle and not just the ones near the line, found the path
        # sitting inside that obstacle and refused to drive it. Planner said
        # go, state machine said no, and the car sat there.
        #
        # A slalom is not one path. It is a path around the first obstacle that
        # STOPS before the second, and then a replan. Ending here makes the
        # obstacle fall past the end, which the state machine already knows how
        # to ignore - "not this path's problem" - and the next replan leads
        # with it and goes round the other way.
        #
        # Only shortened, never lengthened, and never below retreat_floor_m:
        # the return leg's curvature is what the controller reads to set speed,
        # and a snapped-back return at the moment the car is committed is worse
        # than no path. If it will not fit, the retreat stays and occupied_by
        # below refuses the side honestly instead.
        # The next obstacle the RACELINE does not clear, not merely the next
        # one in s. The return leg lands on the raceline and stays there, so
        # that is the only kind of obstacle the tail of this path can hit; one
        # sitting out by the wall is not where the return goes, and occupied_by
        # below still vets the whole span against it.
        #
        # Measured on the car, choosing the next obstacle regardless: leader on
        # the line at s=20.86, a cleared obstacle at s=22.15 out at d=+0.84,
        # and the REAL threat at s~22.9 on the line. Ending before the cleared
        # one left 0.83 m of return, under the floor, so the full return was
        # kept - straight across the on-line obstacle behind it. The state
        # machine refused the path every tick ("blocked by obs 210 ... free
        # -0.319") and the car stopped behind an obstacle it had room to pass.
        blocker = next(
            (o for o in chain
             if ahead(o) > gaps[group_idx[-1]] + 1e-3
             and not any(o is member for member in group)
             and not self._raceline_clears(o, cur_d)), None)
        if blocker is not None:
            room = ahead(blocker) - gaps[group_idx[-1]] - self.retreat_clearance_m
            if room < retreat[-1]:
                if room >= self.retreat_floor_m:
                    was = retreat[-1]
                    retreat = retreat * (room / was)
                    self.get_logger().info(
                        f"ending the path {room:.2f} m past the obstacle at "
                        f"s={obstacle.s_center:.2f} instead of {was:.2f} m: "
                        f"the next obstacle is at s={blocker.s_center:.2f} and "
                        f"is not in this corridor - returning to the raceline "
                        f"before it, and the next replan leads with it",
                        throttle_duration_sec=2.0)
                else:
                    self.get_logger().info(
                        f"cannot end the path before the obstacle at "
                        f"s={blocker.s_center:.2f}: only {room:.2f} m of return "
                        f"leg would be left, under the "
                        f"{self.retreat_floor_m:.2f} m floor - keeping the full "
                        f"return and letting the side check decide",
                        throttle_duration_sec=2.0)

        # Control points: approach on the raceline, the offset held at every
        # member (and at corridor_point_spacing_m in between, so the cubic runs
        # flat rather than bulging between two equal knots), then the return.
        # For a single obstacle the hold region is one point and this is exactly
        # the seven-point layout it has always been.
        hold_s = [first_s]
        for s in member_s[1:]:
            while s - hold_s[-1] > self.corridor_point_spacing_m:
                hold_s.append(hold_s[-1] + self.corridor_point_spacing_m)
            if s - hold_s[-1] >= 0.05:
                hold_s.append(s)
            else:
                hold_s[-1] = s
        control_s = np.concatenate(
            (first_s + approach, np.asarray(hold_s, dtype=float), last_s + retreat))
        hold_mask = np.concatenate(
            (np.zeros(len(approach)), np.ones(len(hold_s)), np.zeros(len(retreat))))

        # Where the car is, in the same unwrapped frame as the control points.
        car_s = first_s - leader_gap

        def path_profile(apex_d):
            """The path's lateral profile for a candidate offset.

            The approach knots carry the car's CURRENT offset rather than
            zero, so a path planned while the car is already displaced starts
            from where it is and transitions to the new offset - instead of
            demanding a return to the raceline first. With the car on the line
            this is identical to what it always was.

            The return knots stay at zero: after the last obstacle the
            raceline is where the car should end up.
            """
            d = np.where(hold_mask > 0.5, float(apex_d), 0.0)
            d[:len(approach)] = cur_d
            return CubicSpline(control_s, d, bc_type='natural')

        # The clip has to admit cur_d too. It used to bound the samples to
        # [0, apex_d]; with the car sitting on the OTHER side of the line from
        # the offset being planned, that clipped the start back to zero and
        # reintroduced exactly the jump this is meant to remove.
        def clip_bounds(apex_d):
            return (min(0.0, float(apex_d), cur_d), max(0.0, float(apex_d), cur_d))

        # corridor_geometry above takes the offset from EVERY member, not just
        # the leader. Taking it from the leader alone was the other half of the
        # same bug: a second obstacle reaching a centimetre further towards the
        # line than the first put the leader's apex inside it, that side was
        # refused, and the only side left was the squeeze between the leader and
        # the wall - which never has room. The answer to "another obstacle
        # overlaps my apex" is to move the apex over, not to give up the side.

        # Do not dodge INTO another obstacle - but only judge that where the
        # path actually is.
        #
        # Two mistakes were in the first version of this and both mattered.
        #
        # It compared the APEX value against obstacles metres away in s, where
        # the path has long since returned to the raceline. The apex is a
        # single point; the path is a cubic that decays to zero by the last
        # control point. Below, the path's real offset at that obstacle's s is
        # what gets tested - a spline on fixed knots is linear in its control
        # values, and only the apex is non-zero, so path_d = apex_d * shape(s).
        #
        # And it vetoed a side even when the raceline was no better. That is
        # what made two obstacles on the SAME side unsolvable: dodging the
        # near one left put the apex inside the far one's widened band, and
        # swinging right put it there too, so both sides were refused and
        # nothing was published at all - the car crept up to an obstacle it
        # had a clear way around. On the car: right-then-right and
        # left-then-left stalled, while right-then-left and left-then-right
        # were fine, because in the staggered cases one side always survived.
        #
        # So the test is comparative: refuse a side only if it leaves LESS
        # room at some other obstacle than simply staying on the raceline
        # would have.
        #
        # Group members are exempt. The offset above is built to clear all of
        # them, and the corridor holds it across all of them, so testing them
        # here would veto the very path that solves them. What is left is what
        # the veto was always meant to catch - something the corridor does not
        # cover and cannot be widened to cover - and a refusal is now honest:
        # the manoeuvre really is not expressible as one corridor.
        span_start = leader_gap + approach[0]
        span_end = gaps[group_idx[-1]] + retreat[-1]
        # The apex_d * shape(s) shortcut is gone: with the approach knots
        # carrying cur_d the profile is no longer linear in apex_d alone. Two
        # splines over ten knots per plan is not worth an optimisation that
        # would now be wrong.

        def clearance(d, other):
            """Signed room between a path at d and an obstacle. Negative inside."""
            if d >= other.d_left:
                return d - other.d_left
            if d <= other.d_right:
                return other.d_right - d
            return -min(d - other.d_right, other.d_left - d)

        def occupied_by(apex_d):
            profile = path_profile(apex_d)
            lo, hi = clip_bounds(apex_d)
            # in_range, not candidates. This is the last thing that looks at
            # the path before it is published, and it has to look at the same
            # obstacles the state machine will.
            #
            # candidates is filtered to trajectory_threshold (0.6 m) - near
            # enough to the line to be worth PLANNING around. That is the
            # wrong question here. The return leg leaves the apex and decays
            # to the raceline over several metres, so it spends that whole
            # distance out where an obstacle at d=0.7 lives, and one of those
            # is invisible to every other check too: the blocker test only
            # truncates for obstacles the raceline does not clear, and one out
            # at 0.7 is cleared by definition.
            #
            # Measured over two runs, the obstacle at s~22.6 reported
            # d=+0.52, +0.63, +0.68, +0.74, +0.77, +0.79 on different frames.
            # It straddles the threshold. On the frames it fell outside,
            # nothing in this planner looked at it at all while the return leg
            # from the obstacle before it went past.
            #
            # Safe to widen only because the veto now has an absolute
            # condition. Comparative-only, this would refuse a side for any
            # obstacle anywhere on it.
            for other in in_range:
                if any(other is member for member in group):
                    continue
                other_ahead = ahead(other)
                if not span_start <= other_ahead <= span_end:
                    continue
                # Clipped exactly as the samples are below. Without this the
                # veto reads the natural cubic's undershoot in the approach ramp
                # - which is on the far side of the raceline and gets clipped
                # away before anything is published - and refuses a side over an
                # offset the path never has.
                path_d = float(np.clip(
                    profile(apex_s + other_ahead - leader_gap), lo, hi))
                # Two conditions, and BOTH have to hold. The path must be
                # genuinely too close, and it must be worse than the raceline.
                #
                # The comparative half alone is what refused a drivable side.
                # Measured on the car: leader at s=21.17 with the only room on
                # the left, and an obstacle back at s=15.78 with its near edge
                # at d=+0.32. The path passes it at d=+0.05 - 27 cm of room,
                # more than evasion_distance - and it was vetoed anyway,
                # because the raceline would have been 32 cm clear and 27 is
                # less than that. Both sides refused, no path published, car
                # stopped in front of an obstacle it had a metre of room past.
                #
                # evasion_distance is the clearance this planner builds every
                # apex to, so it is the honest threshold for "too close" here
                # as well. Above it, being a few centimetres tighter than the
                # raceline is simply what going round something costs.
                #
                # The comparative half stays: it is what stops the planner
                # choosing a side that walks INTO an obstacle the raceline
                # would have cleared. The case above still refuses when the
                # path is actually inside the band - measured, d=-0.64 against
                # an obstacle spanning -0.77..-0.39.
                #
                # The 5 cm is unchanged. Between control points the cubic
                # overshoots by a centimetre or two even where the path is
                # nominally back on the raceline, and a side must not be
                # chosen on that.
                room = clearance(path_d, other)
                # A floor the comparison cannot argue its way past.
                #
                # clearance() measures the path's CENTRELINE to the obstacle's
                # edge. The car is 0.28 m wide, so it fits only if room is at
                # least its half width plus the margin the state machine
                # insists on - which is what min_path_clearance_m is.
                #
                # Measured on the car, and it is why the comparison alone is
                # not enough: leader at s=25.09 out at d=-0.49, so the corridor
                # went left to d=+0.60, and a second obstacle 0.6 m later
                # spanning d=-0.03..+0.45. The path passed its edge with
                # 0.129 m - less than the car's half width, so the body
                # overlapped it by a centimetre - and the comparison waved it
                # through, because the raceline runs THROUGH that obstacle
                # (clearance -0.03) and 0.129 is better than that. Better than
                # the raceline is not the same as passable. The state machine
                # refused the path every tick ("free -0.010 m") and the car
                # drove into it anyway on the path it had already been given.
                #
                # Below the floor there is no path on that side, and if that
                # leaves neither side the honest answer is no path at all -
                # those two obstacles straddle the line 0.6 m apart and there
                # is genuinely no way between them.
                if room < self.min_path_clearance_m:
                    return other, path_d
                # Above the floor the car fits, and the only remaining
                # question is whether it is worse than staying on the line.
                if (room < self.evasion_distance
                        and room < clearance(0.0, other) - 0.05):
                    return other, path_d
            return None

        left_taken = occupied_by(left_apex)
        right_taken = occupied_by(right_apex)
        left_ok = left_room >= self.boundary_margin and left_taken is None
        right_ok = right_room >= self.boundary_margin and right_taken is None
        for taken, name in ((left_taken, 'left'), (right_taken, 'right')):
            if taken is not None:
                other, path_d = taken
                self.get_logger().info(
                    f"not going {name} of the obstacle at s={apex_s:.2f}: that path "
                    f"passes the obstacle at s={other.s_center:.2f} "
                    f"(d {other.d_right:+.2f}..{other.d_left:+.2f}) at d={path_d:+.2f}, "
                    f"leaving {clearance(path_d, other):+.2f} m against a "
                    f"{self.min_path_clearance_m:.2f} m floor",
                    throttle_duration_sec=2.0)
        # Stick to the side already chosen unless the other is clearly better.
        #
        # The two are often within a couple of centimetres of each other, and
        # the obstacle's measured edges move by about that much between frames,
        # so a bare comparison flips sides several times a second. Each flip is
        # a completely different path, the state machine sees a path the car is
        # not on, and avoidance never settles. Requiring the other side to win
        # by side_hysteresis_m makes the choice sticky without making it
        # permanent - a genuinely wider gap still takes it.
        bias = 0.0
        if self.last_side == 'left':
            bias = self.side_hysteresis_m
        elif self.last_side == 'right':
            bias = -self.side_hysteresis_m

        # "The raceline already clears it" is decided up front now, on the whole
        # candidate list rather than on whichever obstacle happened to be
        # nearest. Deciding it here meant a leader that was clear of the line
        # aborted the plan for the ones behind it that were not.
        if left_ok and (not right_ok or left_room + bias >= right_room):
            side, apex_d = 'left', left_apex
        elif right_ok:
            side, apex_d = 'right', right_apex
        else:
            return self._bail(
                out,
                f"no room either side of the {len(group)} obstacle(s) from "
                f"s={first_s % max_s:.2f} to s={last_s % max_s:.2f}: "
                f"left {left_room:.2f} m{' (taken)' if left_taken else ''}, "
                f"right {right_room:.2f} m{' (taken)' if right_taken else ''}, "
                f"need {self.boundary_margin:.2f} "
                f"(corridor must hold d={left_apex:+.2f} or {right_apex:+.2f}, "
                f"tightest track {bound_left + bound_right:.2f} m, "
                f"evasion_distance {self.evasion_distance:.2f})")

        # An offset of nothing is the raceline, and the raceline is what the
        # state machine already said was blocked.
        #
        # _raceline_clears bails when the near edge is raceline_clearance_m off
        # the line, and the offset is that same edge plus evasion_distance -
        # the two numbers are equal on purpose, so an obstacle a millimetre
        # inside the bail threshold produces an offset of a millimetre. The log
        # showed exactly that: "corridor d=0.00 m", published, then refused as
        # "path is at d=+0.005, free -0.169". Say there is nothing to avoid
        # instead of publishing the line and having it thrown back.
        if abs(apex_d) < 0.02:
            return self._bail(
                out,
                f"the offset needed at s={apex_s:.2f} is only {apex_d:+.3f} m - "
                f"that is the raceline, which the state machine has already "
                f"called blocked. Nothing to publish")

        spline = path_profile(apex_d)
        clip_lo, clip_hi = clip_bounds(apex_d)
        # Sampling starts AT THE CAR, not at the first control point.
        #
        # The approach knots reach 4*scale behind the leader, which is usually
        # behind the car as well. They still shape the spline - dropping them
        # would leave two knots a few centimetres apart and a violent
        # transition - but nothing behind the car is worth publishing. The car
        # has been there.
        sample_from = max(control_s[0], car_s)
        sample_unwrapped = np.arange(sample_from, control_s[-1], self.resolution)
        sample_d = np.clip(spline(sample_unwrapped), clip_lo, clip_hi)
        sample_s = sample_unwrapped % max_s
        xy = self.converter.get_cartesian(sample_s, sample_d)

        # Heading and curvature OF THE SPLINE, not of the raceline underneath
        # it. Copying base.psi_rad / base.kappa_radpm told the controller the
        # evasion path was as straight as the line it departs from, and the
        # controller acts on both: curvature sets the L1 lookahead and scales
        # the speed down for a bend, and psi is what the heading-error speed
        # cut compares the car against. An evasion swerve read as straight is
        # taken too fast, with too long a lookahead.
        #
        # Both come from the cartesian samples, so they describe the path the
        # car will actually drive. psi is atan2(dy, dx), which is the same
        # convention the waypoints already carry - the raceline generator adds
        # pi/2 to tph's y-axis heading for exactly this (raceline/map_processing
        # conv_psi). Curvature is (x'y'' - y'x'') / (x'^2 + y'^2)^1.5, which is
        # invariant to the sample spacing, so the gradients need no scaling.
        px = np.asarray(xy[0], dtype=float)
        py = np.asarray(xy[1], dtype=float)
        if px.size >= 3:
            dx, dy = np.gradient(px), np.gradient(py)
            ddx, ddy = np.gradient(dx), np.gradient(dy)
            psi_path = np.arctan2(dy, dx)
            denom = np.power(dx * dx + dy * dy, 1.5)
            kappa_path = np.divide(dx * ddy - dy * ddx, denom,
                                   out=np.zeros_like(denom), where=denom > 1e-9)
        else:
            psi_path = np.zeros(px.size)
            kappa_path = np.zeros(px.size)

        for idx, (s_m, d_m) in enumerate(zip(sample_s, sample_d)):
            waypoint_idx = int(np.argmin(np.abs(s_values - s_m)))
            base = reference[waypoint_idx]
            available = base.d_left if d_m >= 0.0 else base.d_right
            if abs(d_m) > max(0.0, available - self.boundary_margin):
                # Where along the spline it fails matters more than that it
                # does. Near the apex means the obstacle sits too close to a
                # wall. Well past it means the path is still offset while the
                # track narrows underneath it - the excursion is too long for
                # this track, not too wide.
                past_apex = (s_m - apex_s + max_s / 2.0) % max_s - max_s / 2.0
                return self._bail(
                    out,
                    f"path leaves the track at s={s_m:.3f} "
                    f"({past_apex:+.2f} m from the apex at s={apex_s:.2f}): "
                    f"needs d={d_m:.3f} m, bound {available:.3f} m "
                    f"less {self.boundary_margin:.2f} margin")
            out.wpnts.append(Wpnt(
                id=idx, s_m=float(s_m), d_m=float(d_m),
                x_m=float(xy[0, idx]), y_m=float(xy[1, idx]),
                psi_rad=float(psi_path[idx]), kappa_radpm=float(kappa_path[idx]),
                vx_mps=base.vx_mps, ax_mps2=base.ax_mps2,
                d_left=base.d_left, d_right=base.d_right,
            ))
        out.ot_side = side
        out.ot_line = side
        self.last_side = side
        self.get_logger().info(
            f"avoiding {side} around {len(group)} obstacle(s) at "
            f"s={', '.join(f'{o.s_center:.2f}' for o in group)}: "
            f"corridor d={apex_d:.2f} m held over s {first_s:.2f}..{last_s:.2f}, "
            f"{len(out.wpnts)} waypoints over "
            f"s {control_s[0]:.2f}..{control_s[-1]:.2f}",
            throttle_duration_sec=2.0)
        return out

    def _markers(self, path):
        markers = MarkerArray()
        if not path.wpnts:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.action = Marker.DELETEALL
            markers.markers.append(marker)
            return markers
        for waypoint in path.wpnts:
            marker = Marker()
            marker.header = path.header
            marker.ns = 'spline_path'
            marker.id = waypoint.id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = waypoint.x_m
            marker.pose.position.y = waypoint.y_m
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.08
            marker.color.a = 1.0
            marker.color.g = 1.0
            markers.markers.append(marker)
        return markers


def main(args=None):
    rclpy.init(args=args)
    node = SplineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
