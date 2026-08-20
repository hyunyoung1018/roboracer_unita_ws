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
from f110_msgs.msg import ObstacleArray

from .state_machine_node import StateMachine, time_to_float


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


class H2HStateMachine(StateMachine):
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
        # Mirror of head_to_head.launch.xml's `dynamic_avoidance` argument. The
        # GP predictor chain is too heavy for the Orin Nano, so it is launched
        # with prediction:=false dynamic_avoidance:=false and nothing publishes
        # /planner/avoidance/otwpnts. _check_latest_wpnts already refuses a
        # topic with no publisher, but it refuses it every tick, after running
        # _check_ot_sector and _check_getting_closer first. Saying so once here
        # is both cheaper and honest about which branch is switched off.
        #
        # Declared on the wrapper rather than as dynamic_avoidance_planner.
        # enabled, because _load_planner_configs only declares the keys that
        # exist in config/planners/dynamic_avoidance_planner.yaml - an `enabled`
        # key in the head-to-head overlay alone is never declared, so
        # get_planner_param would fall back to the package yaml and silently
        # keep the planner on.
        self.dynamic_avoidance_enabled = bool(
            self._get_or_declare("dynamic_avoidance_enabled", True))
        # [m] How far off the raceline an obstacle may sit and still be
        # something this car trails or plans around. Beyond it, it is scenery.
        #
        # The car drove into a wall following a pillar. Every candidate list in
        # this wrapper was "the nearest thing ahead" with no lateral test at
        # all, so a desk leg 0.8 m off the line - which the raceline clears by
        # a wide margin and no planner would ever move for - became the
        # trailing target, and head to head trails at 0.8 m. time_trials never
        # met this: its trailing target is only ever an obstacle that actually
        # blocks the raceline, and it holds 2.5 m.
        #
        # 0.6 is spline_node's trajectory_threshold, deliberately: the planner
        # already uses exactly this number to decide which obstacles are worth
        # looking at, and a target the planner will not plan for is not a
        # target this should offer either.
        self.trailing_lateral_threshold_m = float(
            self._get_or_declare("trailing_lateral_threshold_m", 0.6))
        # [m] How far AHEAD of the car the static avoidance path may begin and
        # still count as usable. See _check_on_spline and get_splini_wpts.
        #
        # Sized from the geometry it exists to cover. spline samples from
        # max(control_s[0], car_s), so while the car is still behind the first
        # approach knot the path begins 4*scale metres before the obstacle,
        # with scale = clip(1 + v/v_max, 1, 1.5). The shared on-spline test
        # demands the car be within on_spline_min_dist_thres_m (1.5 m) of it,
        # which is only true from about 4*1.5 + 1.5 = 7.5 m out - while
        # trailing has been braking since 11.8 m at 3 m/s. 4.0 covers the whole
        # of that window and stops well short of the 9 m the car can even see.
        self.static_path_lead_in_m = float(
            self._get_or_declare("static_path_lead_in_m", 4.0))
        self._last_gb_blocked_at = None
        self._last_trailing_target = None
        self._last_trailing_target_at = None
        # [s] How long the opponent list is believed. It arrives in the same
        # tracker callback as /tracking/obstacles, so in normal running it is
        # never older than a frame; this only covers the stream stopping.
        self.opponent_stream_timeout_sec = float(
            self._get_or_declare("opponent_stream_timeout_sec", 0.5))
        # The one selected opponent, as h2h_tracking_node chose it. The state
        # machine reads /tracking/obstacles, which carries raw tracker ids, so
        # the opponent cannot be named by id there - it is matched by position
        # instead, and both copies come from the same track in the same frame.
        self._opponent_obstacles = []
        self._opponent_stamp = None
        self.create_subscription(
            ObstacleArray, "/tracking/dynamic_obstacles",
            self._dynamic_obstacles_cb, 10)
        # Per-tick memo for the shared _check_free_frenet; see _memo_free_frenet.
        # Keyed by planner name because the cache objects belong to the shared
        # state machine and this wrapper does not add fields to them.
        self._loop_seq = 0
        self._free_memo = {}

    def _get_or_declare(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def loop(self):
        """Stamp the tick, then run the shared loop unchanged.

        The counter is what makes the free-check memo safe: it says "this is a
        new tick, throw the answers away". Incremented before the shared loop
        rather than after, so every check inside this tick shares one number.
        """
        self._loop_seq += 1
        super().loop()

    def _memo_free_frenet(self, wpnts_data):
        """Run the shared free check at most once per cache per tick.

        One tick asks the same cache the same question up to three times:
        ObstacleTransition checks the raceline at its top and again at its
        bottom, and the static branch is checked by both
        _check_overtaking_mode_sustainability and _check_static_overtaking_mode.
        Each of those walks every obstacle against every point of the path.

        Caching is only sound while the inputs cannot have moved, so the key
        carries three things:

          _loop_seq        a new tick invalidates everything
          init_count       _check_latest_wpnts re-initialises a cache mid-loop,
                           and several callers run it immediately before asking
          is_init          _expire_stale_cache and the source-change rule clear
                           it, which flips the answer for an OT cache

        Nothing else can change underneath: cur_obstacles_in_interest is fixed
        for the tick by update_waypoints, and rclpy's default executor runs no
        subscription callback while loop() is on the stack.

        Only the shared computation is memoised. Everything this wrapper does
        around it - the width swap, the beyond-path correction, the blocked-
        raceline hold - still runs on every call, exactly as often as before,
        so no side effect of theirs is skipped. closest_target, closest_gap and
        free_dbg are left on the cache by the first call and are what a repeat
        computation would have produced anyway.
        """
        key = (self._loop_seq, wpnts_data.init_count, wpnts_data.is_init)
        entry = self._free_memo.get(wpnts_data.name)
        if entry is not None and entry[0] == key:
            return entry[1]
        is_free = super()._check_free_frenet(wpnts_data)
        self._free_memo[wpnts_data.name] = (key, is_free)
        return is_free

    def _nearest_obstacle_is_static(self) -> bool:
        """Is the closest thing ahead of the car a static obstacle?

        No obstacles, or a dynamic one nearest, both answer False - which keeps
        the dynamic-first behaviour for every case this cannot separate.
        """
        _, nearest = nearest_ahead(
            self.cur_obstacles_in_interest, self.cur_s, self.track_length,
            self.track_length)
        return bool(nearest is not None and nearest.is_static)

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

        With dynamic avoidance switched off at the launch there is no planner
        to enter OVERTAKE on, so refuse before the shared checks run at all.
        Static avoidance goes through _check_static_overtaking_mode and is
        untouched by this.

        AND, when the dynamic branch does arm, decide which planner should own
        the manoeuvre. The shared transitions ask

            _check_overtaking_mode() or _check_static_overtaking_mode()

        and `or` short-circuits, so a dynamic overtake meant the static branch
        was never evaluated at all: static avoidance sat structurally beneath
        dynamic overtaking, and a box the car was about to hit lost to an
        opponent further away. Resolved here rather than in the shared
        transitions, which time_trials also imports.

        The choice is by distance, not by kind - whichever obstacle the car
        reaches first is the one whose planner should be driving - and it only
        decides anything when BOTH branches arm. When the dynamic branch does
        not arm this returns False without touching the static branch, and the
        caller's `or` runs it exactly as it always did.

        static_overtaking_mode is the latch that tells _src_cache and
        _check_overtaking_mode_sustainability whose path OVERTAKE slices, so it
        is set explicitly on every path that returns True from here.
        """
        if not self.dynamic_avoidance_enabled:
            return False
        allowed = super()._check_overtaking_mode()
        if allowed and self.force_trailing:
            self.get_logger().info(
                "overtake entry vetoed: predictor asked for force_trailing",
                throttle_duration_sec=2.0)
            allowed = False
        if not allowed:
            return False

        # Dynamic armed. Ask the static branch too - the caller never will now.
        static_ok = self._check_static_overtaking_mode()
        prefer_static = static_ok and self._nearest_obstacle_is_static()
        if prefer_static:
            self.get_logger().info(
                "static obstacle is nearer than the opponent: static avoidance "
                "takes the overtake", throttle_duration_sec=2.0)
        self.static_overtaking_mode = bool(prefer_static)
        return True

    def _dynamic_obstacles_cb(self, msg):
        self._opponent_obstacles = list(msg.obstacles)
        self._opponent_stamp = self.now_sec()

    def _opponent_positions(self):
        """Frenet centres of the selected opponent, or empty."""
        if not self._opponent_obstacles or self._opponent_stamp is None:
            return []
        if self.now_sec() - self._opponent_stamp > self.opponent_stream_timeout_sec:
            return []
        return [(float(obs.s_center), float(obs.d_center))
                for obs in self._opponent_obstacles]

    def _is_opponent(self, obstacle, positions):
        """Match by position, because the two streams disagree about the id.

        /tracking/dynamic_obstacles republishes the opponent under
        logical_opponent_id so the predictor sees a stable name across tracker
        id churn; /tracking/obstacles, which this node reads, carries the raw
        tracker id. Both copies are made from the same track in the same
        tracker callback and neither rewrites the centre, so the match is
        exact - the millimetre tolerance is for the float round-trip, not for
        any real ambiguity.
        """
        return any(
            abs(float(obstacle.s_center) - s) < 1e-3
            and abs(float(obstacle.d_center) - d) < 1e-3
            for s, d in positions)

    def _near_the_line(self, obstacle):
        """Is this obstacle close enough to the raceline to matter at all?

        Scenery is not an obstacle. The lidar returns pillars, table legs and
        anything else standing near the wall, and on this track those are
        metres off the driving line - the raceline clears them, no planner
        moves for them, and nothing should trail them either.
        """
        return abs(float(obstacle.d_center)) <= self.trailing_lateral_threshold_m

    def _avoidable_obstacles(self):
        """What the static planner is actually planning around.

        The contents of /tracking/static_obstacles - everything except the one
        selected opponent, which has a planner of its own - narrowed to what is
        near the line, which is the same narrowing spline_node applies to that
        stream before it plans. Without it a pillar by the wall satisfies the
        closing gate on a lap where nothing is really in the way; path_free
        catches that afterwards, so it was the milder of the two leaks, but it
        is the same leak.
        """
        positions = self._opponent_positions()
        return [obs for obs in self.cur_obstacles_in_interest
                if self._near_the_line(obs)
                and not self._is_opponent(obs, positions)]

    def _closing_on_nearest_avoidable(self, threshold_m) -> bool:
        """Is the car closing on the nearest obstacle the static path is for?

        The shared gate asks _check_getting_closer, which measures the nearest
        obstacle of any kind. Trailing an opponent, that is the opponent - so
        an opponent pulling away faster than 0.5 m/s answered "not closing" and
        refused the static avoidance, with a box sitting three metres ahead
        that the car was very much closing on. The opponent's speed has nothing
        to say about whether to drive around a box.

        It used to exclude the opponent by taking only is_static obstacles, and
        that made this gate depend on a classification it has no business
        depending on. Measured on 2026-08-19, with nothing but boxes on the
        track: 39 of 41 tracks read DYNAMIC for their whole life, because a
        stationary box measures 0.5 to 2.9 m/s of apparent longitudinal speed
        from a moving car. The static list came out empty, nearest_ahead
        returned nothing, and this answered False - 34 of 35 refusals that run,
        nine of them with have_path, path_free and worth_driving all true. The
        planner had done its job and the gate threw it away.

        So it excludes the opponent by NAME instead, which is what it always
        meant. The list is now the same one /tracking/static_obstacles carries,
        and the same principle the split already runs on: time trials' planner
        never looks at is_static, so nothing here should either.

        And the target's own vs is deliberately not read. The shared form is
        `cur_vs - target.vs > -0.5`, which on the same measurement that breaks
        the classification breaks this too: a box misread as moving carries a
        misread speed with it, and 1.0 - 2.4 answers "not closing" for a box
        the car is driving straight at. Excluding the opponent and then
        believing the speed of what is left would have moved the failure
        rather than fixed it.

        Treating them as stationary is not a simplification, it is the same
        assumption the planner behind this gate already makes: spline_node puts
        a fixed apex at a fixed s and holds it. If something in that list is
        genuinely moving and is not the opponent, the geometry check does the
        refusing - path_free is measured against where things actually are.
        """
        _, target = nearest_ahead(
            self._avoidable_obstacles(), self.cur_s, self.track_length,
            threshold_m)
        if target is None:
            return False
        return bool(self.cur_vs > -0.5)

    def _check_static_overtaking_mode(self) -> bool:
        """The shared gate, with the two inputs head to head gets wrong.

        Reimplemented rather than wrapped because both corrections are to what
        goes IN, and the shared version computes its inputs internally and
        returns one bool. state_machine_node.py is what time_trials runs, so it
        is not touched; the four conditions and the speed limit are mirrored
        exactly, and only these change:

          closing    measured against the nearest obstacle the static planner
                     is actually planning around, rather than the nearest
                     obstacle of any kind - see _closing_on_nearest_avoidable.
          path_free  a distant opponent no longer vetoes it - see
                     _blocked_only_by_the_opponent. That correction lives
                     in _check_free_frenet so that OVERTAKE's per-tick
                     sustainability check gets it too, not just this entry gate.

        Also clears the static/dynamic latch, which the shared code only ever
        cleared on a successful *dynamic* check: once static avoidance had armed
        once, a later loop where both checks failed left it true and the next
        OVERTAKE read the static cache for what may be a dynamic opponent.
        """
        slow_enough = self.cur_vs < self.static_overtake_max_speed_mps
        # The horizon the car SEES on, not a number of its own.
        #
        # This was a literal 7.0 while interest_horizon_m is 9.0, and the two
        # metres between them are a hole the car falls into on every straight.
        # An obstacle enters cur_obstacles_in_interest at 9 m and becomes the
        # trailing target on the same tick, and trailing's command is
        #
        #     clip(v_opp - P (gap_should - gap) - D (v_ego - v_opp), 0, v_path)
        #
        # which stops being clipped to v_path - stops braking, that is - only
        # inside
        #
        #     gap = trailing_gap + vel_gain v_ego + (v_path - v_opp
        #                                            + D (v_ego - v_opp)) / P
        #
        # With the head-to-head gains (P 0.5, D 0.25, vel_gain 0.10) behind a
        # stationary box that is 9.30 m at 2.5 m/s and 10.60 m at 3.0 m/s -
        # further out than the car can see. So above about 2.4 m/s the obstacle
        # is already inside the braking region the instant it appears, and for
        # the next two metres this gate answered False and static avoidance
        # could not arm. Brake, then release and swerve, every time.
        #
        # Tying it to interest_horizon_m closes the hole by construction: the
        # gate can no longer refuse something the state machine can see. It
        # only OPENS the gate - have_path, path_free and worth_driving are
        # untouched and still do the actual refusing, and the planner has had a
        # path since 10 m (spline's lookahead), so there is one to check.
        #
        # The other two ways to close it were both worse. Raising
        # interest_horizon_m moves first sight but not the braking point, so
        # the car brakes at the new horizon instead - and it widens the
        # obstacle list every other check pays for. Raising trailing_p_gain
        # moves the braking point but also changes how the car follows a moving
        # opponent, which is the one behaviour on this car that measured good.
        closing = self._closing_on_nearest_avoidable(
            threshold_m=self.interest_horizon_m)
        have_path = self._check_latest_wpnts(
            self.static_avoidance_wpnts, self.cur_static_avoidance_wpnts)
        path_free = self._check_free_frenet(self.cur_static_avoidance_wpnts)

        # _worth_driving, not path_free on its own. Refusing a marginal
        # avoidance does not hold the car back, it drops the transition
        # through to the raceline - the line that was declared blocked. The
        # shared node made that change and this override has to carry it or
        # head to head keeps the old behaviour on static obstacles.
        worth = self._worth_driving(self.cur_static_avoidance_wpnts, path_free)
        allowed = bool(slow_enough and closing and have_path and worth)
        self.static_overtaking_mode = allowed
        if not allowed and len(self.cur_obstacles_in_interest) != 0:
            self.get_logger().info(
                "static avoidance refused: "
                f"slow_enough={slow_enough} "
                f"(vs {self.cur_vs:.2f} < {self.static_overtake_max_speed_mps:.2f}), "
                f"closing={closing}, have_path={have_path}, "
                f"path_free={path_free}, worth_driving={worth}",
                throttle_duration_sec=2.0)
        return allowed

    def _sustainability_detail(self, cache, available, path_free) -> str:
        """Name the thing that refused, not just the term it refused under.

        `available=False` and `path_free=False` are each three or four
        different situations wearing the same word. Availability can fail
        because the planner stopped publishing, because its last path is too
        old, or because the car ran off the end of the one it was following;
        the free check can fail on any obstacle in the list, and which one it
        was decides whether the abort was reasonable.
        """
        bits = []
        if not available:
            stamp = getattr(cache, "stamp", None)
            if stamp is None:
                bits.append("never published")
            else:
                age = self.now_sec() - time_to_float(stamp)
                bits.append(f"path {age:.2f}s old "
                            f"(hyst {cache.hyst_timer_sec}, "
                            f"kill {cache.killing_timer_sec})")
                bits.append(f"on_spline={self._check_on_spline(cache)}")
        if not path_free:
            debug = getattr(cache, "free_dbg", None)
            blocked = [rec for rec in (debug or {}).get("obs", [])
                       if rec.get("blocked")]
            if not blocked:
                bits.append("no path in the cache to judge")
            for rec in blocked[:2]:
                bits.append(
                    f"obs {rec.get('id')} via {rec.get('branch')} "
                    f"free={rec.get('free_dist')} at {rec.get('gap')}m")
        return (" - " + "; ".join(bits)) if bits else ""

    def _check_overtaking_mode_sustainability(self) -> bool:
        """Hand the manoeuvre over instead of abandoning it on a reclassification.

        ``static_overtaking_mode`` decides which planner's cache OVERTAKE
        slices, and it is only ever set by the two ENTRY checks - which
        OvertakingTransition does not call. So the source is chosen once, on
        entry, and cannot change until the car has left OVERTAKE.

        That is wrong for exactly one situation, and it is a situation this
        workspace is built to meet: the opponent stops behind a static
        obstacle, is reclassified STATIC once its speed has been under
        static_speed_threshold for static_min_samples frames, and the car
        begins a static evasion around it - and then it drives off again.
        h2h_tracking_node moves it to /tracking/dynamic_obstacles on the first
        frame the speed estimate clears that threshold, spline_node stops
        seeing it and stops refreshing the static path, and the free check now
        watches the opponent drive along the very line the car is committed to.
        Sustainability fails and the car drops to
        TRAILING mid-evasion, off the raceline and alongside a moving car.

        The other planner has a live path for precisely that case, because the
        object is now dynamic. Switch to it rather than giving up. Both the
        availability and the free check still have to pass on the new source -
        this defers the decision to them, it does not override either.

        The committed source's two terms are evaluated separately rather than
        short-circuited, so a refusal can say which one refused - the same
        shape _check_static_overtaking_mode already uses, and for the same
        reason: from outside the car every abort looks identical. Without it
        the only thing an aborted overtake tells you is that it aborted.
        """
        committed_static = bool(self.static_overtaking_mode)
        if committed_static:
            src, cache = self.static_avoidance_wpnts, self.cur_static_avoidance_wpnts
        else:
            src, cache = self.avoidance_wpnts, self.cur_avoidance_wpnts

        available = bool(self._check_availability(src, cache))
        path_free = bool(self._check_free_frenet(cache))
        # Same test as the entry gate, on whichever cache is committed. With
        # the comparison only on entry the car flicked out of OVERTAKE the
        # moment the path went marginal and had to earn it back.
        if available and self._worth_driving(cache, path_free):
            return True

        self.get_logger().info(
            f"OVERTAKE dropping [{cache.name}]: "
            f"available={available}, path_free={path_free}"
            f"{self._sustainability_detail(cache, available, path_free)}",
            throttle_duration_sec=0.5)

        target_static = not self.static_overtaking_mode
        if target_static:
            wpnts, data = self.static_avoidance_wpnts, self.cur_static_avoidance_wpnts
        else:
            if not self.dynamic_avoidance_enabled:
                return False
            wpnts, data = self.avoidance_wpnts, self.cur_avoidance_wpnts
        if not getattr(data, "enabled", True):
            return False

        # _check_latest_wpnts, not _check_availability. Two reasons, and the
        # first one is fatal: availability reads wpnts_data.stamp, which is None
        # until initialize_traj has run, and the cache being handed TO may never
        # have been initialised - time_to_float(None) took the node down.
        # Second, handing over is an entry into that source, not the
        # continuation of one, so it should demand a fresh path the way
        # _check_overtaking_mode does rather than accept a stale cache.
        if self._check_latest_wpnts(wpnts, data) and self._worth_driving(
                data, self._check_free_frenet(data)):
            self.static_overtaking_mode = target_static
            self.get_logger().info(
                "overtake source handed over to "
                f"{'static' if target_static else 'dynamic'} avoidance - the "
                "committed cache stopped being usable, this one is",
                throttle_duration_sec=1.0)
            return True
        return False

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
        """The nearest thing ahead that is worth trailing.

        This is the fallback for "the free check found no blocking target but
        the raceline was blocked a moment ago", and it used to take the nearest
        obstacle of any kind, at any lateral offset. That is how a pillar by
        the wall became the trailing target and the car followed it into the
        wall at the 0.8 m head-to-head gap.

        The intent was only ever to survive a frame where the detector or the
        id churned - to keep hold of a target that was already there. Something
        the raceline passes by half a metre was never that target, so the list
        is narrowed to what is near the line before the nearest is taken.

        The selected opponent is exempt from that narrowing, and it has to be:
        it IS the trailing target when there is one, and the corridor gate that
        selected it allows it anywhere its footprint fits between the track
        bounds - about 0.75 m on this track, wider than this threshold. An
        opponent that swings wide to overtake would otherwise stop being
        trailed at the moment it draws alongside.
        """
        positions = self._opponent_positions()
        gap, target = nearest_ahead(
            [obs for obs in self.cur_obstacles_in_interest
             if self._near_the_line(obs) or self._is_opponent(obs, positions)],
            self.cur_s,
            self.track_length,
            self.interest_horizon_m,
        )
        return gap, target

    def _planner_validated_span(self, wpnts_data):
        """Distance ahead over which the lane-change planner checked its path.

        Only the dynamic planner publishes /planner/avoidance/merger, and only
        its own cache is bounded by it. The static spline planner avoids a
        stationary obstacle and its path is checked to its end, as before.

        merger carries no stamp, but it is published in the same callback as
        the path it describes, so it is exactly as fresh as the cache being
        judged - and that cache's own staleness is already the job of
        _check_availability.
        """
        if wpnts_data is not self.cur_avoidance_wpnts:
            return None
        if not self.merger or len(self.merger) < 1 or not self.max_s:
            return None
        return (float(self.merger[0]) - self.cur_s) % self.max_s

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

        Poses past the end of what the PLANNER validated are dropped. The
        lane-change planner only holds its evasion offset over
        prediction_span_m of the opponent's future (3 m; its own measurement
        says a 6 m corridor is clear on 2% of test_213 against 53% at 3 m), and
        it publishes the s it validated to as obs_end on
        /planner/avoidance/merger. The shared state machine has always
        subscribed to that topic and stored it in self.merger without ever
        reading it. Checking further out than the planner promised refuses
        paths that were correctly planned.
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

        validated = self._planner_validated_span(wpnts_data)
        if validated is not None:
            within = ((pred_s - self.cur_s) % self.max_s) <= validated
            if np.any(within):
                pred_s, pred_d = pred_s[within], pred_d[within]

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

    def _blocked_only_beyond_path(self, wpnts_data) -> bool:
        """Was the path refused solely by obstacles past its own end?

        The shared free check already answers this correctly for STATIC
        obstacles: an obstacle beyond the end of a non-closed path is not that
        path's problem, the car drives to the end and the state machine decides
        again with the obstacle then inside the horizon. That branch is tagged
        `static/beyond_path (ignored)` and the comment above it records what
        happened before it existed - path_free was permanently false and no
        static avoidance could ever arm.

        The DYNAMIC branch of the same function still sets is_free False in
        exactly that situation, tagged `dyn/nopred/beyond_path`. It stayed
        invisible because a running predictor sends the opponent down the
        `dyn/pred` branch instead. Launched with prediction:=false there are no
        predictions at all, so every dynamic obstacle takes `dyn/nopred`, and an
        opponent farther ahead than the static avoidance path's end - while
        still inside interest_horizon_m - refuses that path on every tick. The
        car then trails the static obstacle it should have driven around, and
        per _check_static_overtaking_mode's own comment a stopped car in front
        of a stationary obstacle is a fixed point that does not recover.

        Read back off free_dbg, which the shared check fills in per obstacle,
        rather than reimplementing the geometry: this stays correct if the
        branch conditions there change. Corrected in the wrapper because
        state_machine_node.py is what time_trials.launch.xml runs.
        """
        debug = getattr(wpnts_data, "free_dbg", None)
        if not isinstance(debug, dict) or not debug.get("is_init"):
            return False
        blocked = [rec for rec in debug.get("obs", []) if rec.get("blocked")]
        if not blocked:
            return False
        return all(
            rec.get("branch") == "dyn/nopred/beyond_path" for rec in blocked)

    def _blocked_only_by_the_opponent(self, wpnts_data) -> bool:
        """Was the STATIC avoidance path refused only by the selected opponent?

        The sibling of _blocked_only_beyond_path, for the opponent that is
        inside the path rather than past its end.

        It used to excuse the opponent only when it was FURTHER along the path
        than the box the path is for, reasoning that it would have driven on by
        the time the car got there. That left the trailing case untouched, and
        the trailing case is where the whole thing falls over: an opponent
        being trailed is by definition NEARER than the box. Measured on
        2026-08-19 over thirteen laps - of 60 aborted evasions, 37 were
        path_free refusals and 28 of those named the selected opponent, every
        one of them nearer than the box. The distance form of this excuse fired
        4 times.

        So it now excuses the opponent wherever it is, on a separation of
        concerns the stack already makes everywhere else:

          LATERAL, where to drive, is the path's job. A box does not move, so
          the only way past it is around it.

          LONGITUDINAL, how fast, is the trailing controller's job. The
          opponent does move, and the gap to it is held by a PID on
          trailing_gap - not by the shape of the path.

        A collision needs both cars in the same place at the same TIME. This
        check only measures place. Refusing a path because the opponent's
        CURRENT position lies on it throws away a manoeuvre whose longitudinal
        safety is being enforced by something else - and the fallback is the
        raceline, which is the line the box is sitting on.

        WHAT IT ASSUMES, and this is not free: that something holds the gap to
        the opponent while the car is on this path. Controller.py runs the
        trailing PID only in state TRAILING, so once OVERTAKE is entered the
        car takes the path at its planned speed with no gap control. Until the
        controller keeps trailing whenever an opponent is ahead, the guarantee
        this leans on stops at the moment the evasion starts.

        Still refused, exactly as before:
          - anything static, at any distance. A box cannot be trailed past.
          - a mixture. If a box blocks the path as well, the refusal stands.

        Scoped by the caller to the static cache: the lane-change path is
        planned around the opponent, so an opponent refusing THAT one is the
        check working, not a false refusal.
        """
        debug = getattr(wpnts_data, "free_dbg", None)
        if not isinstance(debug, dict) or not debug.get("is_init"):
            return False
        records = debug.get("obs", [])
        blocked = [rec for rec in records if rec.get("blocked")]
        if not blocked:
            return False

        # By OPPONENT, not by is_static, for the same reason
        # _closing_on_nearest_avoidable no longer asks that question: with
        # nothing but boxes on the track, 39 of 41 tracks read DYNAMIC for
        # their whole life. The distinction that matters is "the obstacle this
        # path is for" against "the opponent, which the trailing gap covers",
        # and only the second of those is knowable.
        opponent_ids = self._opponent_record_ids(records)
        if not opponent_ids:
            return False
        return all(rec.get("id") in opponent_ids for rec in blocked)

    def _worst_free(self, wpnts_data):
        """Tightest clearance, with the opponent left out of the static path.

        The same principle as _blocked_only_by_the_opponent, one step later.
        _worth_driving compares this path's worst clearance against the
        raceline's, and the shared version takes the minimum over every
        obstacle - so an opponent sitting on the path drags the number under
        static_overtake_min_clearance_m and the floor refuses, even though the
        opponent is the one obstacle a longitudinal gap can handle.

        Without this the veto only moves: is_free is excused above, then the
        same opponent refuses through the floor whenever a box is blocking too.

        The raceline's own number is deliberately NOT filtered. An opponent on
        the raceline is exactly what makes the car trail, and hiding it would
        make the fallback look better than it is - which is the comparison this
        feeds.
        """
        if wpnts_data is not self.cur_static_avoidance_wpnts:
            return super()._worst_free(wpnts_data)
        debug = getattr(wpnts_data, "free_dbg", None)
        if not isinstance(debug, dict):
            return super()._worst_free(wpnts_data)
        records = debug.get("obs", ())
        opponent_ids = self._opponent_record_ids(records)
        room = [rec["free_dist"] for rec in records
                if rec.get("free_dist") is not None
                and rec.get("id") not in opponent_ids]
        return min(room) if room else None

    def _opponent_record_ids(self, records):
        """Ids in a free_dbg record set that are the selected opponent.

        The records carry the raw tracker id and the gap they were measured at,
        so the opponent is found the same way as everywhere else in this class
        - by position - and then reported by the id the record uses.
        """
        positions = self._opponent_positions()
        if not positions or not self.max_s:
            return set()
        ids = set()
        for rec in records:
            gap = rec.get("gap")
            if gap is None or rec.get("id") is None:
                continue
            for s_center, _ in positions:
                opponent_gap = (s_center - self.cur_s) % self.max_s
                # The record rounds the gap to two decimals; nothing else in
                # the list will be within a centimetre of the opponent.
                if abs(opponent_gap - gap) < 0.02:
                    ids.add(rec["id"])
                    break
        return ids

    def _check_free_frenet(self, wpnts_data):
        """Use physical body width and debounce only the global-path result."""
        configured_width = self.gb_ego_width_m
        clearance_width = getattr(
            self, "clearance_vehicle_width_m", configured_width)
        self.gb_ego_width_m = clearance_width
        try:
            is_free = self._memo_free_frenet(wpnts_data)
        finally:
            # Keep the legacy value for close-to-raceline state semantics.
            self.gb_ego_width_m = configured_width

        # Never fires for the raceline (global_tracking is is_closed: true), so
        # the trailing target this same function picks off cur_gb_wpnts is not
        # affected - only the avoidance and recovery caches are.
        if not is_free and self._blocked_only_beyond_path(wpnts_data):
            self.get_logger().info(
                f"[{wpnts_data.name}] only blocked by dynamic obstacles past "
                f"the end of this path - not this path's problem",
                throttle_duration_sec=5.0)
            is_free = True
            wpnts_data.closest_target = None
            wpnts_data.closest_gap = None
            wpnts_data.free_dbg["is_free"] = True

        # The same reasoning one step earlier: an opponent INSIDE the static
        # avoidance path. The path says where to drive; the trailing gap says
        # how close to get. Applied here rather than in
        # _check_static_overtaking_mode so that OVERTAKE's per-tick
        # sustainability check sees it too - correcting only the entry gate
        # would arm the manoeuvre and abort it on the next tick, which is
        # exactly the 0.45 s median lifetime measured on 2026-08-19.
        # Scoped by identity to the static cache: the lane-change path is
        # planned around the opponent, so an opponent refusing it is the check
        # working, not a false refusal.
        if (
            not is_free
            and wpnts_data is self.cur_static_avoidance_wpnts
            and self._blocked_only_by_the_opponent(wpnts_data)
        ):
            self.get_logger().info(
                f"[{wpnts_data.name}] only blocked by the opponent, whose gap "
                f"the trailing controller holds - driving it anyway",
                throttle_duration_sec=5.0)
            is_free = True
            wpnts_data.closest_target = None
            wpnts_data.closest_gap = None
            wpnts_data.free_dbg["is_free"] = True

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

    def _static_path_lead_in(self, wpnt_data):
        """Metres from the car to the start of a path that begins ahead of it.

        None when that is not the situation - no path, not the static cache,
        or a path that already starts at or behind the car.
        """
        if wpnt_data is not self.cur_static_avoidance_wpnts:
            return None
        if not wpnt_data.is_init or not len(wpnt_data.list):
            return None
        if self.max_s is None or self.max_s <= 0.0:
            return None
        lead_in = (float(wpnt_data.list[0].s_m) - self.cur_s) % self.max_s
        # A path that starts behind us is the ordinary case the shared test
        # already handles; only a genuine gap in front is this one.
        if lead_in <= 0.0 or lead_in > self.static_path_lead_in_m:
            return None
        return lead_in

    def _check_on_spline(self, wpnt_data) -> bool:
        """Also accept a static avoidance path that starts a few metres ahead.

        The shared test asks whether the car is within
        on_spline_min_dist_thres_m of the path. For the static avoidance path
        that is not "is this path mine", it is "have I arrived at it yet", and
        the answer is no for several metres longer than it should be.

        spline samples from max(control_s[0], car_s), so while the car is
        behind the first approach knot the published path starts 4*scale
        metres before the obstacle. At scale 1.5 the car is therefore not
        within 1.5 m of it until roughly 7.5 m of gap - but trailing has been
        commanding a brake since 11.8 m (P 0.5, D 0.25, vel_gain 0.10,
        trailing_gap 2.0, 3 m/s behind a stationary box). Four metres of
        braking for an obstacle the planner has already drawn a way around,
        which _check_latest_wpnts refuses only because the car has not reached
        the drawing yet.

        Nothing is filled in here - this decides acceptance only. What the car
        drives over the gap is get_splini_wpts' problem, and the two have to
        change together: accepting the path without the raceline lead-in puts
        local_wpnts several metres in front of the car, and the controller's
        AEB_for_weird_local_wpnt clamps to 2.0 m/s whenever the nearest local
        waypoint is further than AEB_thres (0.5 m). That trades one
        deceleration for another.

        Only widened while the car is on the raceline. The lead-in is built
        out of raceline waypoints, so it is a truthful description of what the
        car will do only if the car is on the raceline to begin with; off it,
        the shared answer stands and RECOVERY keeps its job.
        """
        if super()._check_on_spline(wpnt_data):
            return True
        if self._static_path_lead_in(wpnt_data) is None:
            return False
        if not self._check_close_to_raceline():
            return False
        # The path still has to extend past us, same as the shared test wants.
        gap = (float(wpnt_data.list[-1].s_m) - self.cur_s) % self.max_s
        return bool(gap > wpnt_data.on_spline_front_horizon_thres_m)

    def get_splini_wpts(self):
        """Fill the gap between the car and a path that starts ahead of it.

        The shared version picks the nearest point of the avoidance path and
        slices forward from there, so when the path begins three metres ahead
        the published local path does too, and the controller sees its nearest
        waypoint three metres away. AEB_for_weird_local_wpnt then clamps the
        command to 2.0 m/s - the deceleration this whole change exists to
        remove, arriving through a different door.

        So prepend the raceline the car is already on, from where the car is
        to where the avoidance path starts. This is the same operation the
        shared code already performs at the other end: when the avoidance path
        runs out before n_loc_wpnts it extends the tail with cur_gb_wpnts. Only
        the head was missing.

        The avoidance path itself is untouched - not extended, not resampled,
        not replanned. spline_node.py is what time_trials runs and it is not
        involved at all.
        """
        wpnts = super().get_splini_wpts()
        if not wpnts:
            return wpnts
        lead_in = self._static_path_lead_in(self.cur_static_avoidance_wpnts)
        if lead_in is None:
            return wpnts
        # Only when the slice really did start at the path's own first point;
        # if the car is already inside the path there is no gap to fill.
        if float(wpnts[0].s_m) != float(self.cur_static_avoidance_wpnts.list[0].s_m):
            return wpnts
        if not self.num_glb_wpnts or self.waypoints_dist <= 0.0:
            return wpnts

        start = int(self.cur_s / self.waypoints_dist + 0.5)
        count = int(lead_in / self.waypoints_dist)
        if count <= 0:
            return wpnts
        prefix = [self.cur_gb_wpnts.list[(start + i) % self.num_glb_wpnts]
                  for i in range(count)]
        return (prefix + wpnts)[:self.n_loc_wpnts]

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
    state_machine = H2HStateMachine()
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
