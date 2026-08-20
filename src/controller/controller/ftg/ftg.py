#!/usr/bin/env python3
"""Follow-The-Gap (disparity-extender). Beam/FOV independent: all thresholds in
metres/radians, so one tuning works on the 2160-beam sim and the ~1080-beam car.
Per scan: sanitize NaN->max, window the front +-FRONT_FOV, smooth, mask returns
within track_width/2, bubble each >=DISP_THRESH disparity, take the largest free
gap, aim at its centre, EMA-smooth the steer."""
import math
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray


class FTG:
    # Speed schedule, as fractions of the steering FTG can actually command.
    #
    # These were absolute angles - 30 deg for a corner, 10 for a bend, 3 for
    # dead straight - compared against a steer clipped to MAX_STEER. On this
    # car MAX_STEER is 0.4 rad, 22.9 deg, so the corner branch was under the
    # clip and unreachable: at full lock FTG still commanded MILD_CORNERS_
    # SPEED, 45% of ftg_max_speed, and CORNERS_SPEED was dead code.
    #
    # Fractions instead, so every step is reachable whatever MAX_STEER is set
    # to. The shape is the old one - four bands, slowest hard over - but the
    # top band is given real width rather than sitting exactly at full lock,
    # where the clip means it is either never reached or reached only there.
    CORNER_LOCK_FRAC = 0.70                 # ftg_corner_lock_frac
    MILD_LOCK_FRAC = 0.35                   # ftg_mild_lock_frac
    STRAIGHT_LOCK_FRAC = 0.10               # ftg_straight_lock_frac

    # disparity-extender tunables: DEFAULTS, overridden from controller.yaml (ftg_*)
    # and live-tunable via rqt.
    FRONT_FOV = math.radians(45.0)    # HALF-angle [rad] = ftg_front_fov_deg/2 (total 90 deg)
    SMOOTH_RAD = math.radians(1.0)    # scan smoothing window [rad]     (ftg_smooth_deg)
    DISP_THRESH = 0.5                 # range jump = discontinuity [m]  (ftg_disp_thresh)
    BUBBLE_M = 0.30                   # safety bubble per disparity [m] (ftg_bubble_m)
    STEER_EMA = 0.0                   # steer low-pass a: s=a*prev+(1-a)*new (ftg_steer_ema)
    MAX_STEER = 0.4                   # steering clip [rad]             (ftg_max_steer)
    SPEED_SCALE = 1.0                 # overall speed multiplier

    # Map-aware heading choice. OFF by default: with USE_MAP false this class
    # behaves exactly as it always has, so turning it on is an A/B on the car
    # rather than a change to the fallback everything else relies on.
    #
    # The gap search cannot tell a wall from a cone - both are just returns -
    # so it has no way to prefer the one it must not touch. The map can: what
    # is in it is track, what is not is something that arrived later. That is
    # the whole reason this exists, and it sets the priority. Staying on the
    # map is HARD. Missing an obstacle is soft; brushing one costs nothing
    # here, and stopping costs the run.
    USE_MAP = False                   # ftg_use_map
    HEADING_STEP_DEG = 2.0            # candidate spacing, ftg_heading_step_deg
    MAP_PROBE_M = 1.5                 # how far a candidate is followed, ftg_map_probe_m
    MAP_PROBE_STEP_M = 0.15           # spacing along it, ftg_map_probe_step_m
    # [m] A candidate whose own beam is shorter than this is refused whatever
    # the map says. The map is a belief about where the car is; the scan is
    # what is actually in front of it. When they disagree the scan wins, so a
    # localisation error steers into a wall the lidar can see rather than
    # through it.
    MIN_LIDAR_CLEARANCE_M = 0.35      # ftg_min_lidar_clearance_m
    # [rad^-1] How much a candidate is penalised for pointing away from
    # straight ahead, against a score in metres. 0 takes the roomiest heading
    # wherever it points, which on a track this wide is how the car ends up
    # facing the infield.
    FORWARD_BIAS = 0.35               # ftg_forward_bias
    # [m] base_link -> laser, from albomb_sensors.xacro's laser_joint. The
    # scan's bearings are measured here, not at the car's origin, and 0.26 m
    # is a quarter of the narrowest part of this track.
    LASER_OFFSET_X = 0.259            # ftg_laser_offset_x

    # [m] What has to fit through a gap before it is worth steering at.
    #
    # The gap search compares runs of free bearings by their length in BEAMS,
    # and a gap's room is not an angle - the same angle is a different number
    # of metres at every range. Measured across a 20 deg run:
    #
    #   0.5 m -> 0.17 m     1.0 m -> 0.35 m     2.0 m -> 0.69 m
    #
    # so the widest-angle gap is biased towards the NEAREST one, which is
    # exactly the gap least likely to fit. The car is 0.28 m across; 0.34
    # leaves 3 cm a side, which is what an escape is worth.
    MIN_GAP_M = 0.34                  # ftg_min_gap_m
    # Fraction of the front FOV one disparity's bubble may consume.
    #
    # The bubble is a fixed 0.3 m in METRES, and its angle grows as the thing
    # gets closer: 16.7 deg at 1 m, 31.0 at 0.5, 36.9 at 0.4. Against an 80 deg
    # front FOV an obstacle at 0.4 m spends 92% of the view on its two edges,
    # and what it spends it on is the gap beside it - the bubble is blown into
    # the opening, which is the one place the car was going to go. Close
    # quarters is where FTG is asked to work at all, so the bubble is capped
    # rather than left to eat the answer.
    BUBBLE_MAX_FRAC = 0.25            # ftg_bubble_max_frac

    def __init__(self, node=None, mapping=False, debug=False,
                 safety_radius=None, max_lidar_dist=None, max_speed=1.5,
                 range_offset=None, track_width=2.0,
                 front_fov_deg=None, smooth_deg=None, disp_thresh=None,
                 bubble_m=None, steer_ema=None, max_steer=None,
                 use_map=None, heading_step_deg=None, map_probe_m=None,
                 map_probe_step_m=None, min_lidar_clearance_m=None,
                 forward_bias=None, laser_offset_x=None,
                 min_gap_m=None, bubble_max_frac=None) -> None:
        self.node = node
        # the racecar's laser link is namespaced rather than a bare 'laser'. The
        # markers below are drawn in it, so a wrong name makes them invisible
        # rather than misplaced - the same failure the particle filter had.
        self.laser_frame = 'ego_racecar/laser'
        self.mapping = mapping

        self.DEBUG = debug
        self.SAFETY_RADIUS = safety_radius          # accepted for compat (unused)
        self.range_offset = range_offset            # accepted for compat (unused)
        self.MAX_LIDAR_DIST = max_lidar_dist if max_lidar_dist else 10.0
        self.MAX_SPEED = max_speed
        self.track_width = track_width

        # override class-default tunables from yaml when provided
        if front_fov_deg is not None:
            self.FRONT_FOV = math.radians(float(front_fov_deg) / 2.0)  # param = TOTAL front FOV
        if smooth_deg is not None:
            self.SMOOTH_RAD = math.radians(float(smooth_deg))
        if disp_thresh is not None:
            self.DISP_THRESH = float(disp_thresh)
        if bubble_m is not None:
            self.BUBBLE_M = float(bubble_m)
        if steer_ema is not None:
            self.STEER_EMA = float(steer_ema)
        if max_steer is not None:
            self.MAX_STEER = float(max_steer)
        if use_map is not None:
            self.USE_MAP = bool(use_map)
        if heading_step_deg is not None:
            self.HEADING_STEP_DEG = float(heading_step_deg)
        if map_probe_m is not None:
            self.MAP_PROBE_M = float(map_probe_m)
        if map_probe_step_m is not None:
            self.MAP_PROBE_STEP_M = max(0.02, float(map_probe_step_m))
        if min_lidar_clearance_m is not None:
            self.MIN_LIDAR_CLEARANCE_M = float(min_lidar_clearance_m)
        if forward_bias is not None:
            self.FORWARD_BIAS = float(forward_bias)
        if laser_offset_x is not None:
            self.LASER_OFFSET_X = float(laser_offset_x)
        if min_gap_m is not None:
            self.MIN_GAP_M = float(min_gap_m)
        if bubble_max_frac is not None:
            self.BUBBLE_MAX_FRAC = float(bubble_max_frac)

        self.recompute_speeds()

        self.velocity = 0.0
        self._steer_prev = None
        self.angle_min = -0.75 * np.pi
        self.angle_inc = None
        self.radians_per_elem = None

        self.best_pnt = self.scan_pub = self.best_gap = None
        if self.node is not None:
            self.best_pnt = self.node.create_publisher(Marker, '/best_points/marker', 10)
            self.scan_pub = self.node.create_publisher(MarkerArray, '/scan_proc/markers', 10)
            self.best_gap = self.node.create_publisher(MarkerArray, '/best_gap/markers', 10)

    def recompute_speeds(self) -> None:
        s = self.SPEED_SCALE
        self.CORNERS_SPEED = 0.3 * self.MAX_SPEED * s
        self.MILD_CORNERS_SPEED = 0.45 * self.MAX_SPEED * s
        self.STRAIGHTS_SPEED = 0.8 * self.MAX_SPEED * s
        self.ULTRASTRAIGHTS_SPEED = self.MAX_SPEED * s

    def set_vel(self, velocity) -> None:
        self.velocity = velocity

    def _now(self):
        return self.node.get_clock().now().to_msg() if self.node is not None else None

    def _bubble_beams(self, near_range, fov_beams=None) -> int:
        """Beams to blank beside a disparity, capped at a share of the view.

        BUBBLE_M is metres, so its angle grows without limit as the thing gets
        closer, and it is blown INTO the opening - at 0.4 m two edges spend
        92% of an 80 deg FOV and there is nothing left to steer at. The cap is
        what stops a near obstacle deleting the answer; see BUBBLE_MAX_FRAC.
        """
        near = max(float(near_range), 0.05)
        beams = int(math.ceil(math.atan2(self.BUBBLE_M, near) / self.angle_inc))
        if fov_beams:
            beams = min(beams, max(1, int(fov_beams * self.BUBBLE_MAX_FRAC)))
        return beams

    def _runs(self, mask):
        """Every contiguous True run in `mask`, as (start, end) inclusive."""
        if not mask.any():
            return []
        edges = np.where(
            np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))[0]
        return [(int(a), int(b) - 1)
                for a, b in zip(edges[0::2], edges[1::2])]

    def _gap_width_m(self, proc, gl, gr):
        """The metres a gap actually offers, at its tightest point.

        chord = 2 r sin(theta/2), with r the SHORTEST range in the run rather
        than the mean: a gap is only as wide as its narrowest part, and taking
        the average would let a deep opening excuse a pinch at its mouth.
        """
        theta = (gr - gl + 1) * self.angle_inc
        r = float(np.min(proc[gl:gr + 1]))
        return 2.0 * r * math.sin(min(theta, math.pi) / 2.0)

    def _choose_gap(self, free, proc):
        """Index to steer at: the roomiest gap the car fits through.

        Falls back in two steps, and neither of them is `argmax(proc)`. That
        was the old fallback and it reads the RAW scan - no mask, no bubbles -
        so the one branch taken when the search has failed was also the one
        with no safety in it, aiming at whatever happened to be farthest.

        Instead: the widest gap that fits, else the widest gap there is, else
        - only when the mask left nothing at all - the farthest point. A gap
        too narrow to fit is still a better guess than a bearing chosen with
        the obstacle bubbles thrown away.
        """
        runs = self._runs(free)
        if not runs:
            return int(np.argmax(proc)), 'no free bearings'
        widths = [self._gap_width_m(proc, gl, gr) for gl, gr in runs]
        fits = [i for i, w in enumerate(widths) if w >= self.MIN_GAP_M]
        if fits:
            k = max(fits, key=lambda i: widths[i])
            reason = 'fits'
        else:
            k = int(np.argmax(widths))
            reason = 'nothing fits'
        gl, gr = runs[k]
        return (gl + gr) // 2, reason

    def _choose_heading_on_map(self, proc, base_angle, pose, grid):
        """Index into `proc` of the best heading that stays on the map.

        Every candidate is scored, not just the middles of gaps. A gap is a
        run of free bearings, and its width in BEAMS is what _largest_run
        compares - but the room a gap actually offers is metres, and the same
        angle is a different number of metres at every range. Scoring the
        bearings directly drops the question.

        Three things decide a candidate, in this order:

          the scan     a beam shorter than MIN_LIDAR_CLEARANCE_M is refused
                       outright. This is the veto that survives a bad pose:
                       the map is where the car BELIEVES the track is, and
                       when the two disagree the thing the lidar can see wins.
          the map      the heading is followed out to MAP_PROBE_M and every
                       sample has to be known free space. This is the hard
                       constraint - it is the only one the car cannot buy its
                       way out of by driving carefully.
          the score    among what survives, more room and straighter ahead.

        Returns None only when the map itself is unusable, which leaves the
        caller on the old gap search. When the map IS usable but nothing
        clears it, the heading that stays on it LONGEST comes back instead -
        never nothing. Stopping is the one outcome this must not produce.
        """
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        # Bearings are measured at the laser, so the rays start there.
        origin_x = x + self.LASER_OFFSET_X * math.cos(yaw)
        origin_y = y + self.LASER_OFFSET_X * math.sin(yaw)

        step = max(1, int(round(math.radians(self.HEADING_STEP_DEG) / self.angle_inc)))
        probe = np.arange(self.MAP_PROBE_STEP_M,
                          self.MAP_PROBE_M + 1e-9, self.MAP_PROBE_STEP_M)
        if probe.size == 0:
            return None

        best = best_score = None
        longest = None
        longest_reach = -1.0
        for i in range(0, len(proc), step):
            beam = float(proc[i])
            if beam < self.MIN_LIDAR_CLEARANCE_M:
                continue
            bearing = base_angle + i * self.angle_inc
            heading = yaw + bearing
            # Never probe past what the scan can vouch for: beyond the beam's
            # own return the map would be answering about the far side of
            # whatever is standing there.
            reach = probe[probe <= beam]
            if reach.size == 0:
                continue
            px = origin_x + reach * math.cos(heading)
            py = origin_y + reach * math.sin(heading)
            outside = grid.first_outside_index(np.column_stack((px, py)))
            if outside is not None:
                on_map = 0.0 if outside == 0 else float(reach[outside - 1])
                if on_map > longest_reach:
                    longest_reach, longest = on_map, i
                continue
            score = min(beam, self.MAP_PROBE_M) - self.FORWARD_BIAS * abs(bearing)
            if best_score is None or score > best_score:
                best_score, best = score, i

        if best is not None:
            return best
        return longest

    def process_lidar(self, ranges, angle_min=None, angle_increment=None,
                      pose=None, grid=None) -> tuple:
        """Returns (speed, steering_angle). angle_min/angle_increment from the
        LaserScan make it beam/FOV independent; if omitted a 270-deg FOV is assumed.

        `pose` is (x, y, yaw) of base_link in the map frame and `grid` a
        GridFilter. Both are optional and only consulted when USE_MAP is set;
        without them this is the gap search it has always been."""
        n = len(ranges)
        if angle_increment is not None and angle_increment > 0.0:
            self.angle_inc = float(angle_increment)
            self.angle_min = float(angle_min) if angle_min is not None else -(n - 1) * self.angle_inc / 2.0
        else:
            self.angle_inc = (1.5 * np.pi) / n
            self.angle_min = -(n - 1) * self.angle_inc / 2.0
        self.radians_per_elem = self.angle_inc

        # NaN/inf == no return == open -> max range, then clip
        r = np.asarray(ranges, dtype=float)
        r = np.where(np.isfinite(r), r, self.MAX_LIDAR_DIST)
        r = np.clip(r, 0.0, self.MAX_LIDAR_DIST)

        # front FOV window (angle-based)
        i_lo = max(0, int(math.ceil((-self.FRONT_FOV - self.angle_min) / self.angle_inc)))
        i_hi = min(n, int(math.floor((self.FRONT_FOV - self.angle_min) / self.angle_inc)) + 1)
        if i_hi - i_lo < 3:
            return self._speed_for(0.0), 0.0
        proc = r[i_lo:i_hi].copy()
        base_angle = self.angle_min + i_lo * self.angle_inc

        # smoothing window sized in radians
        w = max(1, int(round(self.SMOOTH_RAD / self.angle_inc)))
        if w > 1:
            proc = np.convolve(proc, np.ones(w) / w, 'same')

        # mask returns within half the track width (too close = wall/self)
        free = proc >= (self.track_width / 2.0)

        # disparity extender: bubble the near side of each >DISP_THRESH jump
        d = np.diff(proc)
        L = len(proc)
        for i in np.where(np.abs(d) > self.DISP_THRESH)[0]:
            if d[i] > 0:
                b = self._bubble_beams(proc[i], L)
                free[i + 1: min(L, i + 1 + b)] = False
            else:
                b = self._bubble_beams(proc[i + 1], L)
                free[max(0, i + 1 - b): i + 1] = False

        # The roomiest gap the car fits through, in metres rather than beams.
        gl, gr = self._largest_run(free)      # debug markers still draw this
        mid, _ = self._choose_gap(free, proc)

        # The map, when there is one, decides instead. Computed after the gap
        # search rather than instead of it so gl/gr still describe what the
        # scan alone would have picked, which is what the debug markers draw -
        # the two side by side are how this gets judged on the car.
        if self.USE_MAP and pose is not None and grid is not None \
                and getattr(grid, 'ready', False):
            on_map = self._choose_heading_on_map(proc, base_angle, pose, grid)
            if on_map is not None:
                mid = on_map

        # steer toward the gap centre (0 = forward, + = left), then EMA-smooth
        raw_steer = float(np.clip(base_angle + mid * self.angle_inc,
                                  -self.MAX_STEER, self.MAX_STEER))
        if self._steer_prev is None:
            steer = raw_steer
        else:
            a = self.STEER_EMA
            steer = a * self._steer_prev + (1.0 - a) * raw_steer
        self._steer_prev = steer

        if self.DEBUG:
            self._publish_debug(proc, base_angle, gl, gr, mid)
        return self._speed_for(steer), steer

    @staticmethod
    def _largest_run(mask) -> tuple:
        if not mask.any():
            return 0, 0
        edges = np.where(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))[0]
        starts, ends = edges[0::2], edges[1::2]
        k = int(np.argmax(ends - starts))
        return int(starts[k]), int(ends[k]) - 1

    def _speed_for(self, steering_angle) -> float:
        """Speed from |steer|, on thresholds scaled to what the steer can be.

        The thresholds were absolute - 30 deg for a corner, 10 for a bend -
        against a steer clipped to MAX_STEER, which is 0.4 rad / 22.9 deg on
        this car. So the corner branch could not be reached at all and the
        slowest FTG would ever go was MILD_CORNERS_SPEED, 45% of ftg_max_speed
        even at full lock. Reading them as FRACTIONS of full lock keeps the
        original intent - the shape of the schedule - and makes every step of
        it reachable whatever MAX_STEER is set to.
        """
        if self.mapping:
            return 1.5
        a = abs(steering_angle) / max(1e-6, self.MAX_STEER)
        if a >= self.CORNER_LOCK_FRAC:
            return self.CORNERS_SPEED
        if a >= self.MILD_LOCK_FRAC:
            return self.MILD_CORNERS_SPEED
        if a >= self.STRAIGHT_LOCK_FRAC:
            return self.STRAIGHTS_SPEED
        return self.ULTRASTRAIGHTS_SPEED

    def _publish_debug(self, proc, base_angle, gl, gr, mid) -> None:
        clr = MarkerArray()
        m = Marker(); m.header.frame_id = self.laser_frame; m.header.stamp = self._now()
        m.action = Marker.DELETEALL
        clr.markers.append(m)
        self.best_gap.publish(clr)

        gap_markers = MarkerArray()
        for i in range(gl, max(gl + 1, gr)):
            ang = base_angle + i * self.angle_inc
            rng = float(proc[i]) if i < len(proc) else 1.0
            mrk = Marker()
            mrk.header.frame_id = self.laser_frame; mrk.header.stamp = self._now()
            mrk.type = mrk.SPHERE
            mrk.scale.x = mrk.scale.y = mrk.scale.z = 0.05
            mrk.color.a = 1.0; mrk.color.r = 1.0; mrk.color.g = 1.0
            mrk.id = i - gl
            mrk.pose.position.x = math.cos(ang) * rng
            mrk.pose.position.y = math.sin(ang) * rng
            mrk.pose.orientation.w = 1.0
            gap_markers.markers.append(mrk)
        self.best_gap.publish(gap_markers)

        ang = base_angle + mid * self.angle_inc
        rng = float(proc[mid]) if mid < len(proc) else 1.0
        bm = Marker()
        bm.header.frame_id = self.laser_frame; bm.header.stamp = self._now()
        bm.type = bm.SPHERE
        bm.scale.x = bm.scale.y = bm.scale.z = 0.2
        bm.color.a = 1.0; bm.color.b = 1.0; bm.color.g = 1.0
        bm.id = 0
        bm.pose.position.x = math.cos(ang) * rng
        bm.pose.position.y = math.sin(ang) * rng
        bm.pose.orientation.w = 1.0
        self.best_pnt.publish(bm)

        sm = MarkerArray()
        step = max(1, len(proc) // 360)
        for i in range(0, len(proc), step):
            ang = base_angle + i * self.angle_inc
            mrk = Marker()
            mrk.header.frame_id = self.laser_frame; mrk.header.stamp = self._now()
            mrk.type = mrk.SPHERE
            mrk.scale.x = mrk.scale.y = mrk.scale.z = 0.05
            mrk.color.a = 1.0; mrk.color.r = 1.0; mrk.color.b = 1.0
            mrk.id = i
            mrk.pose.position.x = math.cos(ang) * float(proc[i])
            mrk.pose.position.y = math.sin(ang) * float(proc[i])
            mrk.pose.orientation.w = 1.0
            sm.markers.append(mrk)
        self.scan_pub.publish(sm)
