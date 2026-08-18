#!/usr/bin/env python3
"""Head-to-head tracker: the shared tracker, plus the head-to-head split.

This exists as a subclass rather than as edits to :mod:`perception.tracking_node`
because time_trials.launch.xml runs that node for static obstacle avoidance and
must keep behaving exactly as it does today. Nothing here changes unless the
tracker has already classified a track as moving, and time_trials never starts
this executable at all.

Three things, all about the opponent car:

1. ``_copy_obstacle`` in the shared tracker rebuilds every published obstacle
   field by field and the list omits ``x_m``, ``y_m``, ``theta`` and the four
   variances. The C++ detector fills them; they arrive at the tracker and leave
   it as zero. That silently breaks ``_check_free_cartesian`` (it measures the
   distance from the path to the origin), puts every trailing/overtaking marker
   at the origin in RViz, and hands the GP a trajectory with no measurement
   uncertainty on it.

2. ``_hold_extents`` keeps the largest extent a track has ever shown. Correct
   for a box - see the docstring it inherits - and wrong for a car, whose
   Frenet extents grow with its yaw relative to the track tangent rather than
   with how much of it the lidar has seen. See h2h_perception.yaml.

3. It splits its own output into /tracking/static_obstacles and
   /tracking/dynamic_obstacles, on the tracker's own ``is_static``, and it
   selects the ONE opponent that goes on the dynamic stream - see
   :mod:`perception.h2h_opponent_selector`.

   That split used to be a separate node - stable_obstacle_router - which
   re-classified every track from the STANDARD DEVIATION of its Frenet
   position and threw the tracker's answer away. On the car that lost: a
   stationary box measured from a moving car walks several centimetres a frame
   as the L-shape fit slides with the viewing angle, which cleared the
   router's 0.04 m "clearly dynamic" bar. It then routed boxes to the dynamic
   stream and even acquired them as the opponent, so the spline planner was
   handed nothing to avoid while the controller trailed a box at the 0.8 m
   opponent gap - hence stopping in front of obstacles, or driving into them
   when the derived speed came out wrong. Ten spurious acquisitions in one run.

   Nothing here re-judges anything. ``is_static`` is the shared tracker's
   verdict, from |v| < static_speed_threshold over static_min_samples frames,
   which is exactly the test time_trials drives on. One classifier, one
   definition of static, in the node that already estimates the velocity.

   SELECTION is a separate question and it was never the part at fault, so all
   of it survived the router: the corridor and forward-window gates, the lock
   on one target, re-identification across tracker ID churn, the logical
   opponent ID and the speed hold. Only the second classifier was dropped.

The three-state view the tracker supports is STATIC / DYNAMIC / UNKNOWN, where
UNKNOWN is a track too new to have collected static_min_samples frames. That
distinction is load-bearing twice over: an UNKNOWN track is the only thing
re-identification will accept as a replacement ID for the opponent, and
driving_mode_monitor refuses to call anything DYNAMIC on is_static=False alone
because UNKNOWN shares that wire representation. It is published on
/tracking/classification_debug.
"""

import json
from copy import deepcopy

import rclpy
from f110_msgs.msg import ObstacleArray
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String

from perception.h2h_opponent_selector import (
    DYNAMIC,
    STATIC,
    UNKNOWN,
    OpponentSelector,
)
from perception.tracking_node import TrackingNode


class HeadToHeadTrackingNode(TrackingNode):
    """Publish complete obstacles, size moving targets like cars, and split."""

    # Fields the shared tracker drops. Kept as a separate tuple so the parent's
    # list stays the single definition of everything else.
    EXTRA_FIELDS = (
        'x_m', 'y_m', 'theta', 's_var', 'd_var', 'vs_var', 'vd_var',
    )

    # Selection knobs. Everything the router carried EXCEPT the second
    # classifier's thresholds (min_std, max_std, *_confirm_count,
    # min_unknown_samples), which no longer have anything to configure.
    SELECTION_DEFAULTS = {
        'single_dynamic_opponent': True,
        'logical_opponent_id': 1000000,
        'opponent_width_m': 0.24,
        'opponent_boundary_margin_m': 0.03,
        'opponent_forward_min_m': 0.2,
        'opponent_forward_max_m': 8.0,
        # [m] Rear allowance for the target ALREADY locked onto. Signed, so
        # negative is behind the car, and it covers the car's own length plus
        # the opponent's. Acquisition still needs opponent_forward_min_m; see
        # OpponentSelector._forward_ok for what happens without it.
        'opponent_active_rear_m': -1.5,
        'dynamic_reid_timeout_sec': 1.0,
        'dynamic_reid_max_distance_m': 1.2,
        'dynamic_reid_max_lateral_m': 0.25,
        'dynamic_reid_ambiguity_margin_m': 0.20,
        'dynamic_speed_hold_sec': 0.75,
        'dynamic_speed_valid_min_mps': 0.15,
        # [m/s] Upper sanity bound on a believable opponent speed, and it is
        # new. The router had only the lower one, so a single bad frame
        # reporting 8.55 m/s for a stationary box was accepted as valid and
        # held for dynamic_speed_hold_sec - and the trailing controller
        # commands opponent speed minus a correction, so the car ACCELERATED
        # towards it. Nothing on this track goes faster than the raceline.
        'dynamic_speed_valid_max_mps': 6.0,
        'classification_debug': True,
    }

    def __init__(self):
        super().__init__()
        for name, default in (
            ('dynamic_extent_min_m', 0.202),
            ('dynamic_extent_max_m', 0.42),
            ('dynamic_extent_decay_mps', 0.5),
        ):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
        for name, default in self.SELECTION_DEFAULTS.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
        self.get_logger().info(
            'head-to-head tracker: full obstacle fields, '
            f'moving-target extents capped at '
            f'{float(self.get_parameter("dynamic_extent_max_m").value):.2f} m')

        # The split. /tracking/obstacles keeps carrying everything, exactly as
        # the parent publishes it, so nothing that reads the shared topic sees a
        # difference; these are additional views of that same list.
        self.static_pub = self.create_publisher(
            ObstacleArray, '/tracking/static_obstacles', 5)
        self.dynamic_pub = self.create_publisher(
            ObstacleArray, '/tracking/dynamic_obstacles', 5)
        self.debug_pub = self.create_publisher(
            String, '/tracking/classification_debug', 10)

        self.selector = OpponentSelector(
            lambda name: self.get_parameter(name).value,
            self.get_logger().warn)
        self.waypoints = []
        self.ego_s = None
        # /global_waypoints_scaled is already subscribed by the parent; the
        # corridor gate needs the waypoints themselves, not just the length, so
        # _path_cb is extended rather than a second subscription opened on the
        # same topic.
        self.create_subscription(
            Odometry, '/car_state/odom_frenet', self._odom_frenet_cb, 10)

    def _copy_obstacle(self, source):
        """Carry the Cartesian pose and the variances through the tracker.

        Not a staticmethod, unlike the parent's. Every call site in the parent
        goes through ``self._copy_obstacle(...)``, so binding it here overrides
        all of them; calling the parent's version through ``super()`` still
        works because Python resolves the staticmethod off the class.
        """
        target = TrackingNode._copy_obstacle(source)
        for field in self.EXTRA_FIELDS:
            setattr(target, field, getattr(source, field))
        return target

    def _is_confirmed_moving(self, track):
        """True once the tracker itself has decided this track is not static.

        Deliberately conservative in both directions. A brand new track has
        ``is_static`` False simply because that is the message default, so the
        history length is required as well - the same evidence
        ``_update_track`` demands before it sets the flag. A box therefore never
        reaches the moving-target ceiling, not even for its first few frames.

        That is precisely the DYNAMIC of the three-state view, so it is defined
        once, in _track_class, and the extent ceiling and the opponent gates
        cannot drift apart.
        """
        return self._track_class(track) == DYNAMIC

    def _obstacles_cb(self, msg):
        """Track as the parent does, then publish the two split views.

        After ``super()`` returns, ``self.tracks`` is the list the parent has
        just published from, so the split is built from the same tracks in the
        same tick - there is no second estimate anywhere and no way for the two
        to disagree.
        """
        super()._obstacles_cb(msg)
        if not self.track_length:
            # The parent returned before touching self.tracks; so do we.
            return
        self._publish_split(msg.header)

    def _path_cb(self, msg):
        """Keep the waypoints too, for the opponent corridor gate."""
        super()._path_cb(msg)
        if msg.wpnts:
            self.waypoints = list(msg.wpnts)

    def _odom_frenet_cb(self, msg):
        self.ego_s = float(msg.pose.pose.position.x)

    def _track_class(self, track):
        """STATIC / DYNAMIC / UNKNOWN for one track.

        UNKNOWN is not a hedge, it is the honest answer for a track that has
        not yet collected static_min_samples frames: is_static defaults False
        on a fresh message, so "not static" alone does not mean moving. Both
        re-identification and driving_mode_monitor depend on that distinction.
        """
        if bool(track.obstacle.is_static):
            return STATIC
        minimum = int(self.get_parameter('static_min_samples').value)
        return DYNAMIC if len(track.s_history) >= minimum else UNKNOWN

    def _publish_split(self, header):
        """Route static tracks by class, and the dynamic stream by selection.

        The static side is every STATIC track, rebuilt every frame rather than
        latched, so a track that starts moving leaves it on the same tick it
        enters the other one.

        The dynamic side is NOT every moving track. It is the one opponent the
        selector has locked onto, republished under logical_opponent_id so that
        opp_prediction and the lane-change planner see a stable ID across the
        tracker losing and recreating its own. Everything downstream of this
        node is written for exactly one opponent.

        ``publish_static`` is honoured for the same reason the parent honours
        it: with it false the static half is deliberately invisible, and a
        split that ignored it would put back what was switched off.
        """
        stamp = header.stamp.sec + header.stamp.nanosec / 1e9
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds / 1e9
        now = self.get_clock().now().nanoseconds * 1e-9
        publish_static = bool(self.get_parameter('publish_static').value)

        obstacles, classes = [], {}
        for track in self.tracks:
            obstacle = self._obstacle_for_output(track, stamp)
            obstacles.append(obstacle)
            classes[int(obstacle.id)] = self._track_class(track)

        static_msg = ObstacleArray(header=header)
        if publish_static:
            static_msg.obstacles = [
                obstacle for obstacle in obstacles
                if classes[int(obstacle.id)] == STATIC
            ]

        selected = self.selector.select(
            obstacles, classes, now, self.waypoints, self.track_length,
            self.ego_s)
        dynamic_msg = ObstacleArray(header=header)
        if selected is not None:
            # A copy: stabilize_speed rewrites vs, and the same object is
            # already in the array the parent published on /tracking/obstacles.
            opponent = deepcopy(selected)
            self.selector.stabilize_speed(opponent, now)
            opponent.id = int(self.get_parameter('logical_opponent_id').value)
            dynamic_msg.obstacles.append(opponent)

        self.static_pub.publish(static_msg)
        self.dynamic_pub.publish(dynamic_msg)
        self._publish_classification_debug(
            header, obstacles, classes, selected)

    def _publish_classification_debug(self, header, obstacles, classes,
                                      selected):
        """The per-track view of every gate, for driving_mode_monitor.

        Skipped when nothing subscribes: it is a json.dumps of every track,
        every frame, for a topic that is empty unless somebody is watching.
        """
        if not bool(self.get_parameter('classification_debug').value):
            return
        if self.debug_pub.get_subscription_count() == 0:
            return
        selected_id = None if selected is None else int(selected.id)
        logical_id = int(self.get_parameter('logical_opponent_id').value)
        records = []
        for obstacle in obstacles:
            tracker_id = int(obstacle.id)
            corridor_ok, forward_ok = self.selector.gates.get(
                tracker_id, (False, False))
            records.append({
                'id': logical_id if tracker_id == selected_id else tracker_id,
                'tracker_id': tracker_id,
                'stable_class': classes[tracker_id],
                'is_static': bool(obstacle.is_static),
                'is_visible': bool(obstacle.is_visible),
                'vs': round(float(obstacle.vs), 4),
                's_center': round(float(obstacle.s_center), 3),
                'd_center': round(float(obstacle.d_center), 3),
                'inside_opponent_corridor': corridor_ok,
                'inside_forward_window': forward_ok,
                'selected_opponent': tracker_id == selected_id,
            })
        self.debug_pub.publish(String(data=json.dumps({
            'stamp': header.stamp.sec + header.stamp.nanosec / 1e9,
            'active_opponent_tracker_id': self.selector.active_id,
            'obstacles': records,
        })))

    def _hold_extents(self, track, measurement, dt):
        """Hold extents with a car's ceiling and leak rate once moving.

        The running maximum is kept: it is what stopped the avoidance apex
        moving centimetres per frame, and that reason applies to a moving target
        too. Only the two numbers bounding it change, so a bad frame or a
        mis-associated detection still bleeds off - just within a tick instead
        of within six seconds - and the believed width stays inside what an
        F1TENTH car can physically be.
        """
        if not self._is_confirmed_moving(track):
            return super()._hold_extents(track, measurement, dt)

        held = track.extents
        measured = self._measured_extents(measurement)
        ceiling = float(self.get_parameter('dynamic_extent_max_m').value) / 2.0
        # A floor as well as a ceiling, and it is the floor that matters.
        #
        # The opponent is detected off a 0.12 m pillar bolted to a car, so the
        # measurement is honest about the pillar and useless about the car.
        # There is one other car on the track - believe the car.
        #
        # This is now the ONLY place head to head knows the opponent is not a
        # box. The detector is time_trials' unchanged, so everything is found
        # as a box and only a track the tracker has confirmed moving is resized
        # here. A static obstacle takes the branch above and is untouched.
        #
        # 0.202 is the circumscribed radius of a 0.24 x 0.325 car, i.e. a disc,
        # so no yaw estimate is needed and no orientation under-reads: the disc
        # is the worst case over every yaw the car can present to the track
        # tangent, which is the right way round when trailing is the first goal.
        floor = float(self.get_parameter('dynamic_extent_min_m').value)
        leak = float(
            self.get_parameter('dynamic_extent_decay_mps').value) * max(0.0, dt)
        for i in range(4):
            held[i] = min(ceiling, max(floor, measured[i], held[i] - leak))
        return held


def main(args=None):
    rclpy.init(args=args)
    node = HeadToHeadTrackingNode()
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
