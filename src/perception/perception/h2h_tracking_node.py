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
   /tracking/dynamic_obstacles, on the tracker's own ``is_static``.

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
"""

import rclpy
from f110_msgs.msg import ObstacleArray
from rclpy.executors import ExternalShutdownException

from perception.tracking_node import TrackingNode


class HeadToHeadTrackingNode(TrackingNode):
    """Publish complete obstacles, size moving targets like cars, and split."""

    # Fields the shared tracker drops. Kept as a separate tuple so the parent's
    # list stays the single definition of everything else.
    EXTRA_FIELDS = (
        'x_m', 'y_m', 'theta', 's_var', 'd_var', 'vs_var', 'vd_var',
    )

    def __init__(self):
        super().__init__()
        for name, default in (
            ('dynamic_extent_min_m', 0.202),
            ('dynamic_extent_max_m', 0.42),
            ('dynamic_extent_decay_mps', 0.5),
        ):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
        self.get_logger().info(
            'head-to-head tracker: full obstacle fields, '
            f'moving-target extents capped at '
            f'{float(self.get_parameter("dynamic_extent_max_m").value):.2f} m')

        # The split. /tracking/obstacles keeps carrying everything, exactly as
        # the parent publishes it, so nothing that reads the shared topic sees a
        # difference; these two are additional views of that same list.
        self.static_pub = self.create_publisher(
            ObstacleArray, '/tracking/static_obstacles', 5)
        self.dynamic_pub = self.create_publisher(
            ObstacleArray, '/tracking/dynamic_obstacles', 5)

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
        """
        minimum = int(self.get_parameter('static_min_samples').value)
        return (
            len(track.s_history) >= minimum
            and not bool(track.obstacle.is_static)
        )

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

    def _publish_split(self, header):
        """Route each track by the tracker's own is_static.

        Both arrays are rebuilt every frame rather than latched. A track that
        stops being static leaves the static array on the same tick it enters
        the dynamic one, so no consumer can ever see it in both.

        ``publish_static`` is honoured for the same reason the parent honours
        it: with it false the static half is deliberately invisible, and a
        split that ignored it would put back exactly what was switched off.
        """
        stamp = header.stamp.sec + header.stamp.nanosec / 1e9
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds / 1e9
        publish_static = bool(self.get_parameter('publish_static').value)

        static_msg = ObstacleArray(header=header)
        dynamic_msg = ObstacleArray(header=header)
        for track in self.tracks:
            if bool(track.obstacle.is_static):
                if publish_static:
                    static_msg.obstacles.append(
                        self._obstacle_for_output(track, stamp))
            else:
                dynamic_msg.obstacles.append(
                    self._obstacle_for_output(track, stamp))
        self.static_pub.publish(static_msg)
        self.dynamic_pub.publish(dynamic_msg)

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
