#!/usr/bin/env python3
"""Head-to-head tracker: the shared tracker with two moving-target fixes.

This exists as a subclass rather than as edits to :mod:`perception.tracking_node`
because time_trials.launch.xml runs that node for static obstacle avoidance and
must keep behaving exactly as it does today. Nothing here changes unless the
tracker has already classified a track as moving, and time_trials never starts
this executable at all.

Two fixes, both about the opponent car:

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
   with how much of it the lidar has seen. See dynamic_perception.yaml.
"""

import rclpy
from rclpy.executors import ExternalShutdownException

from perception.tracking_node import TrackingNode


class HeadToHeadTrackingNode(TrackingNode):
    """Publish complete obstacles and size moving targets like cars."""

    # Fields the shared tracker drops. Kept as a separate tuple so the parent's
    # list stays the single definition of everything else.
    EXTRA_FIELDS = (
        'x_m', 'y_m', 'theta', 's_var', 'd_var', 'vs_var', 'vd_var',
    )

    def __init__(self):
        super().__init__()
        for name, default in (
            ('dynamic_extent_min_m', 0.29),
            ('dynamic_extent_max_m', 0.34),
            ('dynamic_extent_decay_mps', 0.5),
        ):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
        self.get_logger().info(
            'head-to-head tracker: full obstacle fields, '
            f'moving-target extents capped at '
            f'{float(self.get_parameter("dynamic_extent_max_m").value):.2f} m')

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
        # The opponent is detected off a 0.12 m pillar bolted to a car of our
        # own size, so the measurement is honest about the pillar and useless
        # about the car. There is one other car on the track - believe the car.
        #
        # 0.29 is the circumscribed radius of 0.28 x 0.50, i.e. a disc, so no
        # yaw estimate is needed and no orientation under-reads: side on, a car
        # fills 0.28 of BOTH Frenet axes where head on it fills 0.14 and 0.25.
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
