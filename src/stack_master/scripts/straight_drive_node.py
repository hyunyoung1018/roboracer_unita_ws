#!/usr/bin/env python3
"""Constant forward command - the smallest possible autonomous source.

Publishes one AckermannDriveStamped per tick on the topic simple_mux reads as
its autonomous input. Straight ahead, constant speed, no map, no localisation,
no planner: this is what RB drives in straight_test.launch.xml.

It publishes all the time and never looks at the joystick. The mux owns the
mode - nothing here reaches the VESC until RB puts the mux in autodrive - so
there is exactly one place in the stack that decides who is driving.

RATE IS NOT COSMETIC. simple_mux drops any command older than its
joy_freshness_threshold (0.2 s) and publishes a stop instead, so this node has
to keep stamping fresh messages faster than that or the car stutters. 50 Hz
matches the mux's own loop.

Speed and trim are read every tick, so they can be turned while the car is
sitting there:

    ros2 param set /straight_drive speed 1.5
    ros2 param set /straight_drive steering_angle 0.01
"""

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node


class StraightDriveNode(Node):

    def __init__(self):
        super().__init__('straight_drive')

        self.declare_parameter(
            'out_topic',
            '/vesc/high_level/ackermann_cmd'
        )
        self.declare_parameter('speed', 1.0)

        # Steering TRIM, not steering. Zero is straight only if the servo
        # calibration in vesc.yaml is right; if the car pulls, correct it here
        # rather than by moving the calibration, which the whole stack shares.
        self.declare_parameter('steering_angle', 0.0)

        # Ceiling on the live `speed` parameter above. A test that is meant to
        # run at walking pace should not be one typo away from full throttle.
        self.declare_parameter('max_speed', 2.0)

        self.declare_parameter('rate_hz', 50.0)

        p = lambda name: self.get_parameter(name).value

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            p('out_topic'),
            10
        )

        # Last command actually published, so a live parameter change is
        # logged once instead of at the loop rate.
        self.last_logged = None

        self.create_timer(
            1.0 / p('rate_hz'),
            self._loop
        )

        self.get_logger().info(
            f"straight_drive -> {p('out_topic')} at {p('rate_hz')} Hz, "
            f"speed {p('speed')} m/s (RB on the joystick lets it through)"
        )

    def _loop(self):
        speed = float(self.get_parameter('speed').value)
        steer = float(self.get_parameter('steering_angle').value)
        limit = abs(float(self.get_parameter('max_speed').value))

        clipped = max(-limit, min(limit, speed))

        if clipped != speed:
            self.get_logger().warn(
                f'speed {speed} clipped to max_speed {limit}',
                throttle_duration_sec=2.0
            )

        if (clipped, steer) != self.last_logged:
            self.get_logger().info(
                f'command: {clipped} m/s, steering {steer} rad'
            )
            self.last_logged = (clipped, steer)

        msg = AckermannDriveStamped()
        # Stamped every tick. The mux compares this against its own clock and
        # a stale command is treated as a dead controller.
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = clipped
        msg.drive.steering_angle = steer
        msg.drive.acceleration = 0.0
        msg.drive.jerk = 0.0

        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StraightDriveNode()

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
