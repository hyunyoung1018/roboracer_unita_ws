#!/usr/bin/env python3
"""Drive straight, with no map and no localisation.

The smallest thing that can occupy the autonomous side of simple_mux: a
constant steering angle and a constant speed, published at a fixed rate to
/vesc/high_level/ackermann_cmd. There is no waypoint, no pose, no obstacle -
which is the point. It exists to check the parts underneath the racing stack
on a piece of floor with nothing mapped:

    - the joystick contract (B stop, LB human, RB autonomous)
    - speed_to_erpm_gain, by driving a measured distance and comparing
      /vesc/odom's travel against a tape measure
    - steering_angle_to_servo_offset, by seeing whether "straight" is straight

The mux is what makes this safe to run at all. This node publishes
unconditionally from the moment it starts; nothing reaches the VESC until RB
is held, and releasing it commands zero from the next tick.

RAMP RATHER THAN A STEP. Speed is ramped from zero over accel_mps2 rather than
commanded outright, because the whole point of the exercise is often to measure
the drivetrain, and a step command makes the wheels slip - which is exactly the
error a gain calibration must not contain. It also means letting go of RB and
grabbing it again restarts from zero instead of from full speed.
"""

import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class StraightLineNode(Node):

    def __init__(self):
        super().__init__('straight_line')

        self.declare_parameter('out_topic', '/vesc/high_level/ackermann_cmd')
        self.declare_parameter('rate_hz', 50.0)
        # [m/s] Target. Deliberately low: this mode has no idea what is in
        # front of it, and the joystick is the only thing that can stop it.
        self.declare_parameter('speed_mps', 1.0)
        # [m/s^2] How fast the command climbs to speed_mps. See the ramp note
        # in the module docstring.
        self.declare_parameter('accel_mps2', 1.0)
        # [rad] Steering. Nonzero drives a constant-radius circle, which is how
        # steering_angle_to_servo_gain is measured: tan(delta) = wheelbase / R.
        self.declare_parameter('steering_rad', 0.0)

        self.out_topic = str(self.get_parameter('out_topic').value)
        self.rate_hz = max(1.0, float(self.get_parameter('rate_hz').value))
        self.speed_mps = float(self.get_parameter('speed_mps').value)
        self.accel_mps2 = max(0.0, float(self.get_parameter('accel_mps2').value))
        self.steering_rad = float(self.get_parameter('steering_rad').value)

        self.add_on_set_parameters_callback(self._on_params)

        self.commanded = 0.0
        self.pub = self.create_publisher(AckermannDriveStamped, self.out_topic, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

        self.get_logger().warn(
            f"straight_line: {self.speed_mps:.2f} m/s at "
            f"{self.steering_rad:+.4f} rad, ramping at {self.accel_mps2:.2f} "
            f"m/s^2, publishing to {self.out_topic}. NOTHING MOVES UNTIL RB IS "
            f"HELD - the mux drops this to zero otherwise.")

    def _on_params(self, params):
        """Live, so speed can be raised without stopping and restarting.

        Only the three driving numbers. out_topic and rate_hz are structural -
        the publisher and timer are already built around them.
        """
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'speed_mps':
                self.speed_mps = float(p.value)
            elif p.name == 'accel_mps2':
                self.accel_mps2 = max(0.0, float(p.value))
            elif p.name == 'steering_rad':
                self.steering_rad = float(p.value)
        return SetParametersResult(successful=True)

    def tick(self):
        step = self.accel_mps2 / self.rate_hz if self.accel_mps2 > 0.0 else math.inf
        target = self.speed_mps
        if self.commanded < target:
            self.commanded = min(target, self.commanded + step)
        else:
            self.commanded = max(target, self.commanded - step)

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(self.commanded)
        msg.drive.steering_angle = float(self.steering_rad)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StraightLineNode()
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
