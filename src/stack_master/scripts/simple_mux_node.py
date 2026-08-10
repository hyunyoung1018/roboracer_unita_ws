#!/usr/bin/env python3

from copy import deepcopy

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import Bool

from controller.estop import EStop


class SimpleMuxNode(Node):

    def __init__(self):
        super().__init__('simple_mux')

        self.declare_parameter(
            'out_topic',
            'low_level/ackermann_cmd_mux/output'
        )
        self.declare_parameter(
            'in_topic',
            'high_level/ackermann_cmd'
        )
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('keyboard_joy_topic', '/joy_keyboard')
        self.declare_parameter('ego_control_topic', '/ego/use_keyboard')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/vesc/odom')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('joy_max_speed', 5.0)
        self.declare_parameter('joy_max_steer', 0.4)

        # Maximum age of a usable command.
        # Even when a mode is toggled on, a disconnected joystick or failed
        # autonomous controller results in a zero command.
        self.declare_parameter('joy_freshness_threshold', 0.2)

        self.declare_parameter('servo_min', 0.15)
        self.declare_parameter('servo_max', 0.85)
        self.declare_parameter(
            'steering_angle_to_servo_offset',
            0.5
        )
        self.declare_parameter(
            'steering_angle_to_servo_gain',
            -1.2135
        )
        self.declare_parameter('use_estop', False)
        self.declare_parameter('sim', False)

        # False: physical joystick
        # True: keyboard Joy input
        self.declare_parameter('use_keyboard', False)

        p = lambda name: self.get_parameter(name).value

        out_topic = p('out_topic')
        in_topic = p('in_topic')
        joy_topic = p('joy_topic')
        keyboard_joy_topic = p('keyboard_joy_topic')
        ego_control_topic = p('ego_control_topic')
        scan_topic = p('scan_topic')
        odom_topic = p('odom_topic')

        self.use_estop = p('use_estop')
        self.max_speed = p('joy_max_speed')
        self.max_steer = p('joy_max_steer')
        self.joy_freshness_threshold = p(
            'joy_freshness_threshold'
        )

        servo_offset = p('steering_angle_to_servo_offset')
        servo_gain = p('steering_angle_to_servo_gain')

        self.servo_max_abs = min(
            abs(
                (p('servo_max') - servo_offset)
                / servo_gain
            ),
            abs(
                (p('servo_min') - servo_offset)
                / servo_gain
            ),
        )

        # Driving state:
        # None         = no mode selected since startup
        # idle         = actively publish zero
        # humandrive   = joystick control
        # autodrive    = autonomous controller
        self.current_host = None

        self.human_drive = None
        self.autodrive = None
        self.scan = None
        self.odom = None

        self.use_keyboard = p('use_keyboard')

        # Previous button states, used for rising-edge detection.
        # Logitech F710 XInput:
        # B  = button 1
        # LB = button 4
        # RB = button 5
        self.prev_b = False
        self.prev_lb = False
        self.prev_rb = False

        self.create_subscription(
            AckermannDriveStamped,
            in_topic,
            self._drive_cb,
            10
        )
        self.create_subscription(
            Joy,
            joy_topic,
            self._joy_cb,
            10
        )
        self.create_subscription(
            Joy,
            keyboard_joy_topic,
            self._joy_keyboard_cb,
            10
        )
        self.create_subscription(
            Bool,
            ego_control_topic,
            self._ego_control_cb,
            10
        )

        if self.use_estop:
            self.estop = EStop(self)

            self.create_subscription(
                LaserScan,
                scan_topic,
                self._scan_cb,
                qos_profile_sensor_data
            )
            self.create_subscription(
                Odometry,
                odom_topic,
                self._odom_cb,
                10
            )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            out_topic,
            10
        )

        self.create_timer(
            1.0 / p('rate_hz'),
            self._loop
        )

    def _scan_cb(self, msg):
        self.scan = msg

    def _odom_cb(self, msg):
        self.odom = msg

    def _drive_cb(self, msg):
        self.autodrive = msg

    def _ego_control_cb(self, msg):
        new_use_keyboard = bool(msg.data)

        if new_use_keyboard != self.use_keyboard:
            # Stop before changing the human input source.
            self._enter_idle(
                'Human input source changed'
            )

            # Reset edge-detection state so a held button from the previous
            # input source cannot produce an unintended toggle.
            self.prev_b = False
            self.prev_lb = False
            self.prev_rb = False

        self.use_keyboard = new_use_keyboard

    def _is_fresh(self, msg):
        if msg is None:
            return False

        now = self.get_clock().now()
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        dt = (now - stamp).nanoseconds / 1e9

        return abs(dt) < self.joy_freshness_threshold

    def _clip(self, msg):
        out = deepcopy(msg)

        out.drive.steering_angle = max(
            -self.servo_max_abs,
            min(
                self.servo_max_abs,
                out.drive.steering_angle
            )
        )

        return out

    def _make_stop_command(self):
        stop = AckermannDriveStamped()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.drive.speed = 0.0
        stop.drive.steering_angle = 0.0
        stop.drive.acceleration = 0.0
        stop.drive.jerk = 0.0
        return stop

    def _publish_stop(self):
        stop = self._make_stop_command()

        if self.use_estop:
            stop = self.estop.should_stop(
                self.scan,
                self.odom,
                stop
            )

        self.drive_pub.publish(stop)

    def _enter_idle(self, reason):
        # Set the state before publishing so the timer cannot publish an old
        # driving command after this callback decides to stop.
        self.current_host = 'idle'
        self.human_drive = None

        # Do not wait for the next timer tick. Publish the stop immediately.
        self._publish_stop()
        self.get_logger().info(
            f'{reason}: vehicle stopped'
        )

    def _change_mode(self, new_mode):
        old_mode = self.current_host

        if old_mode == new_mode:
            # Pressing the currently active mode button toggles it off.
            self._enter_idle(
                f'{new_mode} toggled off'
            )
            return

        if old_mode in ('humandrive', 'autodrive'):
            # Stop once before transferring control between driving modes.
            self._publish_stop()
            self.human_drive = None

        self.current_host = new_mode
        self.get_logger().info(
            f'Drive mode changed: {old_mode} -> {new_mode}'
        )

    def _loop(self):
        if self.current_host is None:
            # No mode has been selected since startup.
            return

        if (
            self.current_host == 'autodrive'
            and self._is_fresh(self.autodrive)
        ):
            out = deepcopy(self.autodrive)

        elif (
            self.current_host == 'humandrive'
            and self._is_fresh(self.human_drive)
        ):
            out = deepcopy(self.human_drive)

        else:
            # idle, stale joystick command, or stale autonomous command
            out = self._make_stop_command()

        if self.use_estop:
            out = self.estop.should_stop(
                self.scan,
                self.odom,
                out
            )

        self.drive_pub.publish(out)

    def _joy_cb(self, msg):
        # Physical controller is used only when it is the selected input.
        if not self.use_keyboard:
            self._handle_joy(msg)

    def _joy_keyboard_cb(self, msg):
        # Keyboard Joy input is used only when it is selected.
        if self.use_keyboard:
            self._handle_joy(msg)

    def _handle_joy(self, msg):
        # Logitech F710 XInput button indices:
        # B  = 1
        # LB = 4
        # RB = 5
        b_pressed = (
            bool(msg.buttons[1])
            if len(msg.buttons) > 1
            else False
        )
        lb_pressed = (
            bool(msg.buttons[4])
            if len(msg.buttons) > 4
            else False
        )
        rb_pressed = (
            bool(msg.buttons[5])
            if len(msg.buttons) > 5
            else False
        )

        # Rising edges: released(0) -> pressed(1).
        b_rising = b_pressed and not self.prev_b
        lb_rising = lb_pressed and not self.prev_lb
        rb_rising = rb_pressed and not self.prev_rb

        # Store current states before performing mode changes.
        self.prev_b = b_pressed
        self.prev_lb = lb_pressed
        self.prev_rb = rb_pressed

        # B always has the highest priority.
        if b_rising:
            self._enter_idle('B stop button pressed')
            return

        # Simultaneous LB and RB is considered ambiguous and unsafe.
        if lb_rising and rb_rising:
            self._enter_idle(
                'LB and RB pressed simultaneously'
            )
            return

        if lb_rising:
            self._change_mode('humandrive')

        elif rb_rising:
            self._change_mode('autodrive')

        # Keep updating the joystick command while human mode is active.
        # LB does not need to remain held.
        if self.current_host == 'humandrive':
            drive = AckermannDriveStamped()
            drive.header.stamp = (
                self.get_clock().now().to_msg()
            )

            drive.drive.steering_angle = (
                msg.axes[3] * self.max_steer
                if len(msg.axes) > 3
                else 0.0
            )

            drive.drive.speed = (
                msg.axes[1] * self.max_speed
                if len(msg.axes) > 1
                else 0.0
            )

            self.human_drive = drive


def main(args=None):
    rclpy.init(args=args)
    node = SimpleMuxNode()

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