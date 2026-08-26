#!/usr/bin/env python3

"""A text overlay for RViz: what the run is doing, on one marker.

During a qualifying run the numbers worth watching are scattered across four
topics and a parameter server, and none of them are visible from the driver's
station. This draws them as a single TEXT_VIEW_FACING marker that rides above
the car, so a glance at RViz answers "which lap, how fast, and has the filter
been re-seeded".

    LAP 12
    2.51 m/s
    LAST 8.432  BEST 8.201
    PHASE 2  x0.60  t1.10
    RELOC x4  18s ago

Add it in RViz as a Marker display on /qual_hud. The marker lives in the car's
frame, so it follows the car rather than sitting at the map origin.

Everything is read-only - this node subscribes and draws, it never commands
anything, so it can be killed mid-run without consequence.

Re-localisation is shown two ways at once, because the moment it happens is
easier to miss than to read: the count goes up, and the whole overlay turns
cyan for a couple of seconds. Both manual "2D Pose Estimate" clicks and the
relocalizer's automatic re-seeds land on /initialpose, so both are counted.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from f110_msgs.msg import LapData


class QualHud(Node):
    def __init__(self):
        super().__init__('qual_hud')

        self.frame_id = self.declare_parameter(
            'frame_id', 'ego_racecar/base_link').value
        self.offset_x = float(self.declare_parameter('offset_x', 0.0).value)
        self.offset_y = float(self.declare_parameter('offset_y', 0.0).value)
        self.offset_z = float(self.declare_parameter('offset_z', 0.8).value)
        # [m] cap height of the text, in the frame above.
        self.text_height = float(self.declare_parameter('text_height', 0.28).value)
        self.rate_hz = float(self.declare_parameter('rate_hz', 5.0).value)
        # [s] how long the overlay stays highlighted after a re-localisation.
        self.reloc_flash_sec = float(
            self.declare_parameter('reloc_flash_sec', 2.5).value)

        self.speed = None
        self.lap_count = None
        self.last_lap = None
        self.best_lap = None
        self.reloc_count = 0
        self.reloc_at = None
        self.status = None

        self.create_subscription(Odometry, '/car_state/odom', self.odom_cb, 10)
        self.create_subscription(LapData, 'lap_data', self.lap_cb, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.initialpose_cb, 10)
        # Latched: the scheduler publishes a phase once when it changes, and a
        # HUD started afterwards still needs to know which one is in force.
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, 'qual_status', self.status_cb, latched)

        self.marker_pub = self.create_publisher(Marker, 'qual_hud', 10)
        self.create_timer(1.0 / max(1.0, self.rate_hz), self.draw)
        self.get_logger().info(
            f"QualHud drawing on /qual_hud in frame {self.frame_id}")

    # ------------------------------------------------------------------ #
    def odom_cb(self, msg: Odometry):
        self.speed = float(msg.twist.twist.linear.x)

    def lap_cb(self, msg: LapData):
        self.lap_count = int(msg.lap_count)
        self.last_lap = float(msg.lap_time)
        if self.best_lap is None or self.last_lap < self.best_lap:
            self.best_lap = self.last_lap

    def initialpose_cb(self, _msg):
        self.reloc_count += 1
        self.reloc_at = self.get_clock().now()

    def status_cb(self, msg: String):
        self.status = msg.data

    # ------------------------------------------------------------------ #
    def _lines(self):
        lines = [f"LAP {self.lap_count}" if self.lap_count is not None else "LAP -"]
        lines.append(f"{self.speed:.2f} m/s" if self.speed is not None else "- m/s")

        if self.last_lap is not None:
            lines.append(f"LAST {self.last_lap:.3f}  BEST {self.best_lap:.3f}")
        else:
            lines.append("LAST -  BEST -")

        lines.append(self.status if self.status else "PHASE -")

        if self.reloc_count:
            age = (self.get_clock().now() - self.reloc_at).nanoseconds / 1e9
            lines.append(f"RELOC x{self.reloc_count}  {age:.0f}s ago")
        else:
            lines.append("RELOC x0")
        return "\n".join(lines)

    def _fresh_reloc(self):
        if self.reloc_at is None:
            return False
        age = (self.get_clock().now() - self.reloc_at).nanoseconds / 1e9
        return age < self.reloc_flash_sec

    def draw(self):
        mark = Marker()
        mark.header.stamp = self.get_clock().now().to_msg()
        mark.header.frame_id = self.frame_id
        mark.ns = 'qual_hud'
        mark.id = 0
        mark.type = Marker.TEXT_VIEW_FACING
        mark.action = Marker.ADD
        mark.pose.position.x = self.offset_x
        mark.pose.position.y = self.offset_y
        mark.pose.position.z = self.offset_z
        mark.pose.orientation.w = 1.0
        mark.scale.z = self.text_height
        mark.color.a = 1.0
        if self._fresh_reloc():
            mark.color.r, mark.color.g, mark.color.b = 0.30, 0.90, 0.95
        else:
            mark.color.r, mark.color.g, mark.color.b = 0.95, 0.95, 0.95
        mark.text = self._lines()
        self.marker_pub.publish(mark)


def main():
    rclpy.init()
    node = QualHud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
