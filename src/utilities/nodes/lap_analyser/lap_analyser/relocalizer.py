#!/usr/bin/env python3

'''
Experimental standalone node: does NOT modify lap_analyser.py.

Every N laps, re-seeds the particle filter at the finish-line waypoint's
known map pose (x, y, heading), instead of requiring a manual RViz
"2D Pose Estimate" click. Publishing to /initialpose reuses
particle_filter.py's existing clicked_pose() -> initialize_particles_pose()
path unchanged - this node only publishes one message, nothing else in the
stack needs to change.

Gets the lap boundary from lap_analyser's own 'lap_data' topic (published
once per completed lap, contains lap_count) instead of re-implementing the
finish-line-crossing detection - so lap_analyser.py stays untouched and this
can be run or killed independently to A/B test the effect.
'''

import math

import numpy as np
import rclpy
from f110_msgs.msg import LapData, WpntArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class AutoRelocalizer(Node):
    def __init__(self):
        super().__init__('auto_relocalizer')

        self.RELOCALIZE_EVERY_N_LAPS = int(
            self.declare_parameter('relocalize_every_n_laps', 10).value)

        self.wp_flag = False
        self.global_waypoints = None  # [s_m, x_m, y_m, psi_rad] per row
        self.current_s = 0.0

        self.gb_wpnts_sub = self.create_subscription(
            WpntArray, '/global_waypoints', self.waypoints_cb, 10)
        self.odom_frenet_sub = self.create_subscription(
            Odometry, '/car_state/odom_frenet', self.frenet_odom_cb, 10)
        self.lap_data_sub = self.create_subscription(
            LapData, 'lap_data', self.lap_data_cb, 10)

        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 1)

        self.get_logger().info(
            f"AutoRelocalizer started: relocalize_every_n_laps={self.RELOCALIZE_EVERY_N_LAPS}")

    def waypoints_cb(self, data: WpntArray):
        if not self.wp_flag:
            self.global_waypoints = np.array(
                [[w.s_m, w.x_m, w.y_m, w.psi_rad] for w in data.wpnts])
            self.wp_flag = True

    def frenet_odom_cb(self, msg: Odometry):
        self.current_s = msg.pose.pose.position.x

    def lap_data_cb(self, msg: LapData):
        if not self.wp_flag:
            self.get_logger().warn(
                "lap completed but /global_waypoints not received yet, skipping relocalize")
            return
        if self.RELOCALIZE_EVERY_N_LAPS <= 0:
            return
        if msg.lap_count > 0 and msg.lap_count % self.RELOCALIZE_EVERY_N_LAPS == 0:
            self.publish_relocalize_pose(msg.lap_count)

    def publish_relocalize_pose(self, lap_count):
        s_vals = self.global_waypoints[:, 0]
        idx = np.argmin(np.abs(s_vals - self.current_s))
        _, x, y, psi = self.global_waypoints[idx]

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = math.sin(float(psi) / 2.0)
        msg.pose.pose.orientation.w = math.cos(float(psi) / 2.0)
        self.initialpose_pub.publish(msg)
        self.get_logger().warn(
            f"[AutoRelocalizer] auto pose re-estimate at lap {lap_count} "
            f"(s={self.current_s:.2f}) -> x={x:.2f}, y={y:.2f}, psi={psi:.2f} rad")


def main():
    rclpy.init()
    node = AutoRelocalizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
