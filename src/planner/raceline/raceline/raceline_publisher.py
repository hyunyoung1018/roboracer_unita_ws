#!/usr/bin/env python3
"""
raceline_publisher - republish a saved raceline at runtime.

Reads maps/<map>/global_waypoints.json and publishes it on a timer. This is the
only raceline node that runs while racing: no optimization, no map images, no
heavy dependencies.

Everything is latched (TRANSIENT_LOCAL) so a consumer that starts later still
receives the waypoints instead of waiting for the next tick.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from std_msgs.msg import Float32, String
from visualization_msgs.msg import MarkerArray

from f110_msgs.msg import WpntArray

from .paths import resolve_source_dir
from .readwrite_global_waypoints import read_global_waypoints


class RacelinePublisher(Node):

    def __init__(self):
        super().__init__('raceline_publisher')

        self.declare_parameter('map_dir', '')
        self.declare_parameter('rate', 1.0)
        self.declare_parameter('publish_markers', True)

        map_dir = resolve_source_dir(self.get_parameter('map_dir').value)
        if not map_dir:
            raise RuntimeError('map_dir parameter is required')
        self.publish_markers = bool(self.get_parameter('publish_markers').value)

        (self.map_info, self.est_lap_time,
         self.centerline_markers, self.centerline_wpnts,
         self.global_markers, self.global_wpnts,
         self.global_sp_markers, self.global_sp_wpnts,
         self.trackbounds_markers) = read_global_waypoints(map_dir)

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)

        self.pub_global = self.create_publisher(WpntArray, '/global_waypoints', latched)
        self.pub_global_sp = self.create_publisher(
            WpntArray, '/global_waypoints/shortest_path', latched)
        self.pub_centerline = self.create_publisher(
            WpntArray, '/centerline_waypoints', latched)
        self.pub_map_info = self.create_publisher(String, '/map_infos', latched)
        self.pub_lap_time = self.create_publisher(Float32, '/estimated_lap_time', latched)

        if self.publish_markers:
            self.pub_global_mrk = self.create_publisher(
                MarkerArray, '/global_waypoints/markers', latched)
            self.pub_global_sp_mrk = self.create_publisher(
                MarkerArray, '/global_waypoints/shortest_path/markers', latched)
            self.pub_centerline_mrk = self.create_publisher(
                MarkerArray, '/centerline_waypoints/markers', latched)
            self.pub_bounds_mrk = self.create_publisher(
                MarkerArray, '/trackbounds/markers', latched)

        rate = float(self.get_parameter('rate').value)
        self.create_timer(1.0 / rate if rate > 0.0 else 1.0, self.publish)

        self.get_logger().info(
            f'Publishing raceline from {map_dir} '
            f'({len(self.global_wpnts.wpnts)} waypoints, '
            f'est. lap time {self.est_lap_time.data:.3f}s)')

    def publish(self):
        self.pub_global.publish(self.global_wpnts)
        self.pub_global_sp.publish(self.global_sp_wpnts)
        self.pub_centerline.publish(self.centerline_wpnts)
        self.pub_map_info.publish(self.map_info)
        self.pub_lap_time.publish(self.est_lap_time)

        if self.publish_markers:
            self.pub_global_mrk.publish(self.global_markers)
            self.pub_global_sp_mrk.publish(self.global_sp_markers)
            self.pub_centerline_mrk.publish(self.centerline_markers)
            self.pub_bounds_mrk.publish(self.trackbounds_markers)


def main(args=None):
    rclpy.init(args=args)
    node = RacelinePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
