#!/usr/bin/env python3
"""Split the tracker output without changing any obstacle fields."""

import rclpy
from f110_msgs.msg import ObstacleArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class ObstacleRouter(Node):
    """Route each tracked obstacle according to its ``is_static`` flag."""

    def __init__(self):
        super().__init__('obstacle_router')
        self.static_pub = self.create_publisher(
            ObstacleArray, '/tracking/static_obstacles', 10)
        self.dynamic_pub = self.create_publisher(
            ObstacleArray, '/tracking/dynamic_obstacles', 10)
        self.create_subscription(
            ObstacleArray, '/tracking/obstacles', self._route, 10)

    def _route(self, msg: ObstacleArray) -> None:
        static_msg = ObstacleArray()
        dynamic_msg = ObstacleArray()
        static_msg.header = msg.header
        dynamic_msg.header = msg.header

        # Assign the original ROS messages. No obstacle is reconstructed, so all
        # present and future fields survive routing without an adapter update.
        static_msg.obstacles = [obs for obs in msg.obstacles if obs.is_static]
        dynamic_msg.obstacles = [obs for obs in msg.obstacles if not obs.is_static]

        # Publish empty arrays as well. Consumers then learn immediately that a
        # class disappeared instead of retaining their previous non-empty frame.
        self.static_pub.publish(static_msg)
        self.dynamic_pub.publish(dynamic_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleRouter()
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
