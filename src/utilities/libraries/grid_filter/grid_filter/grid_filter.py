"""Conservative occupancy-grid lookup adapted from UNICORN's grid_filter."""

from math import atan2, cos, sin

import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile


class GridFilter:
    """Erode free space and test Cartesian path samples against the result."""

    def __init__(self, node, map_topic='/map', kernel_size=3,
                 occupied_threshold=50, unknown_is_occupied=True):
        self.node = node
        self.kernel_size = max(1, int(kernel_size))
        self.occupied_threshold = int(occupied_threshold)
        self.unknown_is_occupied = bool(unknown_is_occupied)
        self.resolution = None
        self.origin = None
        self.origin_yaw = 0.0
        self.free_image = None
        self.eroded_image = None

        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.subscription = node.create_subscription(
            OccupancyGrid, map_topic, self.map_callback, qos)

    @property
    def ready(self):
        return self.eroded_image is not None

    def map_callback(self, msg):
        self.resolution = float(msg.info.resolution)
        self.origin = (
            float(msg.info.origin.position.x),
            float(msg.info.origin.position.y),
        )
        q = msg.info.origin.orientation
        self.origin_yaw = atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        data = np.asarray(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width))
        free = data < self.occupied_threshold
        if self.unknown_is_occupied:
            free &= data >= 0
        self.free_image = np.where(free, 255, 0).astype(np.uint8)
        self._update_image()
        self.node.get_logger().info(
            f'Grid filter ready: {msg.info.width}x{msg.info.height}, '
            f'{self.resolution:.3f} m/cell, erosion={self.kernel_size}')

    def set_erosion_kernel_size(self, size):
        self.kernel_size = max(1, int(size))
        self._update_image()

    def _update_image(self):
        if self.free_image is None:
            return
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        self.eroded_image = cv2.erode(self.free_image, kernel)

    def world_to_pixel(self, x, y):
        if self.origin is None or self.resolution is None:
            return None
        dx = float(x) - self.origin[0]
        dy = float(y) - self.origin[1]
        # OccupancyGrid coordinates are expressed in the origin pose frame.
        local_x = cos(self.origin_yaw) * dx + sin(self.origin_yaw) * dy
        local_y = -sin(self.origin_yaw) * dx + cos(self.origin_yaw) * dy
        return int(np.floor(local_x / self.resolution)), int(np.floor(local_y / self.resolution))

    def is_point_inside(self, x, y):
        """Return true only for known free space after safety erosion."""
        if self.eroded_image is None:
            return False
        pixel = self.world_to_pixel(x, y)
        if pixel is None:
            return False
        px, py = pixel
        height, width = self.eroded_image.shape
        if px < 0 or py < 0 or px >= width or py >= height:
            return False
        return bool(self.eroded_image[py, px] == 255)

    def is_path_inside(self, xy):
        return all(self.is_point_inside(x, y) for x, y in np.asarray(xy))
