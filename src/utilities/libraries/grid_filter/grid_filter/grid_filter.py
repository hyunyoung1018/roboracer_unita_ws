"""Conservative occupancy-grid lookup."""

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

    def first_outside_index(self, xy):
        """Index of the first sample that is not known free space, or None.

        Same verdict as calling :meth:`is_point_inside` on each point in turn,
        computed as four array operations instead of one Python call per point.
        The per-point form costs a world_to_pixel - two trig multiplies, a
        divide and a floor, all on Python scalars - for every sample of every
        candidate path, and the planner runs it over the raw path AND the
        smoothed one on every plan. On a 120-point path at 20 Hz that is 4800
        interpreted calls a second for a test that is a single array index.

        Returns None when the grid is not ready, matching is_point_inside's
        "not known free" verdict for index 0 in that case is NOT wanted here -
        callers check `ready` first, and reporting no failure on a missing map
        would let an unchecked path through. So: not ready -> index 0.
        """
        points = np.asarray(xy, dtype=float)
        if points.size == 0:
            return None
        if self.eroded_image is None or self.origin is None or self.resolution is None:
            return 0
        dx = points[:, 0] - self.origin[0]
        dy = points[:, 1] - self.origin[1]
        cos_yaw, sin_yaw = cos(self.origin_yaw), sin(self.origin_yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        px = np.floor(local_x / self.resolution).astype(np.int64)
        py = np.floor(local_y / self.resolution).astype(np.int64)
        height, width = self.eroded_image.shape
        inside_grid = (px >= 0) & (py >= 0) & (px < width) & (py < height)
        free = np.zeros(len(points), dtype=bool)
        # Index only the in-bounds samples; an out-of-bounds pixel stays False,
        # which is the same answer is_point_inside gives.
        if inside_grid.any():
            free[inside_grid] = (
                self.eroded_image[py[inside_grid], px[inside_grid]] == 255)
        bad = np.flatnonzero(~free)
        return int(bad[0]) if bad.size else None
