#!/usr/bin/env python3
"""
raceline_generator - build the global raceline from a saved map.

Offline, run-once node. It reads maps/<map>/<map>.{png,yaml}, extracts the
centerline and track bounds, runs two optimizations, then writes
global_waypoints.json into the map directory and publishes everything for RViz.

It never drives the car and never runs SLAM: the map has to exist already.

Compared with the ForzaETH global_planner node this comes from:

  * No create_map / create_global_path / map_editor mode flags. Which stage runs
    is decided by which launch file you start, so the node does exactly one job.
  * No blocking input() and no blocking matplotlib windows in the main path.
  * The start pose is resolved explicitly (see _resolve_start_pose) instead of
    being read from a non-standard `initial_pose` key that only maps produced by
    their own mapping node have.
"""

import os

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from skimage.morphology import skeletonize
from std_msgs.msg import Float32
from visualization_msgs.msg import MarkerArray

from f110_msgs.msg import WpntArray

from . import map_io, map_processing
from .map_processing import interp_track
from .markers import (add_centerline_heading, create_centerline_markers,
                      create_trackbounds_markers, create_wpnts_markers)
from .readwrite_global_waypoints import write_global_waypoints

# Importable as a top level name because raceline/__init__.py puts this
# package's directory on sys.path. See the note there.
from global_racetrajectory_optimization.trajectory_optimizer import (
    trajectory_optimizer)

# The centerline is resampled to this step before widths are measured. Finer
# than the optimizer needs, but it makes the nearest-boundary search accurate.
CENTERLINE_STEP_M = 0.1

# trajectory_optimizer() reads <vehicle_config_dir>/racecar_f110.ini and
# <vehicle_config_dir>/veh_dyn_info/; the filename is hardcoded upstream.
VEHICLE_PARAMS_FILE = 'racecar_f110.ini'


class RacelineGenerator(Node):

    def __init__(self):
        super().__init__('raceline_generator')

        self.declare_parameter('map_name', '')
        self.declare_parameter('map_dir', '')
        # The vehicle parameter FILENAME is fixed at racecar_f110.ini by the
        # vendored trajectory_optimizer(); only its directory is configurable.
        self.declare_parameter('vehicle_config_dir', '')
        self.declare_parameter('safety_width', 0.7)
        self.declare_parameter('safety_width_sp', 0.7)
        self.declare_parameter('reverse', False)
        self.declare_parameter('show_plots', False)
        self.declare_parameter('filter_kernel_size', 0)
        # [x, y, theta]; empty means "fall back to track_meta.yaml / RViz / origin"
        self.declare_parameter('start_pose', [])
        self.declare_parameter('start_pose_timeout', 0.0)

        self.map_name = self.get_parameter('map_name').value
        self.safety_width = float(self.get_parameter('safety_width').value)
        self.safety_width_sp = float(self.get_parameter('safety_width_sp').value)
        self.reverse = bool(self.get_parameter('reverse').value)
        self.show_plots = bool(self.get_parameter('show_plots').value)
        self.vehicle_config_dir = self.get_parameter('vehicle_config_dir').value

        if not self.map_name:
            raise RuntimeError('map_name parameter is required')

        # Write artifacts back into src/ rather than the install tree, so a
        # generated raceline is version controlled with the map it belongs to.
        self.map_dir = map_io.resolve_source_dir(self.get_parameter('map_dir').value)
        if not self.map_dir:
            raise RuntimeError('map_dir parameter is required')
        self.get_logger().info(f'Map directory: {self.map_dir}')

        if not self.vehicle_config_dir:
            raise RuntimeError('vehicle_config_dir parameter is required')

        self.map_info_str = ''
        self.est_lap_time = Float32()
        self._clicked_pose = None

        self.pub_global_wpnts = self.create_publisher(WpntArray, '/global_waypoints', 10)
        self.pub_global_wpnts_sp = self.create_publisher(
            WpntArray, '/global_waypoints/shortest_path', 10)
        self.pub_centerline_wpnts = self.create_publisher(
            WpntArray, '/centerline_waypoints', 10)
        self.pub_global_markers = self.create_publisher(
            MarkerArray, '/global_waypoints/markers', 10)
        self.pub_global_markers_sp = self.create_publisher(
            MarkerArray, '/global_waypoints/shortest_path/markers', 10)
        self.pub_centerline_markers = self.create_publisher(
            MarkerArray, '/centerline_waypoints/markers', 10)
        self.pub_trackbounds = self.create_publisher(MarkerArray, '/trackbounds/markers', 10)

        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self._initialpose_cb, 1)

    # ------------------------------------------------------------------ setup

    def _optimize(self, track_csv: str, opt_type: str, safety_width: float):
        """
        Call the vendored ForzaETH optimizer for one reference track.

        Only the argument shapes differ from what that function wants:
        it takes the track path WITHOUT the .csv suffix (it appends it itself),
        and a directory that holds racecar_f110.ini plus veh_dyn_info/.
        `safety_width` is passed through as w_veh, overriding width_opt in the ini.
        """
        ini_path = os.path.join(self.vehicle_config_dir, VEHICLE_PARAMS_FILE)
        if not os.path.isfile(ini_path):
            raise FileNotFoundError(f'Vehicle parameter file not found: {ini_path}')

        traj, bound_r, bound_l, est_lap_time = trajectory_optimizer(
            input_path=self.vehicle_config_dir,
            track_name=os.path.splitext(track_csv)[0],
            curv_opt_type=opt_type,
            safety_width=safety_width,
            plot=self.show_plots)

        self.get_logger().info(
            f'[{opt_type}] estimated lap time {est_lap_time:.3f}s, '
            f'max speed {np.amax(traj[:, 5]):.3f}m/s')
        return traj, bound_r, bound_l, float(est_lap_time)

    def _initialpose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        q = msg.pose.pose.orientation
        # yaw from a planar quaternion; no need for a full conversion library
        theta = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                           1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._clicked_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, theta)
        self.get_logger().info(
            f'Start pose from RViz: x={self._clicked_pose[0]:.3f} '
            f'y={self._clicked_pose[1]:.3f} theta={self._clicked_pose[2]:.3f}')

    def _resolve_start_pose(self):
        """
        Decide where s=0 sits and which way the track is driven.

        The start pose fixes two things: the point the centerline is cut at, and
        (through its heading) which boundary counts as right vs left. Getting it
        wrong shifts every waypoint index, which then shifts the speed sectors.

        Priority: explicit parameter, then track_meta.yaml from a previous run,
        then an RViz "2D Pose Estimate" click if we were told to wait for one,
        then the map origin. The origin fallback is what race_stack and UNIST
        both do unconditionally; it is only correct when the track happens to
        pass near (0, 0), so it warns loudly.
        """
        param = list(self.get_parameter('start_pose').value or [])
        if len(param) == 3:
            return (float(param[0]), float(param[1]), float(param[2])), 'param'

        meta = map_io.read_track_meta(self.map_dir)
        if meta and 'start_pose' in meta:
            sp = meta['start_pose']
            self.get_logger().info(f'Start pose from {map_io.track_meta_path(self.map_dir)}')
            return (float(sp['x']), float(sp['y']), float(sp['theta'])), 'meta'

        timeout = float(self.get_parameter('start_pose_timeout').value)
        if timeout > 0.0:
            self.get_logger().info(
                f'Waiting up to {timeout:.0f}s for a "2D Pose Estimate" in RViz '
                f'to set the start line...')
            deadline = self.get_clock().now().nanoseconds + int(timeout * 1e9)
            while rclpy.ok() and self._clicked_pose is None:
                if self.get_clock().now().nanoseconds > deadline:
                    break
                rclpy.spin_once(self, timeout_sec=0.1)
            if self._clicked_pose is not None:
                return self._clicked_pose, 'rviz'
            self.get_logger().warn('No pose received before the timeout.')

        self.get_logger().warn(
            'Falling back to the map origin (0, 0, 0) as the start pose. s=0 will '
            'land wherever the centerline passes closest to the origin, which is '
            'arbitrary unless the track actually starts there. Set the start_pose '
            'parameter, or use start_pose_timeout and click in RViz.')
        return (0.0, 0.0, 0.0), 'default'

    # ------------------------------------------------------------------- plan

    def generate(self) -> bool:
        image, resolution, origin, map_meta = map_io.load_map(self.map_dir, self.map_name)
        self.get_logger().info(
            f'Loaded map {self.map_name}: {image.shape[1]}x{image.shape[0]} cells '
            f'@ {resolution} m/cell, origin {origin}')

        binary = map_io.binarize(
            image, map_meta,
            filter_kernel_size=int(self.get_parameter('filter_kernel_size').value))
        skeleton = skeletonize(binary, method='lee')

        start_pose, pose_source = self._resolve_start_pose()

        # --- centerline ----------------------------------------------------
        self.get_logger().info('Extracting centerline...')
        centerline = map_processing.extract_centerline(skeleton, map_resolution=resolution)
        centerline_smooth = map_processing.smooth_centerline(centerline)

        centerline_meter = np.zeros(np.shape(centerline_smooth))
        centerline_meter[:, 0] = centerline_smooth[:, 0] * resolution + origin[0]
        centerline_meter[:, 1] = centerline_smooth[:, 1] * resolution + origin[1]
        centerline_meter = np.column_stack(
            (centerline_meter, np.zeros((centerline_meter.shape[0], 2))))
        centerline_meter_int = interp_track(centerline_meter, stepsize=CENTERLINE_STEP_M)[:, :2]

        # --- driving direction ---------------------------------------------
        # The contour comes out in whatever order cv2 walked it. Compare its
        # local direction at the start pose against the car's heading and flip
        # if they disagree, so s increases the way the car actually drives.
        cent_distance = np.hypot(centerline_meter_int[:, 0] - start_pose[0],
                                 centerline_meter_int[:, 1] - start_pose[1])
        min_dist_ind = int(np.argmin(cent_distance))
        cent_direction = np.angle([complex(
            centerline_meter_int[min_dist_ind, 0] - centerline_meter_int[min_dist_ind - 1, 0],
            centerline_meter_int[min_dist_ind, 1] - centerline_meter_int[min_dist_ind - 1, 1])])
        self.get_logger().info(f'Centerline direction at start: {float(cent_direction[0]):.3f} rad')

        if not map_processing.compare_direction(cent_direction, start_pose[2]):
            centerline_smooth = np.flip(centerline_smooth, axis=0)
            centerline_meter_int = np.flip(centerline_meter_int, axis=0)
            self.get_logger().info('Centerline flipped to match the start heading')

        if self.reverse:
            centerline_smooth = np.flip(centerline_smooth, axis=0)
            centerline_meter_int = np.flip(centerline_meter_int, axis=0)
            self.get_logger().info('Centerline flipped again (reverse:=true)')

        # --- track bounds ---------------------------------------------------
        dist_transform = None
        bound_r_water = bound_l_water = None
        watershed_ok = True
        try:
            self.get_logger().info('Extracting track bounds with watershed...')
            bound_r_water, bound_l_water = map_processing.extract_track_bounds(
                centerline_smooth, binary,
                map_resolution=resolution, map_origin=origin,
                start_pose=start_pose, show_plots=self.show_plots)
        except IOError as exc:
            self.get_logger().warn(f'Watershed failed ({exc}).')
            self.get_logger().warn(
                'Falling back to a distance transform, which yields one symmetric '
                'width per point - the raceline will be conservative wherever the '
                'centerline is not actually centred.')
            watershed_ok = False
            import cv2
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        cent_with_dist = map_processing.add_dist_to_cent(
            centerline_smooth=centerline_smooth,
            centerline_meter=centerline_meter_int,
            map_resolution=resolution,
            dist_transform=dist_transform,
            bound_r=bound_r_water, bound_l=bound_l_water,
            reverse=self.reverse)

        # Make the waypoint at the selected start pose the first point of the
        # completed reference track. Rotating only after attaching the widths
        # keeps pixel-space widths paired with their interpolated metric points.
        start_idx = int(np.argmin(np.hypot(
            cent_with_dist[:, 0] - start_pose[0],
            cent_with_dist[:, 1] - start_pose[1])))
        cent_with_dist = np.roll(cent_with_dist, -start_idx, axis=0)
        start_distance = np.hypot(cent_with_dist[0, 0] - start_pose[0],
                                  cent_with_dist[0, 1] - start_pose[1])
        self.get_logger().info(
            f'Centerline index 0 aligned to start pose '
            f'(distance {start_distance:.3f}m)')

        track_csv = map_io.write_centerline_csv(self.map_dir, cent_with_dist)
        self.get_logger().info(f'Wrote reference track: {track_csv}')

        centerline_wpnts, centerline_markers = create_centerline_markers(cent_with_dist)
        add_centerline_heading(centerline_wpnts, stepsize=CENTERLINE_STEP_M)

        # --- main raceline: iterative minimum curvature ----------------------
        self.get_logger().info('Optimizing raceline (mincurv_iqp)...')
        try:
            traj_iqp, bound_r_iqp, bound_l_iqp, est_t_iqp = self._optimize(
                track_csv, 'mincurv_iqp', self.safety_width)
        except (RuntimeError, ValueError) as exc:
            self.get_logger().error(f'Minimum curvature optimization failed: {exc}')
            return False

        self.map_info_str += f'IQP estimated lap time: {round(est_t_iqp, 4)}s; '
        self.map_info_str += f'IQP maximum speed: {round(float(np.amax(traj_iqp[:, 5])), 4)}m/s; '

        # Watershed bounds follow the real walls; the optimizer's are derived
        # from the reference track normals and cut corners. Prefer the former.
        if watershed_ok:
            bound_r_iqp, bound_l_iqp = bound_r_water, bound_l_water

        bounds_markers = create_trackbounds_markers(bound_r_iqp, bound_l_iqp)

        d_right_iqp, d_left_iqp = map_processing.dist_to_bounds(
            traj_iqp, bound_r_iqp, bound_l_iqp, reverse=self.reverse)
        wpnts_iqp, markers_iqp = create_wpnts_markers(
            traj_iqp, d_right_iqp, d_left_iqp)

        # --- overtaking line: shortest path ----------------------------------
        # Run on the same reference track but with the wider/narrower safety
        # margin, giving a second line the state machine can switch to.
        self.get_logger().info('Optimizing overtaking line (shortest_path)...')
        try:
            traj_sp, bound_r_sp, bound_l_sp, est_t_sp = self._optimize(
                track_csv, 'shortest_path', self.safety_width_sp)
        except (RuntimeError, ValueError) as exc:
            self.get_logger().error(f'Shortest path optimization failed: {exc}')
            return False

        self.est_lap_time.data = float(est_t_sp)
        self.map_info_str += f'SP estimated lap time: {round(est_t_sp, 4)}s; '
        self.map_info_str += f'SP maximum speed: {round(float(np.amax(traj_sp[:, 5])), 4)}m/s; '

        if watershed_ok:
            bound_r_sp, bound_l_sp = bound_r_water, bound_l_water

        d_right_sp, d_left_sp = map_processing.dist_to_bounds(
            traj_sp, bound_r_sp, bound_l_sp, reverse=self.reverse)
        wpnts_sp, markers_sp = create_wpnts_markers(
            traj_sp, d_right_sp, d_left_sp, second_traj=True)

        # --- publish + persist ------------------------------------------------
        self.pub_centerline_wpnts.publish(centerline_wpnts)
        self.pub_centerline_markers.publish(centerline_markers)
        self.pub_trackbounds.publish(bounds_markers)
        self.pub_global_wpnts.publish(wpnts_iqp)
        self.pub_global_markers.publish(markers_iqp)
        self.pub_global_wpnts_sp.publish(wpnts_sp)
        self.pub_global_markers_sp.publish(markers_sp)

        write_global_waypoints(
            self.map_dir,
            self.map_info_str,
            self.est_lap_time,
            centerline_markers,
            centerline_wpnts,
            markers_iqp,
            wpnts_iqp,
            markers_sp,
            wpnts_sp,
            bounds_markers)

        meta_path = map_io.write_track_meta(
            self.map_dir, start_pose, self.reverse, pose_source)
        self.get_logger().info(f'Wrote {meta_path}')
        self.get_logger().info(
            f'Done. {os.path.join(self.map_dir, "global_waypoints.json")} is ready.')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = RacelineGenerator()
    ok = False
    try:
        ok = node.generate()
    finally:
        # Give the transport a moment to flush the markers to RViz before the
        # node goes away - this process exits as soon as planning is done.
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    main()
