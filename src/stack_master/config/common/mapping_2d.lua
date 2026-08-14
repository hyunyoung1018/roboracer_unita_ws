-- Cartographer 2D SLAM config for MAPPING (real car only).
--
-- LiDAR + IMU + wheel odometry, each for the one thing it is good at.
--
-- Odometry was dropped here for a long time, for a real reason: the VESC is
-- sensorless, speed comes from a back-EMF flux observer that degrades as the
-- motor slows, vesc_to_odom clamps anything under 0.05 m/s to zero, and yaw
-- is not measured at all - it is integrated from the servo COMMAND through a
-- bicycle model. Creeping round tight turns is where all of that is worst.
--
-- What that left behind was a scan matcher with NO translation prior, and on
-- a track of long parallel walls it cannot tell 30 cm of travel from none.
-- Every lap came back short: the map compressed along its own length. That is
-- the failure use_online_correlative_scan_matching below was added to fight,
-- and it was not enough on its own.
--
-- Two things changed and both are needed:
--
--   * Mapping is now driven at a constant 0.8 m/s (hold A - see
--     joy_teleop.yaml), three times the openloop threshold, so the observer
--     is never in the region where it is untrusted.
--   * Cartographer only takes the MAGNITUDE from odometry. PoseExtrapolator
--     rotates the odometry velocity into the tracking frame using its own
--     orientation, which comes from the IMU whenever use_imu_data is true -
--     so the bicycle model's yaw never enters. Wheels say how far, the gyro
--     says which way.
--
-- tracking_frame must be the IMU frame. Cartographer hard-CHECKs that the IMU
-- is colocated with the tracking frame:
--
--     CHECK(sensor_to_tracking->translation().norm() < 1e-5)
--         << "The IMU frame must be colocated with the tracking frame."
--
-- and the racecar's IMU sits at (0.07, 0, 0.09) from base_link, so tracking on
-- base_link would abort at the first IMU message. Tracking on ego_racecar/imu
-- makes that transform the identity. published_frame stays on base_link; only
-- the IMU->tracking transform is constrained.
--
-- REQUIRES the vesc_driver IMU fixes (frame_id, and the g -> m/s^2 /
-- deg/s -> rad/s conversions). Without them there is no usable prior at all.

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  map_frame = "map",
  tracking_frame = "ego_racecar/imu",
  published_frame = "ego_racecar/base_link",
  odom_frame = "odom",
  provide_odom_frame = false,

  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,

  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  -- The UST-10LX sweeps for 25 ms. Splitting each scan into 10 chunks lets the
  -- IMU un-rotate them individually, instead of treating the whole sweep as one
  -- instant - at racing speeds the car turns noticeably within a single scan.
  -- Must be paired with num_accumulated_range_data below.
  num_subdivisions_per_laser_scan = 10,
  num_point_clouds = 0,

  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,

  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,

  -- Cartographer owns TF: map -> ego_racecar/base_link, published directly.
  -- Not REP-105 compliant (no odom frame appears), which is ForzaETH's choice
  -- and kept here: the alternative puts the drifty wheel odometry back into the
  -- TF path between the localizer and the car. Set provide_odom_frame = true if
  -- something downstream ever needs the odom frame.
  publish_to_tf = true,
  publish_tracked_pose = true,
  publish_frame_projected_to_2d = true,
}

MAP_BUILDER.use_trajectory_builder_2d = true
MAP_BUILDER.use_trajectory_builder_3d = false
MAP_BUILDER.num_background_threads = 4

TRAJECTORY_BUILDER_2D.use_imu_data = true

-- Range limits from the UST-10LX manual (datasheets/UST-10LX/): guaranteed
-- 0.06-10 m. The floor is raised to 0.12 to reject returns off the car's own
-- mount. missing_data_ray_length is what a max-range return is treated as when
-- carving free space.
TRAJECTORY_BUILDER_2D.min_range = 0.12
TRAJECTORY_BUILDER_2D.max_range = 10.0
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 5.0

-- One full sweep per matched scan: 10 subdivisions in, 10 accumulated out.
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 10

TRAJECTORY_BUILDER_2D.voxel_filter_size = 0.025
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_length = 0.5
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.min_num_points = 200
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_range = 10.0

-- A brute-force search around the IMU prior before the Ceres refinement. It
-- costs CPU but it is what keeps matching from sliding along a corridor when
-- the only prior is a rotation estimate.
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true

-- Insert a node for even a small rotation, so turns stay densely sampled.
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.1)

POSE_GRAPH.optimize_every_n_nodes = 100
POSE_GRAPH.constraint_builder.min_score = 0.6
POSE_GRAPH.constraint_builder.sampling_ratio = 0.3

-- Default is 7 m. A race track is self-similar - long parallel walls, repeated
-- corner shapes - and a wide search window lets a loop closure attach to the
-- wrong lap. Tight window, and the constraint has to be nearly where the pose
-- already says it is.
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 1.5
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(30.)

-- Zero ON PURPOSE, with use_odometry on. Odometry is a FRONT-END hint only:
-- it steadies the local scan matcher and never becomes a back-end constraint.
--
-- The back end is where loop closure lives, and a wheel odometry with any
-- scale error would pull against the closure that is meant to correct it -
-- the map would come back consistent with the wheels rather than with the
-- track. Raise these only with a measured reason to trust the distance the
-- wheels report to better than a percent.
POSE_GRAPH.optimization_problem.odometry_translation_weight = 0
POSE_GRAPH.optimization_problem.odometry_rotation_weight = 0

return options
