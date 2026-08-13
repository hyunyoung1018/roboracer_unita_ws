#ifndef PERCEPTION_DETECT_HPP_
#define PERCEPTION_DETECT_HPP_

// C++ obstacle detector, ported from unicorn-racing-stack/perception.
//
// Differences from that port, all deliberate - see detect.cpp for the
// reasoning at each site:
//
//   - No grid_filter. UNIST decides whether a lidar point is on the track by
//     testing it against an eroded occupancy image; we test it in Frenet
//     against the waypoints' own d_left/d_right, which is what ForzaETH's
//     laserPointOnTrack did before that port replaced it. Their header still
//     declares that function with no definition left anywhere - this file is
//     where it comes back.
//   - The check is per point, before clustering, which is where UNIST applies
//     theirs. The python detector this replaced applied it to the cluster mean
//     instead, so a wall was only rejected when its fitted centre happened to
//     land off the track - which is how a stationary car came to report
//     anywhere between one and six obstacles from one frame to the next.
//   - Obstacles report their real per-axis extent rather than a square whose
//     side is the cluster's longest dimension.
//   - detect_fov_deg, max_viewing_distance and max_detect_d_m are
//     honoured; the last is a hard cap on how far off the raceline
//     anything is worth clustering, because the track boundary is a poor
//     proxy for that on a map with wide sections.
//
// The python detector is in the history up to the merge of the perception-cpp
// branch, where both were installed and selectable so that the two could be
// compared on the same track.

#include <rclcpp/rclcpp.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include <f110_msgs/msg/obstacle.hpp>
#include <f110_msgs/msg/obstacle_array.hpp>
#include <f110_msgs/msg/wpnt_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/float32.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "frenet_conversion.h"

#include <string>
#include <vector>

namespace perception
{

/// One lidar return that survived the on-track test, in both frames.
///
/// The Frenet pair is carried alongside the cartesian one because the
/// on-track test has already paid for it. Clustering and the L-shape fit
/// want x/y; the published extents want s/d; converting twice would be the
/// only reason to keep them apart.
struct ScanPoint
{
  double x{0.0};
  double y{0.0};
  double s{0.0};
  double d{0.0};
};

using Cluster = std::vector<ScanPoint>;

/// A fitted obstacle, in the form the message wants it.
struct Obstacle
{
  int id{0};

  // L-shape fit, map frame.
  double center_x{0.0};
  double center_y{0.0};
  double size{0.0};   ///< side of the bounding square: max(width, height)
  double theta{0.0};  ///< rectangle orientation, for the marker

  // Frenet position, and the measured per-axis extents.
  double s_center{0.0};
  double d_center{0.0};
  double s_back{0.0};   ///< distance from s_center to the trailing edge
  double s_front{0.0};  ///< distance from s_center to the leading edge
  double d_right{0.0};  ///< distance from d_center to the right edge
  double d_left{0.0};   ///< distance from d_center to the left edge
};

class Detect : public rclcpp::Node
{
public:
  Detect();

private:
  // --- callbacks ---
  void laserCb(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg);
  void pathCb(const f110_msgs::msg::WpntArray::ConstSharedPtr msg);
  rcl_interfaces::msg::SetParametersResult dynParamCb(
    const std::vector<rclcpp::Parameter> & params);
  void timerCallback();

  // --- pipeline ---
  bool lookupLaserToMap(
    const sensor_msgs::msg::LaserScan & scan, tf2::Transform * transform);
  std::vector<Cluster> clustering(
    const sensor_msgs::msg::LaserScan & scan, const tf2::Transform & laser_to_map);
  std::vector<Obstacle> fittingLShape(const std::vector<Cluster> & clusters);
  void checkObstacles(std::vector<Obstacle> * obstacles);

  // --- helpers ---
  /// True when (s, d) falls inside the inflated track boundaries.
  bool laserPointOnTrack(double d, int idx) const;
  /// Declare a numeric parameter that accepts an int or a double from yaml.
  double declareNumber(const std::string & name, double default_value);
  static double wrap(double value, double length);

  // --- publishing ---
  void publishBreakpoints(const std::vector<Cluster> & clusters);
  void publishObstaclesMessage();
  void publishObstaclesMarkers();
  visualization_msgs::msg::MarkerArray clearMarkers() const;

  // --- ros plumbing ---
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr global_wpnts_sub_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr breakpoints_markers_pub_;
  rclcpp::Publisher<f110_msgs::msg::ObstacleArray>::SharedPtr obstacles_msg_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacles_marker_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr latency_pub_;

  rclcpp::TimerBase::SharedPtr timer_;
  OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // --- parameters ---
  double rate_{20.0};
  double lambda_angle_{0.0};
  double sigma_{0.0};
  double min_2_points_dist_{0.01};
  double new_cluster_threshold_m_{0.4};
  int min_size_n_{4};
  /// Floor on a measured extent, per axis. NOT the modelled size: the
  /// footprint is settled over time by the tracker.
  double min_size_m_{0.20};
  double max_size_m_{0.6};
  double max_viewing_distance_{9.0};
  double boundaries_inflation_{0.1};
  double detect_fov_deg_{360.0};
  double max_detect_d_m_{0.90};
  bool measuring_{false};

  // --- state ---
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg_;
  rclcpp::Time current_stamp_;
  std::vector<Obstacle> tracked_obstacles_;
  std::vector<double> d_left_array_;   ///< already inflated inward
  std::vector<double> d_right_array_;  ///< already inflated inward, positive
  double track_length_{0.0};
  bool have_trajectory_{false};
  /// Seeded into the converter's proximity search; carried between points.
  int closest_idx_{0};
  tf2::Vector3 sensor_origin_;

  frenet_conversion::FrenetConverter frenet_converter_;
};

}  // namespace perception

#endif  // PERCEPTION_DETECT_HPP_
