#include "detect.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace perception
{

using std::placeholders::_1;

namespace
{
/// L-shape fit search: 90 candidate orientations over a quarter turn. A
/// rectangle is symmetric under a quarter turn, so this covers every
/// distinguishable orientation at 1 degree resolution.
constexpr int kNumCandidates = 90;
}  // namespace

Detect::Detect()
: rclcpp::Node("detect"), current_stamp_(0, 0, RCL_ROS_TIME)
{
  measuring_ = this->declare_parameter<bool>("measure", false);

  rate_ = declareNumber("rate_detect", 20.0);
  min_size_n_ = static_cast<int>(declareNumber("min_size_n", 4));
  min_size_m_ = declareNumber("min_size_m", 0.20);
  max_size_m_ = declareNumber("max_size_m", 0.60);
  lambda_angle_ = declareNumber("lambda_deg", 10.0) * M_PI / 180.0;
  sigma_ = declareNumber("sigma", 0.03);
  min_2_points_dist_ = declareNumber("min_2_points_dist", 0.01);
  new_cluster_threshold_m_ = declareNumber("new_cluster_threshold_m", 0.40);
  max_viewing_distance_ = declareNumber("max_viewing_distance", 9.0);
  boundaries_inflation_ = declareNumber("boundaries_inflation", 0.10);
  detect_fov_deg_ = declareNumber("detect_fov_deg", 360.0);
  max_detect_d_m_ = declareNumber("max_detect_d_m", 0.90);

  on_track_mode_ = this->declare_parameter<std::string>("on_track_mode", "grid");
  map_name_ = this->declare_parameter<std::string>("map", "");
  filter_kernel_size_ = static_cast<int>(declareNumber("filter_kernel_size", 2));
  if (on_track_mode_ == "grid") {
    grid_ready_ = loadGridFilter();
    if (!grid_ready_) {
      RCLCPP_ERROR(
        this->get_logger(),
        "on_track_mode is 'grid' but the map could not be loaded; falling back "
        "to the frenet test. Check the `map` parameter.");
      on_track_mode_ = "frenet";
    }
  }
  RCLCPP_INFO(this->get_logger(), "on-track test: %s", on_track_mode_.c_str());

  breakpoints_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/detect/breakpoints_markers", 5);
  obstacles_msg_pub_ = this->create_publisher<f110_msgs::msg::ObstacleArray>(
    "/detect/raw_obstacles", 5);
  obstacles_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/detect/obstacles_markers_new", 5);
  latency_pub_ = this->create_publisher<std_msgs::msg::Float32>("/detect/latency", 5);

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  global_wpnts_sub_ = this->create_subscription<f110_msgs::msg::WpntArray>(
    "/global_waypoints_scaled", 10, std::bind(&Detect::pathCb, this, _1));
  scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "/scan", rclcpp::SensorDataQoS(), std::bind(&Detect::laserCb, this, _1));

  param_cb_handle_ = this->add_on_set_parameters_callback(
    std::bind(&Detect::dynParamCb, this, _1));

  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / std::max(1.0, rate_)),
    std::bind(&Detect::timerCallback, this));

  RCLCPP_INFO(
    this->get_logger(),
    "Waiting for /global_waypoints_scaled and /scan (detect rate: %.1f Hz)", rate_);
}

double Detect::declareNumber(const std::string & name, double default_value)
{
  // A yaml may write `rate_detect: 20` or `rate_detect: 20.0`. Without dynamic
  // typing the first one aborts the node with InvalidParameterType, which on
  // this stack means the whole launch dies at a stray missing decimal point.
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.dynamic_typing = true;
  this->declare_parameter(name, rclcpp::ParameterValue(default_value), descriptor);

  const rclcpp::Parameter parameter = this->get_parameter(name);
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
    return static_cast<double>(parameter.as_int());
  }
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
    return parameter.as_double();
  }
  return default_value;
}

double Detect::wrap(double value, double length)
{
  if (length <= 0.0) {
    return value;
  }
  return std::fmod(std::fmod(value, length) + length, length);
}

void Detect::laserCb(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg)
{
  scan_msg_ = msg;
}

void Detect::pathCb(const f110_msgs::msg::WpntArray::ConstSharedPtr msg)
{
  if (msg->wpnts.empty()) {
    return;
  }

  frenet_converter_.SetGlobalTrajectory(&(msg->wpnts), true);

  d_left_array_.clear();
  d_right_array_.clear();
  d_left_array_.reserve(msg->wpnts.size());
  d_right_array_.reserve(msg->wpnts.size());

  // Inflated once here rather than at every point of every scan. Both stay in
  // the waypoints' own convention: d_left is a signed offset to the left
  // boundary, d_right a positive magnitude to the right one, so the corridor
  // is (-d_right, d_left) and inflation shrinks it from both sides.
  for (const auto & wpnt : msg->wpnts) {
    d_left_array_.push_back(wpnt.d_left - boundaries_inflation_);
    d_right_array_.push_back(wpnt.d_right - boundaries_inflation_);
  }

  track_length_ = msg->wpnts.back().s_m;
  have_trajectory_ = true;
}

rcl_interfaces::msg::SetParametersResult Detect::dynParamCb(
  const std::vector<rclcpp::Parameter> & params)
{
  auto as_number = [](const rclcpp::Parameter & p) -> double {
      return p.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER
             ? static_cast<double>(p.as_int())
             : p.as_double();
    };

  for (const auto & param : params) {
    const std::string & name = param.get_name();
    if (name == "min_size_n") {
      min_size_n_ = static_cast<int>(as_number(param));
    } else if (name == "min_size_m") {
      min_size_m_ = as_number(param);
    } else if (name == "max_size_m") {
      max_size_m_ = as_number(param);
    } else if (name == "max_viewing_distance") {
      max_viewing_distance_ = as_number(param);
    } else if (name == "detect_fov_deg") {
      detect_fov_deg_ = as_number(param);
    } else if (name == "max_detect_d_m") {
      max_detect_d_m_ = as_number(param);
    } else if (name == "new_cluster_threshold_m") {
      new_cluster_threshold_m_ = as_number(param);
    } else if (name == "lambda_deg") {
      lambda_angle_ = as_number(param) * M_PI / 180.0;
    } else if (name == "sigma") {
      sigma_ = as_number(param);
    } else if (name == "min_2_points_dist") {
      min_2_points_dist_ = as_number(param);
    } else if (name == "measure") {
      measuring_ = param.as_bool();
    } else if (name == "filter_kernel_size") {
      // The only knob the grid has. Re-eroding the map costs one cv::erode
      // over the whole image, which is worth paying here so the wall/obstacle
      // trade-off can be found on the car instead of over a rebuild.
      filter_kernel_size_ = static_cast<int>(as_number(param));
      if (grid_ready_) {
        grid_filter_.setErosionKernelSize(filter_kernel_size_);
        RCLCPP_INFO(
          this->get_logger(), "erosion now %d px (%.2f m)",
          filter_kernel_size_, filter_kernel_size_ * 0.05);
      } else {
        RCLCPP_WARN(
          this->get_logger(),
          "filter_kernel_size set, but the on-track test is '%s' - no effect",
          on_track_mode_.c_str());
      }
    } else if (name == "boundaries_inflation") {
      // Only takes effect on the next /global_waypoints_scaled message: the
      // inflated bounds are precomputed in pathCb, which is a latched topic
      // in practice. Say so rather than appear to have done nothing.
      boundaries_inflation_ = as_number(param);
      RCLCPP_WARN(
        this->get_logger(),
        "boundaries_inflation applies to the precomputed track bounds; "
        "it takes effect when the global waypoints are next published");
    }
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  return result;
}

bool Detect::loadGridFilter()
{
  if (map_name_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "no `map` parameter, cannot load the grid");
    return false;
  }
  const std::string base =
    ament_index_cpp::get_package_share_directory("stack_master") + "/maps/" + map_name_ + "/" +
    map_name_;
  if (!grid_filter_.loadMapFromYAML(base + ".yaml", base + ".png")) {
    return false;
  }
  grid_filter_.setErosionKernelSize(filter_kernel_size_);
  RCLCPP_INFO(
    this->get_logger(), "grid loaded from %s.yaml, erosion %d px (%.2f m)",
    base.c_str(), filter_kernel_size_, filter_kernel_size_ * 0.05);
  return true;
}

bool Detect::laserPointOnTrack(double d, int idx) const
{
  if (idx < 0 || idx >= static_cast<int>(d_left_array_.size())) {
    return false;
  }

  // A hard cap on how far off the line anything is worth clustering.
  //
  // The track boundary is not a good proxy for "somewhere the car could
  // hit something". On map_no it reaches 1.47 m to the right around
  // s = 17.5, so the wall there is legitimately inside the mapped track and
  // the boundary test cannot reject its returns - the detector was fitting
  // boxes along it and reporting obstacles at d = -1.4 while the car drove
  // past. The planner already ignores those (trajectory_threshold is 0.6 on
  // the obstacle's centre), so they cost L-shape fits and clutter rviz
  // without ever changing a decision.
  //
  // 0.90 is that 0.6 plus half of the largest obstacle we accept, so an
  // obstacle whose centre the planner would still consider keeps all of its
  // points. Raise it if a wide track ever needs obstacles further out.
  if (std::fabs(d) > max_detect_d_m_) {
    return false;
  }

  const double left_limit = d_left_array_[idx];
  const double right_limit = -d_right_array_[idx];

  // Inflating a corridor narrower than twice the inflation turns it inside
  // out. Reject rather than accept everything.
  if (left_limit <= right_limit) {
    return false;
  }

  return d > right_limit && d < left_limit;
}

bool Detect::lookupLaserToMap(
  const sensor_msgs::msg::LaserScan & scan, tf2::Transform * transform)
{
  // Zero timeout, deliberately. A blocking lookup inside the timer of a
  // single-threaded node waits on the very /tf subscription that the same
  // executor has to service, so a one second timeout is a one second stall,
  // every tick, and the transform it eventually returns is older than the one
  // it was blocking for. Take the latest and report how old it is.
  geometry_msgs::msg::TransformStamped tf_msg;
  const std::string & source = scan.header.frame_id;

  try {
    tf_msg = tf_buffer_->lookupTransform(
      "map", source, tf2::TimePointZero, tf2::durationFromSec(0.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "no transform from %s to map: %s", source.c_str(), ex.what());
    return false;
  }

  const rclcpp::Time tf_stamp(tf_msg.header.stamp);
  const rclcpp::Time scan_stamp(scan.header.stamp);
  const double lag = (scan_stamp - tf_stamp).seconds();
  if (lag > 0.15) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "placing obstacles with a %.2f s old map->%s transform; "
      "they will sit where the car was that long ago",
      lag, source.c_str());
  }

  tf2::fromMsg(tf_msg.transform, *transform);
  sensor_origin_ = transform->getOrigin();
  return true;
}

std::vector<Cluster> Detect::clustering(
  const sensor_msgs::msg::LaserScan & scan, const tf2::Transform & laser_to_map)
{
  std::vector<Cluster> clusters;

  const double d_phi = scan.angle_increment;
  // Adaptive breakpoint detection (Borges & Aldon): the distance at which two
  // consecutive returns stop belonging to the same surface grows with range,
  // because the beams diverge. Undefined if lambda <= d_phi.
  if (lambda_angle_ <= d_phi) {
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "lambda_deg (%.2f deg) must exceed the scan's angular resolution "
      "(%.3f deg); no clustering possible",
      lambda_angle_ * 180.0 / M_PI, d_phi * 180.0 / M_PI);
    return clusters;
  }
  const double div_const = std::sin(d_phi) / std::sin(lambda_angle_ - d_phi);

  const double range_min = std::max(static_cast<double>(scan.range_min), 0.02);
  const double range_max = std::min(
    static_cast<double>(scan.range_max), max_viewing_distance_);
  const bool limit_fov = detect_fov_deg_ > 0.0 && detect_fov_deg_ < 360.0;
  const double half_fov = detect_fov_deg_ * M_PI / 360.0;

  for (size_t i = 0; i < scan.ranges.size(); i++) {
    const double range = static_cast<double>(scan.ranges[i]);
    if (!std::isfinite(range) || range < range_min || range > range_max) {
      continue;
    }

    const double angle = scan.angle_min + static_cast<double>(i) * d_phi;
    if (limit_fov && std::fabs(angle) > half_fov) {
      continue;
    }

    // Offset the beam by the sensor's own height so the point lands on z = 0
    // in the map. The scan is planar and only x/y are used, but the transform
    // is a full 3D one and any roll or pitch in the chain would otherwise leak
    // the lidar's mounting height into the ground-plane coordinates.
    const tf2::Vector3 point_map = laser_to_map * tf2::Vector3(
      range * std::cos(angle), range * std::sin(angle), -laser_to_map.getOrigin().z());

    ScanPoint point;
    point.x = point_map.x();
    point.y = point_map.y();

    // Two ways to ask "could the car be here", chosen by on_track_mode.
    //
    // grid: look the point up in the eroded map. One array lookup, no
    // waypoint, no coordinate transform - and no way to get the answer
    // wrong because the track doubles back. It also stops the track
    // boundary being used as a proxy: an obstacle against a wall is in free
    // space and reads as such, where a lateral offset from the raceline
    // cannot tell an obstacle near a wall from the wall itself.
    //
    // frenet: the offset test. Kept so the two can be compared on the same
    // track, and as the fallback if the map fails to load.
    //
    // s and d stay zero in grid mode. Nothing downstream reads them - the
    // obstacle's Frenet position and extents come from the fitted box's
    // corners, not from the points.
    if (on_track_mode_ == "grid") {
      if (!grid_filter_.isPointInside(point.x, point.y)) {
        continue;
      }
    } else {
      // FULL search, not the proximity one. The proximity search seeds
      // from the previous point's index and scans about 2 m of track,
      // falling back to a full sweep only if nothing in that window is
      // within 2 m. Where the track doubles back those two branches are
      // themselves within 2 m, so the fallback never fires and the return
      // is assigned to the wrong one - which is what made a box at the end
      // of a straight invisible while the same box mid-straight was seen,
      // and made turning the car reveal it. The grid test above has no
      // equivalent failure because it never assigns a waypoint at all.
      frenet_converter_.GetFrenetPoint(
        point.x, point.y, &point.s, &point.d, &closest_idx_, true);
      if (!std::isfinite(point.s) || !std::isfinite(point.d)) {
        continue;
      }
      if (!laserPointOnTrack(point.d, closest_idx_)) {
        continue;
      }
      point.s = wrap(point.s, track_length_);
    }

    if (clusters.empty()) {
      clusters.push_back({point});
      continue;
    }

    const double d_max = range * div_const + 3.0 * sigma_;
    const ScanPoint & previous = clusters.back().back();
    const double gap = std::hypot(point.x - previous.x, point.y - previous.y);

    if (gap < d_max) {
      clusters.back().push_back(point);
      continue;
    }

    // The scan is ordered by angle, so a break can simply mean the beam swept
    // past a nearer object and back onto one it already touched. Rejoin the
    // closest existing cluster when it is near enough, and move it to the back
    // so the next point compares against it.
    double min_distance = std::numeric_limits<double>::max();
    size_t min_index = 0;
    for (size_t j = 0; j < clusters.size(); j++) {
      const ScanPoint & tail = clusters[j].back();
      const double distance = std::hypot(point.x - tail.x, point.y - tail.y);
      if (distance < min_distance) {
        min_distance = distance;
        min_index = j;
      }
    }

    if (min_distance < new_cluster_threshold_m_) {
      Cluster reopened = std::move(clusters[min_index]);
      clusters.erase(clusters.begin() + static_cast<long>(min_index));
      reopened.push_back(point);
      clusters.push_back(std::move(reopened));
    } else {
      clusters.push_back({point});
    }
  }

  clusters.erase(
    std::remove_if(
      clusters.begin(), clusters.end(),
      [this](const Cluster & cluster) {
        return cluster.size() < static_cast<size_t>(min_size_n_);
      }),
    clusters.end());

  return clusters;
}

std::vector<Obstacle> Detect::fittingLShape(const std::vector<Cluster> & clusters)
{
  std::vector<Obstacle> obstacles;
  obstacles.reserve(clusters.size());

  const double start_angle = 0.0;
  const double end_angle = M_PI / 2.0 - M_PI / 180.0;
  const double angle_step = (end_angle - start_angle) / (kNumCandidates - 1);

  std::vector<double> candidate_angles(kNumCandidates);
  std::vector<double> candidate_cos(kNumCandidates);
  std::vector<double> candidate_sin(kNumCandidates);
  for (int j = 0; j < kNumCandidates; j++) {
    candidate_angles[j] = start_angle + j * angle_step;
    candidate_cos[j] = std::cos(candidate_angles[j]);
    candidate_sin[j] = std::sin(candidate_angles[j]);
  }

  std::vector<double> proj1;
  std::vector<double> proj2;

  for (const Cluster & cluster : clusters) {
    if (cluster.empty()) {
      continue;
    }
    const int n = static_cast<int>(cluster.size());
    proj1.resize(n);
    proj2.resize(n);

    // Score each candidate orientation by closeness-to-edge: for the correct
    // one, every point lies near one of the two sides of the rectangle, so the
    // sum of 1/distance is largest.
    double best_score = -std::numeric_limits<double>::max();
    int best_index = 0;

    for (int j = 0; j < kNumCandidates; j++) {
      const double cos_val = candidate_cos[j];
      const double sin_val = candidate_sin[j];

      double max1 = -std::numeric_limits<double>::max();
      double min1 = std::numeric_limits<double>::max();
      double max2 = -std::numeric_limits<double>::max();
      double min2 = std::numeric_limits<double>::max();

      for (int i = 0; i < n; i++) {
        proj1[i] = cluster[i].x * cos_val + cluster[i].y * sin_val;
        proj2[i] = -cluster[i].x * sin_val + cluster[i].y * cos_val;
        max1 = std::max(max1, proj1[i]);
        min1 = std::min(min1, proj1[i]);
        max2 = std::max(max2, proj2[i]);
        min2 = std::min(min2, proj2[i]);
      }

      // Along each axis, keep the side the points cling to - the one whose
      // residuals are smaller. The visible face of an object is one side of
      // the rectangle, not both.
      double norm10 = 0.0, norm11 = 0.0, norm20 = 0.0, norm21 = 0.0;
      for (int i = 0; i < n; i++) {
        const double d10 = max1 - proj1[i];
        const double d11 = proj1[i] - min1;
        const double d20 = max2 - proj2[i];
        const double d21 = proj2[i] - min2;
        norm10 += d10 * d10;
        norm11 += d11 * d11;
        norm20 += d20 * d20;
        norm21 += d21 * d21;
      }
      const bool use_min1 = norm10 > norm11;
      const bool use_min2 = norm20 > norm21;

      double score = 0.0;
      for (int i = 0; i < n; i++) {
        const double d1 = use_min1 ? (proj1[i] - min1) : (max1 - proj1[i]);
        const double d2 = use_min2 ? (proj2[i] - min2) : (max2 - proj2[i]);
        score += 1.0 / std::max(std::min(d1, d2), min_2_points_dist_);
      }

      if (score > best_score) {
        best_score = score;
        best_index = j;
      }
    }

    const double theta = candidate_angles[best_index];
    const double cos_opt = std::cos(theta);
    const double sin_opt = std::sin(theta);

    double max1 = -std::numeric_limits<double>::max();
    double min1 = std::numeric_limits<double>::max();
    double max2 = -std::numeric_limits<double>::max();
    double min2 = std::numeric_limits<double>::max();
    for (int i = 0; i < n; i++) {
      const double p1 = cluster[i].x * cos_opt + cluster[i].y * sin_opt;
      const double p2 = -cluster[i].x * sin_opt + cluster[i].y * cos_opt;
      max1 = std::max(max1, p1);
      min1 = std::min(min1, p1);
      max2 = std::max(max2, p2);
      min2 = std::min(min2, p2);
    }

    // The lidar only ever sees the near face, so the far side of the fitted
    // rectangle is a guess. Anchor on the corner closest to the sensor and
    // extend a square of the fitted size away from it.
    const double sensor_1 = sensor_origin_.x() * cos_opt + sensor_origin_.y() * sin_opt;
    const double sensor_2 = -sensor_origin_.x() * sin_opt + sensor_origin_.y() * cos_opt;

    const std::pair<double, double> corners[4] = {
      {max1, max2}, {max1, min2}, {min1, max2}, {min1, min2}};

    int closest_corner = 0;
    double min_corner_distance = std::numeric_limits<double>::max();
    for (int k = 0; k < 4; k++) {
      const double distance =
        std::hypot(corners[k].first - sensor_1, corners[k].second - sensor_2);
      if (distance < min_corner_distance) {
        min_corner_distance = distance;
        closest_corner = k;
      }
    }

    // Per axis, measured, floored at min_size_m.
    //
    // This was briefly a fixed 0.50 m cube on every obstacle, to stop the
    // measured box jittering frame to frame and dragging the avoidance apex
    // with it. The jitter is real - the lidar sees only the near faces, so
    // every frame under-estimates by a different amount as the angle changes
    // - but a fixed cube is the wrong place to answer it. 0.50 m is the
    // rule's MAXIMUM, not the size of the thing in front of the car, and
    // assuming it throws away gaps that are really there: on this track a
    // centred 0.50 m obstacle is passable at 22% of positions against 67%
    // for a 0.30 m one.
    //
    // The tracker settles it instead, in tracking_node._hold_extents: it is
    // the only stage that sees an obstacle across frames, each frame is a
    // lower bound, so the running maximum converges upward and then holds
    // still - and it holds at the obstacle's own size, capped at the rule's
    // 0.50 m. This node reports what it actually saw.
    //
    // `size = max(width, height)` as a square inflates
    // whichever dimension is smaller. The lidar sees a cone as a shallow arc,
    // wider across the beam than it is deep, so squaring it off turned a
    // 0.15 m cone into a 0.5 m obstacle - and both the spline planner and the
    // state machine subtract that width from the room either side. On a
    // corridor barely a metre across it consumed the entire gap.
    //
    const double measured_width = max1 - min1;
    const double measured_height = max2 - min2;
    if (measured_width > max_size_m_ || measured_height > max_size_m_) {
      continue;
    }

    const double width = std::max(measured_width, min_size_m_);
    const double height = std::max(measured_height, min_size_m_);

    std::pair<double, double> center = corners[closest_corner];
    center.first += (closest_corner < 2) ? -width / 2.0 : width / 2.0;
    center.second += (closest_corner % 2 == 0) ? -height / 2.0 : height / 2.0;

    Obstacle obstacle;
    obstacle.center_x = cos_opt * center.first - sin_opt * center.second;
    obstacle.center_y = sin_opt * center.first + cos_opt * center.second;
    // Kept as the bounding square's side, which is what it has always meant:
    // the marker draws a cube with it and checkObstacles rejects on it.
    obstacle.size = std::max(width, height);
    obstacle.theta = theta;

    int idx = closest_idx_;
    frenet_converter_.GetFrenetPoint(
      obstacle.center_x, obstacle.center_y, &obstacle.s_center, &obstacle.d_center,
      &idx, true);
    if (!std::isfinite(obstacle.s_center) || !std::isfinite(obstacle.d_center)) {
      continue;
    }
    obstacle.s_center = wrap(obstacle.s_center, track_length_);

    // Extents from the fitted rectangle's corners, NOT from the points.
    //
    // The points cannot give the extents directly, because they do not
    // surround the centre. The rectangle is anchored at the corner nearest the
    // sensor and extends away from it - that is how the occluded far side gets
    // modelled at all - so every measured point lies near one corner, and
    // measuring outward from the centre gave, on the same obstacle:
    //
    //   the side the points are on:  their spread PLUS the corner-to-centre
    //                                displacement, so too wide, and the
    //                                planner refused a gap that was there
    //   the side away from them:     no points at all, so the min_size_m/2
    //                                floor, so too narrow, and the car cut in
    //                                and hit the obstacle
    //
    // One error, both symptoms, on opposite sides of the same object. The
    // rectangle is already the model that accounts for occlusion; project it
    // instead. Four corners, and this also handles the rotation properly -
    // centre +/- size/2 assumed the box was aligned with the track, and a box
    // at 45 degrees is 41% wider in d than that.
    const double half_w = width / 2.0;
    const double half_h = height / 2.0;
    const std::pair<double, double> box[4] = {
      {center.first - half_w, center.second - half_h},
      {center.first - half_w, center.second + half_h},
      {center.first + half_w, center.second - half_h},
      {center.first + half_w, center.second + half_h}};

    double s_back = 0.0;
    double s_front = 0.0;
    double d_right = 0.0;
    double d_left = 0.0;
    bool corners_valid = true;

    for (const auto & corner : box) {
      const double corner_x = cos_opt * corner.first - sin_opt * corner.second;
      const double corner_y = sin_opt * corner.first + cos_opt * corner.second;

      double corner_s = 0.0;
      double corner_d = 0.0;
      int corner_idx = idx;
      frenet_converter_.GetFrenetPoint(corner_x, corner_y, &corner_s, &corner_d,
        &corner_idx, false);
      if (!std::isfinite(corner_s) || !std::isfinite(corner_d)) {
        corners_valid = false;
        break;
      }

      // Signed, so the extent survives a box straddling the s = 0 seam.
      const double ds =
        wrap(corner_s - obstacle.s_center + track_length_ / 2.0, track_length_) -
        track_length_ / 2.0;
      s_back = std::max(s_back, -ds);
      s_front = std::max(s_front, ds);
      d_right = std::max(d_right, obstacle.d_center - corner_d);
      d_left = std::max(d_left, corner_d - obstacle.d_center);
    }

    if (!corners_valid) {
      continue;
    }

    obstacle.s_back = s_back;
    obstacle.s_front = s_front;
    obstacle.d_right = d_right;
    obstacle.d_left = d_left;

    obstacles.push_back(obstacle);
  }

  return obstacles;
}

void Detect::checkObstacles(std::vector<Obstacle> * obstacles)
{
  tracked_obstacles_.clear();
  int id = 0;
  for (Obstacle & obstacle : *obstacles) {
    if (obstacle.size > max_size_m_) {
      continue;
    }
    obstacle.id = id++;
    tracked_obstacles_.push_back(obstacle);
  }
}

void Detect::publishObstaclesMessage()
{
  f110_msgs::msg::ObstacleArray message;
  message.header.stamp = current_stamp_;
  message.header.frame_id = "map";

  for (const Obstacle & obstacle : tracked_obstacles_) {
    f110_msgs::msg::Obstacle obstacle_msg;
    obstacle_msg.id = obstacle.id;
    obstacle_msg.s_start = wrap(obstacle.s_center - obstacle.s_back, track_length_);
    obstacle_msg.s_end = wrap(obstacle.s_center + obstacle.s_front, track_length_);
    obstacle_msg.s_center = obstacle.s_center;
    obstacle_msg.d_center = obstacle.d_center;
    obstacle_msg.d_left = obstacle.d_center + obstacle.d_left;
    obstacle_msg.d_right = obstacle.d_center - obstacle.d_right;
    obstacle_msg.size = obstacle.size;
    // The message carries the cartesian pose too, and the state machine's
    // _check_free_cartesian measures the recovery path against it. The python
    // detector never sets these, so that check has been computing distances to
    // the map origin - latent only because the recovery planner is not
    // running. Set them; they cost nothing here.
    obstacle_msg.x_m = obstacle.center_x;
    obstacle_msg.y_m = obstacle.center_y;
    obstacle_msg.theta = obstacle.theta;
    // Velocity and the static/dynamic verdict belong to the tracker, which
    // is the only stage that sees an obstacle across frames.
    obstacle_msg.vs = 0.0;
    obstacle_msg.vd = 0.0;
    obstacle_msg.is_static = false;
    obstacle_msg.is_visible = true;
    message.obstacles.push_back(obstacle_msg);
  }

  obstacles_msg_pub_->publish(message);
}

visualization_msgs::msg::MarkerArray Detect::clearMarkers() const
{
  // Returns an array that STARTS with the delete, for the caller to append to.
  //
  // It used to be published as its own message, immediately before the one
  // carrying the markers, which is what made the boxes
  // strobe at exactly the detection rate. RViz applies each message as it
  // arrives, so between the two there was a moment with nothing on screen, 20
  // times a second. Markers inside ONE array are applied in order, so putting
  // the delete at the front of the same message clears the previous frame and
  // draws this one atomically.
  visualization_msgs::msg::MarkerArray array;
  visualization_msgs::msg::Marker marker;
  // RViz drops a DELETEALL whose frame is empty.
  marker.header.frame_id = "map";
  marker.action = visualization_msgs::msg::Marker::DELETEALL;
  array.markers.push_back(marker);
  return array;
}

void Detect::publishBreakpoints(const std::vector<Cluster> & clusters)
{
  visualization_msgs::msg::MarkerArray array = clearMarkers();
  const double lifetime = 3.0 / std::max(1.0, rate_);

  for (size_t idx = 0; idx < clusters.size(); idx++) {
    const Cluster & cluster = clusters[idx];
    if (cluster.empty()) {
      continue;
    }

    for (int end = 0; end < 2; end++) {
      const ScanPoint & point = (end == 0) ? cluster.front() : cluster.back();
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "map";
      marker.header.stamp = current_stamp_;
      marker.id = static_cast<int>(idx) * 10 + end * 2;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.lifetime = rclcpp::Duration::from_seconds(lifetime);
      marker.scale.x = marker.scale.y = marker.scale.z = 0.1;
      marker.color.a = 0.5;
      marker.color.g = 1.0;
      marker.color.b = static_cast<float>(idx) / static_cast<float>(clusters.size());
      marker.pose.position.x = point.x;
      marker.pose.position.y = point.y;
      marker.pose.orientation.w = 1.0;
      array.markers.push_back(marker);
    }
  }

  breakpoints_markers_pub_->publish(array);
}

void Detect::publishObstaclesMarkers()
{
  visualization_msgs::msg::MarkerArray array = clearMarkers();
  const double lifetime = 3.0 / std::max(1.0, rate_);

  for (const Obstacle & obstacle : tracked_obstacles_) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_stamp_;
    marker.id = obstacle.id;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.lifetime = rclcpp::Duration::from_seconds(lifetime);
    // The per-axis extents, not a square of the longest one. `size` is the
    // bounding square and is still what checkObstacles rejects on, but a
    // marker drawn at `size` on both axes is wider than the obstacle this
    // message reports - and the whole point of debugging from rviz is that
    // the picture is the message.
    //
    // The box is drawn in the L-shape fit's own frame, which is what theta
    // orients it by. The extents published on the message are Frenet, so the
    // two are not the same rectangle when the track curves under the
    // obstacle; this draws the fit, and /tracking/static_dynamic_marker_pub
    // draws the Frenet width the planner actually divides by.
    marker.scale.x = std::max(1e-3, obstacle.s_front + obstacle.s_back);
    marker.scale.y = std::max(1e-3, obstacle.d_left + obstacle.d_right);
    marker.scale.z = obstacle.size;
    marker.color.a = 0.8;
    marker.color.r = 1.0;
    marker.pose.position.x = obstacle.center_x;
    marker.pose.position.y = obstacle.center_y;
    // On the ground rather than centred at z = 0, which would leave the
    // lower half below the map plane in RViz.
    marker.pose.position.z = obstacle.size / 2.0;
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, obstacle.theta);
    marker.pose.orientation = tf2::toMsg(q);
    array.markers.push_back(marker);
  }

  obstacles_marker_pub_->publish(array);
}

void Detect::timerCallback()
{
  if (!scan_msg_ || !have_trajectory_ || track_length_ <= 0.0) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Waiting for /global_waypoints_scaled and /scan (detect rate: %.1f Hz)", rate_);
    return;
  }

  const sensor_msgs::msg::LaserScan::ConstSharedPtr scan = scan_msg_;
  const auto started = std::chrono::steady_clock::now();

  tf2::Transform laser_to_map;
  if (!lookupLaserToMap(*scan, &laser_to_map)) {
    return;
  }

  // The scan's own stamp, not now(). The tracker differentiates position
  // between frames to get velocity, so stamping with wall time attributes the
  // motion to the wrong instant and inflates every speed by the pipeline's
  // own latency.
  current_stamp_ = rclcpp::Time(scan->header.stamp);

  const std::vector<Cluster> clusters = clustering(*scan, laser_to_map);
  publishBreakpoints(clusters);

  std::vector<Obstacle> obstacles = fittingLShape(clusters);
  checkObstacles(&obstacles);

  publishObstaclesMessage();
  publishObstaclesMarkers();

  if (measuring_) {
    std_msgs::msg::Float32 latency;
    latency.data = static_cast<float>(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    latency_pub_->publish(latency);
  }
}

}  // namespace perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<perception::Detect>());
  rclcpp::shutdown();
  return 0;
}
