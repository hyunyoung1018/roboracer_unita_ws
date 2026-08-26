#ifndef GRID_FILTER_HPP_
#define GRID_FILTER_HPP_

// Occupancy-grid "is this point on the track" test.
//
// It answers the question the detector actually needs - could the car be
// here - by looking the point up in the eroded map image. That is O(1) and,
// more importantly, it needs no Frenet coordinate: the alternative measures
// a lateral offset from the raceline, which requires assigning the point to
// a waypoint, and where a track doubles back that assignment can pick the
// wrong branch. See detect.cpp.
//
// Free space is decided from the map yaml's free_thresh the way map_server
// does it. Our maps are trinary, so 205 means unknown and must not read as
// free, and a test against a fixed white level breaks on any map saved
// differently.

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace grid_filter
{

class GridFilter
{
public:
  GridFilter();

  /// Load the map image and its metadata. Returns false and leaves the
  /// filter unusable if either is missing.
  bool loadMapFromYAML(const std::string & yaml_path, const std::string & image_path);

  /// Erosion in pixels. Widens the walls, so a lidar return that lands
  /// just inside one is rejected. Recomputes the eroded image.
  void setErosionKernelSize(int pixels);

  /// True when the map frame point is in free space.
  bool isPointInside(double x, double y) const;

  bool ready() const {return !eroded_image_.empty();}

private:
  void updateImage();

  cv::Mat image_;
  cv::Mat eroded_image_;
  cv::Mat kernel_;
  cv::Size kernel_size_;

  double resolution_{1.0};
  std::vector<double> origin_{0.0, 0.0, 0.0};
  /// Pixel value above which a cell counts as free, from free_thresh.
  int free_pixel_{205};
};

}  // namespace grid_filter

#endif  // GRID_FILTER_HPP_
