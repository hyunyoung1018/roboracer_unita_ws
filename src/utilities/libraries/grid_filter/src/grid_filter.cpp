#include "grid_filter.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <iostream>

namespace grid_filter
{

GridFilter::GridFilter()
: kernel_size_(3, 3)
{
  kernel_ = cv::getStructuringElement(cv::MORPH_RECT, kernel_size_);
}

bool GridFilter::loadMapFromYAML(
  const std::string & yaml_path, const std::string & image_path)
{
  YAML::Node config;
  try {
    config = YAML::LoadFile(yaml_path);
  } catch (const std::exception & e) {
    std::cerr << "[grid_filter] cannot read " << yaml_path << ": " << e.what() << std::endl;
    return false;
  }

  origin_ = config["origin"].as<std::vector<double>>();
  resolution_ = config["resolution"].as<double>();

  // map_server's convention for negate: 0 - occupancy is (255 - p) / 255,
  // and a cell is free when that is below free_thresh. Inverted here into
  // the pixel value it corresponds to, so the per-point test is one
  // comparison. 0.196 on a trinary map puts the threshold at 205, which is
  // exactly the "unknown" grey, so unknown is NOT free.
  const double free_thresh = config["free_thresh"] ? config["free_thresh"].as<double>() : 0.196;
  const int negate = config["negate"] ? config["negate"].as<int>() : 0;
  free_pixel_ = static_cast<int>(255.0 * (1.0 - free_thresh));

  image_ = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  if (image_.empty()) {
    std::cerr << "[grid_filter] cannot read " << image_path << std::endl;
    return false;
  }
  if (negate) {
    cv::bitwise_not(image_, image_);
  }

  // Row 0 of the image is the TOP of the map, while the yaml's origin is
  // its bottom-left corner. Flipping once here lets isPointInside index
  // rows directly.
  cv::flip(image_, image_, 0);

  updateImage();
  return !eroded_image_.empty();
}

void GridFilter::setErosionKernelSize(int pixels)
{
  // 0 or negative makes an empty kernel and cv::erode aborts inside
  // normalizeAnchor. Guarded in the original too.
  pixels = std::max(1, pixels);
  kernel_size_ = cv::Size(pixels, pixels);
  kernel_ = cv::getStructuringElement(cv::MORPH_RECT, kernel_size_);
  updateImage();
}

void GridFilter::updateImage()
{
  if (image_.empty()) {
    return;
  }
  cv::erode(image_, eroded_image_, kernel_);
}

bool GridFilter::isPointInside(double x, double y) const
{
  if (eroded_image_.empty()) {
    return false;
  }

  const int px = static_cast<int>((x - origin_[0]) / resolution_);
  const int py = static_cast<int>((y - origin_[1]) / resolution_);

  if (px < 0 || py < 0 || px >= eroded_image_.cols || py >= eroded_image_.rows) {
    return false;
  }

  return eroded_image_.at<uchar>(py, px) > free_pixel_;
}

}  // namespace grid_filter
