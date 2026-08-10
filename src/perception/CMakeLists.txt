cmake_minimum_required(VERSION 3.8)
project(perception)

# This package was ament_python. It is now ament_cmake so that it can carry a
# C++ detector, and ament_cmake_python so the two python nodes it already had
# keep working unchanged.
#
# BOTH detectors are installed. `detect` is still the python one, so nothing
# that launches this package changes behaviour by being rebuilt; the C++ one
# is `detect_cpp` and has to be asked for. They publish the same topic, so
# only one may run at a time - time_trials.launch.xml takes a `detector` arg.
# Keeping them side by side is the point of this branch: the two disagree
# about which lidar returns belong to an obstacle at all (see detect.hpp), and
# that difference is only measurable by running them against the same track.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(f110_msgs REQUIRED)
find_package(frenet_conversion REQUIRED)

add_executable(detect_cpp src/detect.cpp)
target_include_directories(detect_cpp PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(detect_cpp
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  sensor_msgs
  nav_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
  f110_msgs
  frenet_conversion
)
target_link_libraries(detect_cpp frenet_conversion::frenet_conversion_lib)

install(TARGETS detect_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# The python package, imported as `perception.detect_node` exactly as before.
ament_python_install_package(${PROJECT_NAME})

# ament_python_install_package does not create console scripts, so the two
# entry points setup.py used to declare are installed as programs. The names
# on disk are unchanged, so `ros2 run perception detect` and the launch file's
# exec="detect" both still resolve.
install(PROGRAMS
  scripts/detect
  scripts/tracking_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
