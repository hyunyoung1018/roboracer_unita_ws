cmake_minimum_required(VERSION 3.8)
project(perception)

# This package was ament_python. It is now ament_cmake for the C++ detector,
# and ament_cmake_python because the tracker is still python.
#
# `detect` is the C++ node. The python detector it replaced is in the history
# up to the merge of the perception-cpp branch, where both were installed and
# selectable so the two could be compared on the same track; the C++ one was
# driven and kept.

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
find_package(grid_filter REQUIRED)
find_package(ament_index_cpp REQUIRED)

add_executable(detect src/detect.cpp)
target_include_directories(detect PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(detect
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
  grid_filter
  ament_index_cpp
)
target_link_libraries(detect
  frenet_conversion::frenet_conversion_lib
  grid_filter::grid_filter_lib
)

install(TARGETS detect
  DESTINATION lib/${PROJECT_NAME}
)

# The python package, imported as `perception.tracking_node` as before.
ament_python_install_package(${PROJECT_NAME})

# ament_python_install_package does not create console scripts, so the entry
# point setup.py used to declare is installed as a program under the same
# name, and `ros2 run perception tracking_node` still resolves.
#
# h2h_tracking_node is the head-to-head subclass. It is a separate executable
# so that time_trials keeps running the unmodified tracking_node.
install(PROGRAMS
  scripts/tracking_node
  scripts/h2h_tracking_node
  DESTINATION lib/${PROJECT_NAME}
)

# The pure-python opponent selection gates. ament_cmake does not collect
# test/ on its own, unlike the ament_python packages in this workspace.
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(h2h_opponent_selector test/test_h2h_opponent_selector.py)
  ament_add_pytest_test(h2h_classification test/test_h2h_classification.py)
endif()

ament_package()
