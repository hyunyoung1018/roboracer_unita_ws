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
)
target_link_libraries(detect frenet_conversion::frenet_conversion_lib)

install(TARGETS detect
  DESTINATION lib/${PROJECT_NAME}
)

# The python package, imported as `perception.tracking_node` as before.
ament_python_install_package(${PROJECT_NAME})

# ament_python_install_package does not create console scripts, so the entry
# point setup.py used to declare is installed as a program under the same
# name, and `ros2 run perception tracking_node` still resolves.
install(PROGRAMS
  scripts/tracking_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
