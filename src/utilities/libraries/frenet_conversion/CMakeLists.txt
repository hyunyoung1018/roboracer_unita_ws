cmake_minimum_required(VERSION 3.8)
project(frenet_conversion)

# This package was ament_python and is now a hybrid: the same Frenet
# conversion is available to python nodes as before, and to C++ ones as a
# linkable library. Nothing about the python module changed - it is still
# frenet_conversion.frenet_converter, imported exactly as it was.
#
# The two implementations are NOT the same algorithm. The python one fits a
# cubic spline through the waypoints and iterates a perpendicular projection
# onto it; the C++ one projects onto the waypoint polyline and reads s_m off
# the waypoints directly. On a raceline sampled every 0.10 m they agree to
# well under a centimetre, but they are not identical, so a value produced by
# one and compared against a value produced by the other carries that
# difference. See perception/src/detect.cpp, which is the first place in this
# workspace where that comparison actually happens.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(f110_msgs REQUIRED)

add_library(frenet_conversion_lib src/frenet_conversion.cc)
target_include_directories(frenet_conversion_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(frenet_conversion_lib rclcpp f110_msgs)

install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS frenet_conversion_lib
  EXPORT frenet_conversion_export
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_include_directories(include)
ament_export_targets(frenet_conversion_export HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp f110_msgs)

# The python module, installed under the same import path as before.
ament_python_install_package(${PROJECT_NAME})

# ament_python_install_package does not create console scripts, so the demo
# node that setup.py used to declare as an entry point is installed here
# instead. It is reached as `ros2 run frenet_conversion
# frenet_converter_demo_node.py` rather than `frenet_converter_demo`.
install(PROGRAMS
  ${PROJECT_NAME}/frenet_converter_demo_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
