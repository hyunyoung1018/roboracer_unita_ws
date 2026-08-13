cmake_minimum_required(VERSION 3.8)
project(grid_filter)

# Hybrid, the same shape as frenet_conversion: the python module the team
# added is untouched and still imported as grid_filter.grid_filter, and a
# C++ library is added alongside it for the detector.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(grid_filter_lib src/grid_filter.cpp)
target_include_directories(grid_filter_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(grid_filter_lib PUBLIC ${OpenCV_INCLUDE_DIRS})
# yaml-cpp exports a namespaced target on newer releases and a bare one on
# older ones, and Humble is on the boundary. Pick whichever exists rather
# than hardcoding the guess that happens to work on this machine.
if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
  set(YAML_CPP_TARGET yaml-cpp)
endif()
target_link_libraries(grid_filter_lib ${OpenCV_LIBS} ${YAML_CPP_TARGET})

install(DIRECTORY include/ DESTINATION include)

install(TARGETS grid_filter_lib
  EXPORT grid_filter_export
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_include_directories(include)
ament_export_targets(grid_filter_export HAS_LIBRARY_TARGET)
ament_export_dependencies(OpenCV yaml-cpp)

ament_python_install_package(${PROJECT_NAME})

ament_package()
