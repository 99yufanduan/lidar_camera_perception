
cmake_minimum_required(VERSION 3.5)
project(tier4_ground_plane_utils)

# Find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(tier4_ground_plane_utils_lib SHARED
  src/ground_plane_utils.cpp
)

target_link_libraries(tier4_ground_plane_utils_lib
)

target_include_directories(tier4_ground_plane_utils_lib
  PUBLIC
  include)

ament_export_include_directories(
  include
)

ament_auto_package()
