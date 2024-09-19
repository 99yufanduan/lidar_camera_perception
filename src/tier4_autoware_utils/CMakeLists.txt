cmake_minimum_required(VERSION 3.14)
project(tier4_autoware_utils)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -Wpedantic )

# Find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
# Suppress CGAL warning
set(CGAL_DATA_DIR ".")

find_package(Boost REQUIRED)
find_package(autoware_auto_perception_msgs)
find_package(autoware_auto_planning_msgs)
find_package(tier4_debug_msgs)


ament_auto_add_library(tier4_autoware_utils SHARED
  src/tier4_autoware_utils.cpp
  src/geometry/boost_polygon_utils.cpp
  src/ros/msg_operation.cpp
)

if(BUILD_TESTING)
  find_package(ament_cmake_ros REQUIRED)

  file(GLOB_RECURSE test_files test/**/*.cpp)

  ament_add_ros_isolated_gtest(test_tier4_autoware_utils ${test_files})

  target_link_libraries(test_tier4_autoware_utils
    tier4_autoware_utils
  )
endif()

ament_auto_package()
