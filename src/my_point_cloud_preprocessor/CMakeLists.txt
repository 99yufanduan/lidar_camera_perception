cmake_minimum_required(VERSION 3.14)
project(my_pointcloud_preprocessor)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -Wpedantic )

# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")

# Find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
# Suppress CGAL warning
set(CGAL_DATA_DIR ".")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(PCL REQUIRED)
find_package(CGAL REQUIRED COMPONENTS Core)
find_package(autoware_point_types REQUIRED)
find_package(lanelet2_extension REQUIRED)
find_package(tier4_pcl_extensions REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tier4_autoware_utils REQUIRED)
find_package(nav2_costmap_2d REQUIRED)

include_directories(
  include
  SYSTEM
   ${Boost_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(my_pointcloud_preprocessor src/my_pointcloud_preprocessor.cpp src/passthrough_uint16.cpp src/occupancy_grid_map.cpp)
target_include_directories(my_pointcloud_preprocessor PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

ament_target_dependencies(my_pointcloud_preprocessor 
  message_filters
  pcl_conversions
  rclcpp
  sensor_msgs
  tf2_ros
  tier4_autoware_utils
  pcl_ros
  autoware_point_types
  tf2_eigen
  nav2_costmap_2d
)

target_link_libraries(my_pointcloud_preprocessor
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)


install(TARGETS
  my_pointcloud_preprocessor  
  DESTINATION lib/${PROJECT_NAME})


ament_package()