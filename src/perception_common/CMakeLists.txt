cmake_minimum_required(VERSION 3.5)
project(perception_common)

find_package(ament_cmake REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(PCL REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

add_library(perception_common SHARED
  src/perception_common.cpp
)

ament_target_dependencies(perception_common
  nav_msgs
  rclcpp
  pcl_msgs
  pcl_conversions
)

target_link_libraries(perception_common
  ${PCL_LIBRARIES}
)

install(DIRECTORY include/
  DESTINATION include)

install(
  TARGETS perception_common
  EXPORT export_perception_common
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)



ament_export_include_directories(include)
ament_export_libraries(perception_common)
ament_export_dependencies(nav_msgs rclcpp pcl_conversions pcl_msgs)
ament_export_targets(export_perception_common HAS_LIBRARY_TARGET)
ament_package()