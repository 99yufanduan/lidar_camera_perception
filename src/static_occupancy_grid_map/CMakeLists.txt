cmake_minimum_required(VERSION 3.5)
project(static_occupancy_grid_map)

# set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/ubt/navigation2/install/perception_common")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav2_util)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(perception_common REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${perception_common_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_publish test/static_map_to_point_cloud.cpp)
ament_target_dependencies(${PROJECT_NAME}_publish 
  rclcpp 
  nav_msgs 
  nav2_costmap_2d 
  nav2_util 
  sensor_msgs 
  Eigen3
  tf2_ros 
  pcl_msgs
  pcl_conversions
  perception_common
  geometry_msgs
)

target_link_libraries(${PROJECT_NAME}_publish 
  ${PCL_LIBRARIES}
  ${perception_common_LIBRARIES}
)



# add_executable(${PROJECT_NAME}_node src/occupancy_grid_map_to_cost_map_node.cpp)
# ament_target_dependencies(${PROJECT_NAME}_node rclcpp nav_msgs nav2_costmap_2d nav2_util)
# target_link_libraries(${PROJECT_NAME}_node 
#   ${PCL_LIBRARIES}
# )

add_executable(${PROJECT_NAME}_visual test/costmap_visualizer.cpp)
ament_target_dependencies(${PROJECT_NAME}_visual rclcpp nav_msgs nav2_costmap_2d nav2_util)
target_link_libraries(${PROJECT_NAME}_visual 
  ${PCL_LIBRARIES}
)

install(TARGETS
  ${PROJECT_NAME}_publish
  # ${PROJECT_NAME}_node
  ${PROJECT_NAME}_visual
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()