cmake_minimum_required(VERSION 3.8)
project(my_calibration_tools)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++17")
add_compile_options(-Wall -Wextra -Wpedantic) #-Wall：启用大多数的常规编译警告。-Wextra：启用一些额外的警告（在启用 -Wall 的基础上）。-Wpedantic：启用严格的 ISO C 和 ISO C++ 标准兼容性警告。

# find dependencies

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

add_executable(point_cloud_registration_integration src/point_cloud_registration_integration.cpp)
ament_target_dependencies(point_cloud_registration_integration rclcpp sensor_msgs pcl_conversions cv_bridge)

target_link_libraries(point_cloud_registration_integration
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(pnp src/pnp.cpp)
ament_target_dependencies(pnp rclcpp)
target_link_libraries(pnp
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
)

add_executable(3d_lidar_base_link_calobration src/3d_lidar_base_link_calobration.cpp)
ament_target_dependencies(3d_lidar_base_link_calobration sensor_msgs rclcpp pcl_conversions)
target_link_libraries(3d_lidar_base_link_calobration
  ${PCL_LIBRARIES}
)

install(TARGETS
  pnp  
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS
  3d_lidar_base_link_calobration  
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS
  point_cloud_registration_integration  
  DESTINATION lib/${PROJECT_NAME})

ament_package()