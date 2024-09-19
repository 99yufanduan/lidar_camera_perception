cmake_minimum_required(VERSION 3.8)
project(lidar_camera_align)
# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++17")
add_compile_options(-Wall -Wextra -Wpedantic) #-Wall：启用大多数的常规编译警告。-Wextra：启用一些额外的警告（在启用 -Wall 的基础上）。-Wpedantic：启用严格的 ISO C 和 ISO C++ 标准兼容性警告。

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})


add_executable(lidar_camera_align src/lidar_camera_align.cpp)
# Link libraries
target_link_libraries(lidar_camera_align ${OpenCV_LIBS} ${PCL_LIBRARIES} )
ament_target_dependencies(lidar_camera_align rclcpp  tf2 tf2_ros rclcpp pcl_conversions sensor_msgs cv_bridge) #链接ros2库,自动添加头文件和库文件路径

install(TARGETS
  lidar_camera_align 
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package() # 编译ros2使用