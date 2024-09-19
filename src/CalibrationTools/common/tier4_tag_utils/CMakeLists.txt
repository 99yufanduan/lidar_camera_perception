
cmake_minimum_required(VERSION 3.5)
project(tier4_tag_utils)

# find_package(autoware_cmake REQUIRED)
find_package(OpenCV REQUIRED)

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp_components REQUIRED)

find_package(rclcpp REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(image_geometry REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(apriltag_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(lidartag_msgs REQUIRED)




# autoware_package()

include_directories(${EIGEN3_INCLUDE_DIRS} ) #添加头文件的搜索路径.cpp中搜索.h就在这路径里面找 gcc和g++ 的标准搜索路径为 /usr/local/include /usr/include /usr/target/include /usr/lib/gcc-lib/target/version/include

ament_auto_add_library(tier4_tag_utils_lib SHARED
  src/lidartag_hypothesis.cpp
  src/apriltag_hypothesis.cpp
)
ament_target_dependencies(tier4_tag_utils_lib lidartag_msgs) #链接ros2库,自动添加头文件和库文件路径



target_link_libraries(tier4_tag_utils_lib
${OpenCV_LIBS}
)

target_include_directories(tier4_tag_utils_lib
  PUBLIC
  include
  ${OpenCV_INCLUDE_DIRS})

  ament_export_include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
  )

ament_target_dependencies(tier4_tag_utils_lib rclcpp image_geometry tf2_eigen apriltag_msgs tf2_geometry_msgs)

ament_auto_add_library(apriltag_filter SHARED
  src/apriltag_filter.cpp
)

ament_auto_add_library(lidartag_filter SHARED
  src/lidartag_filter.cpp
)

target_link_libraries(apriltag_filter
  ${OpenCV_LIBS}
  tier4_tag_utils_lib
rclcpp_components::component_manager
)

target_link_libraries(lidartag_filter
  ${OpenCV_LIBS}
  tier4_tag_utils_lib
rclcpp_components::component_manager
)

rclcpp_components_register_node(apriltag_filter
  PLUGIN "tier4_tag_utils::ApriltagFilter"
  EXECUTABLE apriltag_filter_node
)

rclcpp_components_register_node(lidartag_filter
  PLUGIN "tier4_tag_utils::LidartagFilter"
  EXECUTABLE lidartag_filter_node
)

ament_auto_package(
  INSTALL_TO_SHARE
    launch
)
