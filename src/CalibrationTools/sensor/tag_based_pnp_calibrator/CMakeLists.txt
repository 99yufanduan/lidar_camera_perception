
cmake_minimum_required(VERSION 3.5)
project(tag_based_pnp_calibrator)

# find_package(autoware_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(PCL REQUIRED)
find_package(image_transport REQUIRED)
find_package(image_geometry REQUIRED)
find_package(tier4_tag_utils REQUIRED)
find_package(apriltag_msgs REQUIRED)
find_package(lidartag_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tier4_calibration_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# autoware_package()

ament_export_include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  
)

include_directories(${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS}) #添加头文件的搜索路径.cpp中搜索.h就在这路径里面找 gcc和g++ 的标准搜索路径为 /usr/local/include /usr/include /usr/target/include /usr/lib/gcc-lib/target/version/include
  

ament_auto_add_executable(tag_based_pnp_calibrator
  src/brute_force_matcher.cpp
  src/calibration_estimator.cpp
  src/tag_based_pnp_calibrator.cpp
  src/tag_calibrator_visualizer.cpp
  src/main.cpp
)

link_directories(${PCL_LIBRARY_DIRS})
target_link_libraries(tag_based_pnp_calibrator
${PCL_LIBRARIES}
)
ament_target_dependencies(tag_based_pnp_calibrator rclcpp image_transport  tier4_tag_utils image_geometry apriltag_msgs lidartag_msgs tf2 visualization_msgs tf2_eigen tf2_geometry_msgs tier4_calibration_msgs cv_bridge)

target_link_libraries(tag_based_pnp_calibrator
  ${OpenCV_LIBS}
)

ament_auto_package(
  INSTALL_TO_SHARE
    launch
    rviz
)

