cmake_minimum_required(VERSION 3.14)
project(autoware_utils)



# Find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
ament_auto_find_build_dependencies()



ament_auto_add_library(autoware_utils SHARED
  src/autoware_utils.cpp
)
ament_target_dependencies(autoware_utils rclcpp) #链接ros2库,自动添加头文件和库文件路径

  # Find test dependencies
  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
  endif()

include_directories(include) #添加头文件的搜索路径.cpp中搜索.h就在这路径里面找 gcc和g++ 的标准搜索路径为 /usr/local/include /usr/include /usr/target/include /usr/lib/gcc-lib/target/version/include


if(BUILD_TESTING)
  file(GLOB_RECURSE test_files test/**/*.cpp)
  ament_add_ros_isolated_gtest(test_autoware_utils ${test_files})
  target_link_libraries(test_autoware_utils
    autoware_utils
  )
endif()

ament_auto_package()
