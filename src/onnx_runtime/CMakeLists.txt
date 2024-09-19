cmake_minimum_required(VERSION 3.8)
project(onnx_runtime)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++17")
add_compile_options(-Wall -Wextra -Wpedantic) #-Wall：启用大多数的常规编译警告。-Wextra：启用一些额外的警告（在启用 -Wall 的基础上）。-Wpedantic：启用严格的 ISO C 和 ISO C++ 标准兼容性警告。


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

set(CMAKE_INSTALL_RPATH "/home/dyf/library/onnxruntime-linux-x64-1.19.2/lib")

include_directories(
   "/home/dyf/library/onnxruntime-linux-x64-1.19.2/include"
   ${OpenCV_INCLUDE_DIRS}
) 

link_directories(
    "/home/dyf/library/onnxruntime-linux-x64-1.19.2/lib"
)

#这将构建一个名为myProgram的目标可执行文件，它由3个源文件构建：src / main.cpp，src / some_file.cpp和src / another_file.cpp。
add_executable(onnx_runtime_inferencing src/onnx_runtime_inferencing)
ament_target_dependencies(onnx_runtime_inferencing rclcpp sensor_msgs cv_bridge) #链接ros2库,自动添加头文件和库文件路径
target_link_libraries(onnx_runtime_inferencing
   ${OpenCV_LIBS}
    onnxruntime
)

install(TARGETS
   onnx_runtime_inferencing
  DESTINATION lib/${PROJECT_NAME})


install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package() # 编译ros2使用

