cmake_minimum_required(VERSION 3.5)
project(map_server)

find_package(ament_cmake REQUIRED)
find_package(nav2_common REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_library(YAML_CPP_LIBRARIES yaml-cpp)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)

find_path(GRAPHICSMAGICKCPP_INCLUDE_DIRS
  NAMES "Magick++.h"
  PATH_SUFFIXES GraphicsMagick)

find_library(GRAPHICSMAGICKCPP_LIBRARIES
  NAMES "GraphicsMagick++" "graphicsmagick")
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/LoadMasterMap.srv"
)

include_directories(
  include
  ${JSONCPP_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${GRAPHICSMAGICKCPP_INCLUDE_DIRS}
)

set(dependencies
  rclcpp
  rclcpp_lifecycle
  rclcpp_components
  rosidl_default_runtime
  nav_msgs
  nav2_costmap_2d 
  nav2_util   
)

add_executable(master_map_executable
  src/master_map/master_map.cpp
)

ament_target_dependencies(master_map_executable
  ${dependencies}
)
rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
target_link_libraries(master_map_executable
  ${JSONCPP_LIBRARIES} 
  ${OpenCV_LIBS} 
  ${YAML_CPP_LIBRARIES} 
  ${GRAPHICSMAGICKCPP_LIBRARIES}  
  "${cpp_typesupport_target}"
)

install(TARGETS 
  master_map_executable
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
)

add_executable(map_visualization_executable
  tests/costmap_visualization.cpp
)

ament_target_dependencies(map_visualization_executable
  ${dependencies}
)

install(TARGETS 
  map_visualization_executable
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
