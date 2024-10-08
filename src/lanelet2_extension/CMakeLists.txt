cmake_minimum_required(VERSION 3.14)
project(lanelet2_extension)

# Find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

  # Find test dependencies
  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
  endif()

find_package(autoware_planning_msgs)
find_package(autoware_utils)
find_package(PkgConfig)
find_path(GeographicLib_INCLUDE_DIR GeographicLib/Config.h
  PATH_SUFFIXES GeographicLib
)
set(GeographicLib_INCLUDE_DIRS ${GeographicLib_INCLUDE_DIR})

find_library(GeographicLib_LIBRARIES
  NAMES Geographic
)

find_library(PUGIXML_LIBRARIES
  NAMES pugixml
)

find_path(PUGIXML_INCLUDE_DIRS
  NAMES pugixml.hpp
  PATH_SUFFIXES pugixml
)

include_directories(
  ${GeographicLib_INCLUDE_DIRS}
  ${PUGIXML_INCLUDE_DIRS}
)

add_definitions(${GeographicLib_DEFINITIONS})

ament_auto_add_library(lanelet2_extension_lib SHARED
  lib/autoware_osm_parser.cpp
  lib/autoware_traffic_light.cpp
  lib/detection_area.cpp
  lib/no_stopping_area.cpp
  lib/message_conversion.cpp
  lib/mgrs_projector.cpp
  lib/query.cpp
  lib/road_marking.cpp
  lib/speed_bump.cpp
  lib/utilities.cpp
  lib/virtual_traffic_light.cpp
  lib/visualization.cpp
  lib/route_checker.cpp
)
target_link_libraries(lanelet2_extension_lib
  ${GeographicLib_LIBRARIES}
)

ament_auto_add_executable(lanelet2_extension_sample src/sample_code.cpp)
add_dependencies(lanelet2_extension_sample lanelet2_extension_lib)
target_link_libraries(lanelet2_extension_sample
  lanelet2_extension_lib
)

ament_auto_add_executable(autoware_lanelet2_validation src/validation.cpp)
add_dependencies(autoware_lanelet2_validation lanelet2_extension_lib)
target_link_libraries(autoware_lanelet2_validation
  ${catkin_LIBRARIES}
  ${PUGIXML_LIBRARIES}
  lanelet2_extension_lib
)

ament_auto_add_executable(check_right_of_way src/check_right_of_way.cpp)
add_dependencies(check_right_of_way lanelet2_extension_lib)
target_link_libraries(check_right_of_way
  ${catkin_LIBRARIES}
  ${PUGIXML_LIBRARIES}
  lanelet2_extension_lib
)

if(BUILD_TESTING)
  ament_add_ros_isolated_gtest(message_conversion-test test/src/test_message_conversion.cpp)
  target_link_libraries(message_conversion-test lanelet2_extension_lib)
  ament_add_ros_isolated_gtest(projector-test test/src/test_projector.cpp)
  target_link_libraries(projector-test lanelet2_extension_lib)
  ament_add_ros_isolated_gtest(query-test test/src/test_query.cpp)
  target_link_libraries(query-test lanelet2_extension_lib)
  ament_add_ros_isolated_gtest(regulatory_elements-test test/src/test_regulatory_elements.cpp)
  target_link_libraries(regulatory_elements-test lanelet2_extension_lib)
  ament_add_ros_isolated_gtest(utilities-test test/src/test_utilities.cpp)
  target_link_libraries(utilities-test lanelet2_extension_lib)
  ament_add_ros_isolated_gtest(route-test test/src/test_route_checker.cpp)
  target_link_libraries(route-test lanelet2_extension_lib)
endif()

ament_auto_package()
