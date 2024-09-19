#pragma once
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav2_msgs/msg/costmap.hpp>
#include <nav2_costmap_2d/costmap_2d.hpp>
// #include <nav2_costmap_2d/costmap_2d_publisher.hpp>

namespace occupancy_grid_map_to_cost_map
{
class OccupancyGridMapToCostMap : public rclcpp::Node {
public:
  OccupancyGridMapToCostMap();

private:
  void callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg); 

private:
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr subscription_;
  rclcpp::Publisher<nav2_msgs::msg::Costmap>::SharedPtr publisher_;
};
} 