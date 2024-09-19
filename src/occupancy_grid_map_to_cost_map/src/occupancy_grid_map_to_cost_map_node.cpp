#include "occupancy_grid_map_to_cost_map_node.h"
#include "nav2_util/occ_grid_values.hpp"
#include "nav2_costmap_2d/cost_values.hpp"

namespace occupancy_grid_map_to_cost_map
{
OccupancyGridMapToCostMap::OccupancyGridMapToCostMap()
: Node("occupancy_grid_map_to_cost_map")
{ 
  auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(1));

  subscription_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/occupancy_grid", qos_profile, 
      std::bind(&OccupancyGridMapToCostMap::callback, this, std::placeholders::_1));

  auto qos = rclcpp::QoS(rclcpp::KeepLast(10)); 
  qos.reliable();  
  publisher_ = this->create_publisher<nav2_msgs::msg::Costmap>("/output_costmap", qos);
}

void OccupancyGridMapToCostMap::callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
    nav2_costmap_2d::Costmap2D costmap(*msg);

    auto costmap_msg = std::make_unique<nav2_msgs::msg::Costmap>();
    costmap_msg->header = msg->header;
    costmap_msg->metadata.size_x = costmap.getSizeInCellsX();
    costmap_msg->metadata.size_y = costmap.getSizeInCellsY();
    costmap_msg->metadata.resolution = costmap.getResolution();
    costmap_msg->metadata.layer = "master";
    costmap_msg->metadata.update_time = this->now();
    costmap_msg->metadata.origin = msg->info.origin;

    costmap_msg->data.resize(costmap.getSizeInCellsX() * costmap.getSizeInCellsY());
    unsigned char* char_map = costmap.getCharMap();
    std::copy(char_map, char_map + costmap_msg->data.size(), costmap_msg->data.begin());

    publisher_->publish(std::move(costmap_msg));
} 
}  // namespace occupancy_grid_map_to_cost_map

using namespace occupancy_grid_map_to_cost_map;
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OccupancyGridMapToCostMap>());
  rclcpp::shutdown();
  return 0;
}