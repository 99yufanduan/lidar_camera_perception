#include <rclcpp/rclcpp.hpp>
#include <nav2_msgs/msg/costmap.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

class CostmapVisualizer : public rclcpp::Node
{
public:
  CostmapVisualizer() : Node("costmap_visualizer")
  {

    subscription_ = this->create_subscription<nav2_msgs::msg::Costmap>(
      "/output_costmap", 10, 
      std::bind(&CostmapVisualizer::costmap_callback, this, std::placeholders::_1));

    publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/visualization_costmap", 10);

    RCLCPP_INFO(this->get_logger(), "CostmapVisualizer node initialized");
  }

private:
  void costmap_callback(const nav2_msgs::msg::Costmap::SharedPtr msg)
  {
    auto occupancy_grid = std::make_unique<nav_msgs::msg::OccupancyGrid>();

    occupancy_grid->header = msg->header;
    occupancy_grid->info.resolution = msg->metadata.resolution;
    occupancy_grid->info.width = msg->metadata.size_x;
    occupancy_grid->info.height = msg->metadata.size_y;
    occupancy_grid->info.origin = msg->metadata.origin;

    occupancy_grid->data.resize(msg->data.size());
    for (size_t i = 0; i < msg->data.size(); ++i) {
      // Convert from 0-255 to 0-100 range, -1 for unknown
      if (msg->data[i] == 255) {
        occupancy_grid->data[i] = -1;  // Unknown
      } else {
        occupancy_grid->data[i] = static_cast<int8_t>((msg->data[i] * 100) / 254);
      }
    }

    publisher_->publish(std::move(occupancy_grid));
  }

  rclcpp::Subscription<nav2_msgs::msg::Costmap>::SharedPtr subscription_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CostmapVisualizer>());
  rclcpp::shutdown();
  return 0;
}