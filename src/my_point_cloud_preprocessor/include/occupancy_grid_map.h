#ifndef POINTCLOUD_BASED_OCCUPANCY_GRID_MAP__OCCUPANCY_GRID_MAP_H_
#define POINTCLOUD_BASED_OCCUPANCY_GRID_MAP__OCCUPANCY_GRID_MAP_H_

#include <nav2_costmap_2d/nav2_costmap_2d/costmap_2d.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

class OccupancyGridMap : public nav2_costmap_2d::Costmap2D
{
public:
    OccupancyGridMap(
        const unsigned int cells_size_x, const unsigned int cells_size_y, const float resolution, const int origin_x, const int origin_y);

    void updateWithPointCloud(
        const sensor_msgs::msg::PointCloud2 &obstacle_pointcloud,
        const geometry_msgs::msg::Pose &robot_pose);

    void setCellValue(const double wx, const double wy, const unsigned char cost);

    void raytrace(
        const double source_x, const double source_y, const double target_x, const double target_y,
        const unsigned char cost);

private:
    bool worldToMap(double wx, double wy, unsigned int &mx, unsigned int &my) const;

    rclcpp::Logger logger_{rclcpp::get_logger("pointcloud_based_occupancy_grid_map")};
};

#endif // POINTCLOUD_BASED_OCCUPANCY_GRID_MAP__OCCUPANCY_GRID_MAP_H_