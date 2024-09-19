
#include <tf2_ros/buffer.h>
#include <Eigen/Core>
#include <tf2_eigen/tf2_eigen.hpp>
#include "occupancy_grid_map.h"
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>
#include <nav2_costmap_2d/nav2_costmap_2d/costmap_2d.hpp>
#include <pcl_ros/transforms.hpp>

#include <sensor_msgs/point_cloud2_iterator.hpp>

const unsigned int NO_INFORMATION = 128; // 0.5 * 255
const unsigned int LETHAL_OBSTACLE = 255;
const unsigned int FREE_SPACE = 0;

void transformPointcloud(
    const sensor_msgs::msg::PointCloud2 &input, const geometry_msgs::msg::Pose &pose,
    sensor_msgs::msg::PointCloud2 &output)
{
    const auto transform = tier4_autoware_utils::pose2transform(pose);
    Eigen::Matrix4f tf_matrix = tf2::transformToEigen(transform).matrix().cast<float>();

    pcl_ros::transformPointCloud(tf_matrix, input, output);
    output.header.stamp = input.header.stamp;
    output.header.frame_id = "";
}

OccupancyGridMap::OccupancyGridMap(
    const unsigned int cells_size_x, const unsigned int cells_size_y, const float resolution, const int origin_x, const int origin_y)
    : Costmap2D(cells_size_x, cells_size_y, resolution, origin_x, origin_y, NO_INFORMATION)
{
}

bool OccupancyGridMap::worldToMap(double wx, double wy, unsigned int &mx, unsigned int &my) const
{
    if (wx < origin_x_ || wy < origin_y_)
    {
        return false;
    }

    mx = static_cast<int>(std::floor((wx - origin_x_) / resolution_));
    my = static_cast<int>(std::floor((wy - origin_y_) / resolution_));

    if (mx < size_x_ && my < size_y_)
    {
        return true;
    }

    return false;
}

void OccupancyGridMap::raytrace(
    const double source_x, const double source_y, const double target_x, const double target_y,
    const unsigned char cost)
{
    unsigned int x0{};
    unsigned int y0{};
    const double ox{source_x};
    const double oy{source_y};
    if (!worldToMap(ox, oy, x0, y0))
    {
        RCLCPP_DEBUG(
            logger_,
            "The origin for the sensor at (%.2f, %.2f) is out of map bounds. So, the costmap cannot "
            "raytrace for it.",
            ox, oy);
        return;
    }

    // we can pre-compute the endpoints of the map outside of the inner loop... we'll need these later
    const double origin_x = origin_x_, origin_y = origin_y_;
    const double map_end_x = origin_x + size_x_ * resolution_;
    const double map_end_y = origin_y + size_y_ * resolution_;

    double wx = target_x;
    double wy = target_y;

    // now we also need to make sure that the endpoint we're ray-tracing
    // to isn't off the costmap and scale if necessary
    const double a = wx - ox;
    const double b = wy - oy;

    // the minimum value to raytrace from is the origin
    if (wx < origin_x)
    {
        const double t = (origin_x - ox) / a;
        wx = origin_x;
        wy = oy + b * t;
    }
    if (wy < origin_y)
    {
        const double t = (origin_y - oy) / b;
        wx = ox + a * t;
        wy = origin_y;
    }

    // the maximum value to raytrace to is the end of the map
    if (wx > map_end_x)
    {
        const double t = (map_end_x - ox) / a;
        wx = map_end_x - .001;
        wy = oy + b * t;
    }
    if (wy > map_end_y)
    {
        const double t = (map_end_y - oy) / b;
        wx = ox + a * t;
        wy = map_end_y - .001;
    }

    // now that the vector is scaled correctly... we'll get the map coordinates of its endpoint
    unsigned int x1{};
    unsigned int y1{};

    // check for legality just in case
    if (!worldToMap(wx, wy, x1, y1))
    {
        return;
    }

    constexpr unsigned int cell_raytrace_range = 10000; // large number to ignore range threshold
    MarkCell marker(costmap_, cost);
    raytraceLine(marker, x0, y0, x1, y1, cell_raytrace_range);
}

void OccupancyGridMap::setCellValue(const double wx, const double wy, const unsigned char cost)
{
    MarkCell marker(costmap_, cost);
    unsigned int mx{};
    unsigned int my{};
    if (!worldToMap(wx, wy, mx, my))
    {
        RCLCPP_DEBUG(logger_, "Computing map coords failed");
        return;
    }
    const unsigned int index = getIndex(mx, my);
    marker(index);
}

void OccupancyGridMap::updateWithPointCloud(
    const sensor_msgs::msg::PointCloud2 &obstacle_pointcloud,
    const geometry_msgs::msg::Pose &robot_pose)
{
    constexpr double min_angle = -M_PI;
    constexpr double max_angle = M_PI;
    constexpr double angle_increment = 0.001745; // 0.1 度
    const size_t angle_bin_size = ((max_angle - min_angle) / angle_increment) + size_t(1 /*margin*/);

    // Transform to map frame
    sensor_msgs::msg::PointCloud2 trans_obstacle_pointcloud;
    transformPointcloud(obstacle_pointcloud, robot_pose, trans_obstacle_pointcloud);

    // Create angle bins
    struct BinInfo
    {
        BinInfo() = default;
        BinInfo(const double _range, const double _wx, const double _wy)
            : range(_range), wx(_wx), wy(_wy)
        {
        }
        double range;
        double wx;
        double wy;
    };
    std::vector</*angle bin*/ std::vector<BinInfo>> obstacle_pointcloud_angle_bins;
    obstacle_pointcloud_angle_bins.resize(angle_bin_size);
    for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(obstacle_pointcloud, "x"),
         iter_y(obstacle_pointcloud, "y"), iter_wx(trans_obstacle_pointcloud, "x"),
         iter_wy(trans_obstacle_pointcloud, "y");
         iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_wx, ++iter_wy)
    {
        const double angle = atan2(*iter_y, *iter_x);
        int angle_bin_index = (angle - min_angle) / angle_increment;
        obstacle_pointcloud_angle_bins.at(angle_bin_index)
            .push_back(BinInfo(std::hypot(*iter_y, *iter_x), *iter_wx, *iter_wy));
    }

    // Sort by distance
    for (auto &obstacle_pointcloud_angle_bin : obstacle_pointcloud_angle_bins)
    {
        std::sort(
            obstacle_pointcloud_angle_bin.begin(), obstacle_pointcloud_angle_bin.end(),
            [](auto a, auto b)
            { return a.range < b.range; });
    }

    // First step: Initialize cells to the final point with freespace
    constexpr double distance_margin = 1.0;
    for (size_t bin_index = 0; bin_index < obstacle_pointcloud_angle_bins.size(); ++bin_index)
    {
        auto &obstacle_pointcloud_angle_bin = obstacle_pointcloud_angle_bins.at(bin_index);

        BinInfo end_distance;
        if (obstacle_pointcloud_angle_bin.empty())
        {
            continue;
        }
        else
        {
            end_distance = obstacle_pointcloud_angle_bin.back();
        }
        raytrace(
            robot_pose.position.x, robot_pose.position.y, end_distance.wx, end_distance.wy,
            FREE_SPACE);
    }

    // Second step: Overwrite occupied cell
    for (size_t bin_index = 0; bin_index < obstacle_pointcloud_angle_bins.size(); ++bin_index)
    {
        auto &obstacle_pointcloud_angle_bin = obstacle_pointcloud_angle_bins.at(bin_index);
        for (size_t dist_index = 0; dist_index < obstacle_pointcloud_angle_bin.size(); ++dist_index)
        {
            const auto &source = obstacle_pointcloud_angle_bin.at(dist_index);
            setCellValue(source.wx, source.wy, LETHAL_OBSTACLE);

            if (dist_index + 1 == obstacle_pointcloud_angle_bin.size())
            {
                continue;
            }

            auto next_obstacle_point_distance = std::abs(
                obstacle_pointcloud_angle_bin.at(dist_index + 1).range -
                obstacle_pointcloud_angle_bin.at(dist_index).range);
            if (next_obstacle_point_distance <= distance_margin)
            {
                const auto &source = obstacle_pointcloud_angle_bin.at(dist_index);
                const auto &target = obstacle_pointcloud_angle_bin.at(dist_index + 1);
                raytrace(source.wx, source.wy, target.wx, target.wy, LETHAL_OBSTACLE);
                continue;
            }
        }
    }
}
