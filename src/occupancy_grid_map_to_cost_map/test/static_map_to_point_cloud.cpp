#include <rclcpp/rclcpp.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <nav2_msgs/msg/costmap.hpp>
#include <nav2_costmap_2d/costmap_2d.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include "nav2_util/occ_grid_values.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/pcl_base.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/centroid.h>

#include "pointcloud_common.h"

class PCDToOccupancyGrid : public rclcpp::Node
{
public:
    PCDToOccupancyGrid() : Node("pcd_to_occupancy_grid")
    {
        this->declare_parameter("mean_k", 50);
        this->declare_parameter("stddev_mul_thresh", 1.0);
        this->declare_parameter("radius_search", 0.2);
        this->declare_parameter("min_neighbors_inRadius", 50);
        this->declare_parameter("height_threshold", -1.0);
        this->declare_parameter("ground_height_threshold", 0.09);

        this->get_parameter("mean_k", mean_k_);
        this->get_parameter("stddev_mul_thresh", stddev_mul_thresh_);
        this->get_parameter("radius_search", radius_search_);
        this->get_parameter("min_neighbors_inRadius", min_neighbors_inRadius_);
        this->get_parameter("height_threshold", height_threshold_);
        this->get_parameter("ground_height_threshold", ground_height_threshold_);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);
        std::string pcd_file = "/home/dyf/project/rosbag/rosbag2_2024_09_19-17_24_25/map.pcd";
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud) == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Couldn't read file %s", pcd_file.c_str());
            return;
        }
        static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

        RCLCPP_INFO(this->get_logger(), "Loaded PCD file with %ld points", cloud->points.size());

        publish_static_transform();

        auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
        qos.reliable();
        costmap_publisher_ = this->create_publisher<nav2_msgs::msg::Costmap>("/output_costmap", qos);
        pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/output_pointcloud", qos);

        generateOccupancyGrid(cloud);

        double low = height_threshold_;
        double high = 0;

        passThroughFilter(cloud, cloud_filtered, low, high);
        // voxelGridFilter(cloud_filtered, cloud_filtered);

        low = height_threshold_;
        high = -1.27;
        passThroughFilter(cloud, cloud_ground, low, high);
        RadiusOutlierFilter(cloud_ground, cloud_ground);
        statisticalOutlierRemovalFilter(cloud_ground, cloud_ground);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        // applyRANSAC(cloud_filtered, cloud_filtered, coefficients);
        applyRANSAC(cloud_ground, inliers, coefficients);

        if (coefficients->values.empty())
        {
            RCLCPP_INFO(this->get_logger(), "Failed to find a plane");
        }

        Eigen::Vector3d plane_normal(
            coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        {
            const auto plane_slope = std::abs(
                std::acos(plane_normal.dot(unit_vec_) / (plane_normal.norm() * unit_vec_.norm())) * 180 /
                M_PI);
            if (plane_slope > 10.0)
            {
                RCLCPP_INFO(this->get_logger(), "Plane slope is too high");
            }
        }

        // extract pointcloud from indices
        pcl::PointCloud<pcl::PointXYZ>::Ptr segment_ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr segment_no_ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

        extractPointsIndices(
            cloud_ground, *inliers, segment_ground_cloud_ptr, segment_no_ground_cloud_ptr);
        const Eigen::Affine3d plane_affine = getPlaneAffine(*segment_ground_cloud_ptr, plane_normal);
        pcl::PointCloud<pcl::PointXYZ>::Ptr no_ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto &p : cloud_filtered->points)
        {
            const Eigen::Vector3d transformed_point =
                plane_affine.inverse() * Eigen::Vector3d(p.x, p.y, p.z);
            if (std::abs(transformed_point.z()) > ground_height_threshold_) // height_threshold
            {
                no_ground_cloud_ptr->points.push_back(p);
            }
        }

        RadiusOutlierFilter(no_ground_cloud_ptr, no_ground_cloud_ptr);
        statisticalOutlierRemovalFilter(no_ground_cloud_ptr, no_ground_cloud_ptr);
        voxelGridFilter(no_ground_cloud_ptr, no_ground_cloud_ptr, 0.02);

        updateOccupancyGrid(no_ground_cloud_ptr);
        generatePointCloud2(no_ground_cloud_ptr);
        publishData();

        // timer_ = this->create_wall_timer(
        //     std::chrono::milliseconds(500),
        //     std::bind(&PCDToOccupancyGrid::publishData, this));
    }

private:
    void publish_static_transform()
    {
        geometry_msgs::msg::TransformStamped static_transform;

        static_transform.header.stamp = this->get_clock()->now();
        static_transform.header.frame_id = "odom";     // 父坐标系
        static_transform.child_frame_id = "base_link"; // 子坐标系

        static_transform.transform.translation.x = 0.0;
        static_transform.transform.translation.y = 0.0;
        static_transform.transform.translation.z = 0.0;

        static_transform.transform.rotation.x = 0.0;
        static_transform.transform.rotation.y = 0.0;
        static_transform.transform.rotation.z = 0.0;
        static_transform.transform.rotation.w = 1.0;

        static_broadcaster_->sendTransform(static_transform);
    }

    void passThroughFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
                           double low, double high)
    {
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(input_cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(low, high);
        pass.filter(*filtered_cloud);
    }

    void voxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
                         float leaf_size = 0.05)
    {
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(input_cloud);
        sor.setLeafSize(leaf_size, leaf_size, leaf_size);
        sor.filter(*filtered_cloud);
    }

    void applyRANSAC(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, pcl::PointIndices::Ptr &filtered_cloud,
        pcl::ModelCoefficients::Ptr &output_coefficients)
    {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setRadiusLimits(0.3, std::numeric_limits<double>::max());
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(input_cloud);
        seg.setMaxIterations(1000);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.segment(*filtered_cloud, *output_coefficients);
    }

    struct PlaneBasis
    {
        Eigen::Vector3d e_x;
        Eigen::Vector3d e_y;
        Eigen::Vector3d e_z;
    };

    Eigen::Vector3d getArbitraryOrthogonalVector(const Eigen::Vector3d &input)
    {
        const double x = input.x();
        const double y = input.y();
        const double z = input.z();
        const double x2 = std::pow(x, 2);
        const double y2 = std::pow(y, 2);
        const double z2 = std::pow(z, 2);

        Eigen::Vector3d unit_vec{0, 0, 0};
        if (x2 <= y2 && x2 <= z2)
        {
            unit_vec.x() = 0;
            unit_vec.y() = z;
            unit_vec.z() = -y;
            unit_vec = unit_vec / std::sqrt(y2 + z2);
        }
        else if (y2 <= x2 && y2 <= z2)
        {
            unit_vec.x() = -z;
            unit_vec.y() = 0;
            unit_vec.z() = x;
            unit_vec = unit_vec / std::sqrt(z2 + x2);
        }
        else if (z2 <= x2 && z2 <= y2)
        {
            unit_vec.x() = y;
            unit_vec.y() = -x;
            unit_vec.z() = 0;
            unit_vec = unit_vec / std::sqrt(x2 + y2);
        }
        return unit_vec;
    }

    PlaneBasis getPlaneBasis(const Eigen::Vector3d &plane_normal)
    {
        PlaneBasis basis;
        basis.e_z = plane_normal;
        basis.e_x = getArbitraryOrthogonalVector(plane_normal);
        basis.e_y = basis.e_x.cross(basis.e_z);
        return basis;
    }

    Eigen::Affine3d getPlaneAffine(
        const pcl::PointCloud<pcl::PointXYZ> segment_ground_cloud, const Eigen::Vector3d &plane_normal)
    {
        pcl::CentroidPoint<pcl::PointXYZ> centroid;
        for (const auto p : segment_ground_cloud.points)
        {
            centroid.add(p);
        }
        pcl::PointXYZ centroid_point;
        centroid.get(centroid_point);
        Eigen::Translation<double, 3> trans(centroid_point.x, centroid_point.y, centroid_point.z);
        const PlaneBasis basis = getPlaneBasis(plane_normal);
        Eigen::Matrix3d rot;
        rot << basis.e_x.x(), basis.e_y.x(), basis.e_z.x(), basis.e_x.y(), basis.e_y.y(), basis.e_z.y(),
            basis.e_x.z(), basis.e_y.z(), basis.e_z.z();
        return trans * rot;
    }

    void extractPointsIndices(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, const pcl::PointIndices &in_indices,
        pcl::PointCloud<pcl::PointXYZ>::Ptr out_only_indices_cloud_ptr,
        pcl::PointCloud<pcl::PointXYZ>::Ptr out_removed_indices_cloud_ptr)
    {
        pcl::ExtractIndices<pcl::PointXYZ> extract_ground;
        extract_ground.setInputCloud(in_cloud_ptr);
        extract_ground.setIndices(pcl::make_shared<pcl::PointIndices>(in_indices));

        extract_ground.setNegative(false); // true removes the indices, false leaves only the indices
        extract_ground.filter(*out_only_indices_cloud_ptr);

        extract_ground.setNegative(true); // true removes the indices, false leaves only the indices
        extract_ground.filter(*out_removed_indices_cloud_ptr);
    }

    void RadiusOutlierFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud)
    {

        pcl::RadiusOutlierRemoval<pcl::PointXYZ> radiusoutlier;

        radiusoutlier.setInputCloud(input_cloud);

        radiusoutlier.setRadiusSearch(radius_search_);
        radiusoutlier.setMinNeighborsInRadius(min_neighbors_inRadius_);
        radiusoutlier.filter(*filtered_cloud);
    }

    void statisticalOutlierRemovalFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                                         pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud)
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(input_cloud);
        sor.setMeanK(mean_k_);                      // 邻域点的数量
        sor.setStddevMulThresh(stddev_mul_thresh_); // 离群点的标准差阈值
        sor.filter(*filtered_cloud);
    }

    void generateOccupancyGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        statisticalOutlierRemovalFilter(cloud, filtered_cloud);
        // voxelGridFilter(cloud, filtered_cloud, 0.03);

        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::lowest();

        for (const auto &point : cloud->points)
        {
            if (point.x < min_x)
                min_x = point.x;
            if (point.x > max_x)
                max_x = point.x;
            if (point.y < min_y)
                min_y = point.y;
            if (point.y > max_y)
                max_y = point.y;
        }

        resolution_ = 0.05;

        width_ = static_cast<int>((max_x - min_x) / resolution_);
        height_ = static_cast<int>((max_y - min_y) / resolution_);

        RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "17:03:20: " << width_);
        RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "17:03:40: " << height_);
        RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "17:03:50: " << min_x);
        RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "17:04:03: " << min_y);

        occupancy_grid_.info.resolution = resolution_;
        occupancy_grid_.info.width = width_;
        occupancy_grid_.info.height = height_;
        occupancy_grid_.info.origin.position.x = min_x;
        occupancy_grid_.info.origin.position.y = min_y;
        occupancy_grid_.info.origin.position.z = 0.0;
        occupancy_grid_.info.origin.orientation.w = 1.0;
        occupancy_grid_.header.frame_id = "odom";
        occupancy_grid_.header.stamp = this->now();

        occupancy_grid_.data.resize(width_ * height_, -1);

        for (const auto &point : cloud->points)
        {
            int x_idx = static_cast<int>((point.x - occupancy_grid_.info.origin.position.x) / resolution_);
            int y_idx = static_cast<int>((point.y - occupancy_grid_.info.origin.position.y) / resolution_);

            if (x_idx >= 0 && x_idx < width_ && y_idx >= 0 && y_idx < height_)
            {
                for (size_t i = 0; i < 3; i++)
                {
                    for (size_t j = 0; j < 3; j++)
                    {
                        if (x_idx >= 0 && x_idx < width_ && y_idx >= 0 && y_idx < height_)
                        {
                            int index = (y_idx + i) * width_ + (x_idx + j);
                            occupancy_grid_.data[index] = 0;
                        }
                    }
                }
            }
        }
    }

    void updateOccupancyGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        occupancy_grid_.header.frame_id = "odom";
        occupancy_grid_.header.stamp = this->now();

        for (const auto &point : cloud->points)
        {
            int x_idx = static_cast<int>((point.x - occupancy_grid_.info.origin.position.x) / resolution_);
            int y_idx = static_cast<int>((point.y - occupancy_grid_.info.origin.position.y) / resolution_);

            if (x_idx >= 0 && x_idx < width_ && y_idx >= 0 && y_idx < height_)
            {
                for (size_t i = 0; i < 1; i++)
                {
                    for (size_t j = 0; j < 1; j++)
                    {
                        if (x_idx >= 0 && x_idx < width_ && y_idx >= 0 && y_idx < height_)
                        {
                            int index = (y_idx + i) * width_ + x_idx + j;
                            occupancy_grid_.data[index] = 100;
                        }
                    }
                }
            }
        }
        nav_msgs::msg::OccupancyGrid occupancy_grid_before = occupancy_grid_;
        erode(occupancy_grid_before, occupancy_grid_, 3, 4);
    }

    void generatePointCloud2(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        pcl::toROSMsg(*cloud, pointcloud2_);
        pointcloud2_.header.frame_id = "odom";
        pointcloud2_.header.stamp = this->now();
    }

    void publishData()
    {
        nav2_costmap_2d::Costmap2D costmap(occupancy_grid_);

        auto costmap_msg = std::make_unique<nav2_msgs::msg::Costmap>();
        costmap_msg->header = occupancy_grid_.header;
        costmap_msg->metadata.size_x = costmap.getSizeInCellsX();
        costmap_msg->metadata.size_y = costmap.getSizeInCellsY();
        costmap_msg->metadata.resolution = costmap.getResolution();
        costmap_msg->metadata.layer = "master";
        costmap_msg->metadata.update_time = this->now();
        costmap_msg->metadata.origin = occupancy_grid_.info.origin;

        costmap_msg->data.resize(costmap.getSizeInCellsX() * costmap.getSizeInCellsY());
        unsigned char *char_map = costmap.getCharMap();
        std::copy(char_map, char_map + costmap_msg->data.size(), costmap_msg->data.begin());

        costmap_publisher_->publish(std::move(costmap_msg));
        RCLCPP_INFO(this->get_logger(), "Published costmap");

        pointcloud_publisher_->publish(pointcloud2_);
        RCLCPP_INFO(this->get_logger(), "Published pointcloud");
    }

private:
    int mean_k_;
    double stddev_mul_thresh_;
    double radius_search_;
    int min_neighbors_inRadius_;
    double height_threshold_;
    double ground_height_threshold_;
    rclcpp::Publisher<nav2_msgs::msg::Costmap>::SharedPtr costmap_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
    nav_msgs::msg::OccupancyGrid occupancy_grid_;
    sensor_msgs::msg::PointCloud2 pointcloud2_;
    rclcpp::TimerBase::SharedPtr timer_;
    Eigen::Vector3d unit_vec_ = Eigen::Vector3d::UnitZ();
    double resolution_;
    int width_;
    int height_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PCDToOccupancyGrid>());
    rclcpp::shutdown();
    return 0;
}
