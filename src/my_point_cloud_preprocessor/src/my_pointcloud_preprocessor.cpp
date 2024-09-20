#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <geometry_msgs/msg/polygon_stamped.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include "nav_msgs/msg/odometry.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <deque>

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl/filters/passthrough.h>
#include "passthrough_uint16.hpp"
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>

#include "occupancy_grid_map.h"
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <nav2_costmap_2d/nav2_costmap_2d/costmap_2d.hpp>
#include <pcl_ros/transforms.hpp>

#include <pointcloud_common.h>

struct CropBoxParam
{
    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float min_z;
    float max_z;
    bool negative{false};
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

struct PlaneBasis
{
    Eigen::Vector3d e_x;
    Eigen::Vector3d e_y;
    Eigen::Vector3d e_z;
};

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

void applyRANSAC(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &input, pcl::PointIndices::Ptr &output_inliers,
    pcl::ModelCoefficients::Ptr &output_coefficients)
{
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setRadiusLimits(0.3, std::numeric_limits<double>::max());
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(input);
    seg.setMaxIterations(1000);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.segment(*output_inliers, *output_coefficients);
}

struct CostTranslationTable
{
    CostTranslationTable()
    {
        for (int i = 0; i < 256; i++)
        {
            const auto value = static_cast<char>(static_cast<float>(i) * 100.f / 255.f);
            data[i] = std::max(std::min(value, static_cast<char>(99)), static_cast<char>(1));
        }
    }
    char operator[](unsigned char n) const { return data[n]; }
    char data[256];
};
static const CostTranslationTable cost_translation_table;

class PointCloudSubscriber : public rclcpp::Node
{
public:
    PointCloudSubscriber()
        : Node("pointcloud_subscriber")
    {
        // 创建一个订阅者，订阅 /pointcloud 话题
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar_points", 2,
            std::bind(&PointCloudSubscriber::pointcloud_callback, this, std::placeholders::_1));

        crop_box_polygon_pub_ =
            this->create_publisher<geometry_msgs::msg::PolygonStamped>("/crop_box_polygon", 2);

        crop_box_point_cloud_pub_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("/crop_box_point_cloud", 2);

        ring_passthrough_point_cloud_pub_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("/ring_passthrough_point_cloud", 2);

        // Subscriber
        twist_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 2,
            std::bind(
                &PointCloudSubscriber::onTwistWithCovarianceStamped, this, std::placeholders::_1));
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/Imu_data", 2,
            std::bind(&PointCloudSubscriber::onImu, this, std::placeholders::_1));

        // Publisher
        undistorted_points_pub_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("/undistort_pointcloud", 2);

        no_ground_points_pub_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("/no_ground_pointcloud", 2);

        no_static_points_pub_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("/no_static_pointcloud", 2);

        occupancy_grid_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/occupancy_grid_map", 2);
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/ekf_pose_with_covariance", 2,
                                                                                             std::bind(&PointCloudSubscriber::poseCallback, this, std::placeholders::_1));

        occupancy_grid_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("/visualization_costmap", 2,
                                                                                          std::bind(&PointCloudSubscriber::gridMaCallback, this, std::placeholders::_1));

        crop_outer_param_.negative = false;
        crop_outer_param_.min_x = -5;
        crop_outer_param_.min_y = -5;
        crop_outer_param_.min_z = -2;
        crop_outer_param_.max_x = 5;
        crop_outer_param_.max_y = 5;
        crop_outer_param_.max_z = 0;

        crop_inner_param_.negative = true;
        crop_inner_param_.min_x = -0.5;
        crop_inner_param_.min_y = -0.5;
        crop_inner_param_.min_z = -0.5;
        crop_inner_param_.max_x = 0.5;
        crop_inner_param_.max_y = 0.5;
        crop_inner_param_.max_z = 0;

        // set initial parameters
        int filter_min = 32;
        int filter_max = 64;
        impl_.setFilterLimits(filter_min, filter_max);

        impl_.setFilterFieldName("ring");
        impl_.setKeepOrganized(false);
        impl_.setFilterLimitsNegative(false);
    }

    void ground_filter(
        const sensor_msgs::msg::PointCloud2 &input, [[maybe_unused]] const pcl::IndicesPtr &indices,
        sensor_msgs::msg::PointCloud2 &output)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_sensor_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(input, *current_sensor_cloud_ptr);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(current_sensor_cloud_ptr);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-1.47, -1.07); // 雷达离地面约1.27m
        pass.filter(*cloud_filtered);

        // downsample pointcloud to reduce ransac calculation cost
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        downsampled_cloud->points.reserve(current_sensor_cloud_ptr->points.size());
        pcl::VoxelGrid<pcl::PointXYZ> filter;
        filter.setInputCloud(cloud_filtered);
        filter.setLeafSize(0.02, 0.02, 0.02);
        filter.filter(*downsampled_cloud);

        // apply ransac
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        applyRANSAC(downsampled_cloud, inliers, coefficients);

        if (coefficients->values.empty())
        {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), std::chrono::milliseconds(1000).count(),
                "failed to find a plane");
            output = input;
            return;
        }

        // filter too tilt plane to avoid mis-fitting (e.g. fitting to wall plane)
        Eigen::Vector3d plane_normal(
            coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        {
            const auto plane_slope = std::abs(
                std::acos(plane_normal.dot(unit_vec_) / (plane_normal.norm() * unit_vec_.norm())) * 180 /
                M_PI);
            if (plane_slope > 10.0)
            {
                output = input;
                return;
            }
        }

        // extract pointcloud from indices
        pcl::PointCloud<pcl::PointXYZ>::Ptr segment_ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr segment_no_ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        extractPointsIndices(
            downsampled_cloud, *inliers, segment_ground_cloud_ptr, segment_no_ground_cloud_ptr);
        const Eigen::Affine3d plane_affine = getPlaneAffine(*segment_ground_cloud_ptr, plane_normal);
        pcl::PointCloud<pcl::PointXYZ>::Ptr no_ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

        // use not downsampled pointcloud for extract pointcloud that higher than height threshold
        for (const auto &p : current_sensor_cloud_ptr->points)
        {
            const Eigen::Vector3d transformed_point =
                plane_affine.inverse() * Eigen::Vector3d(p.x, p.y, p.z);
            if (std::abs(transformed_point.z()) > 0.08) // height_threshold
            {
                no_ground_cloud_ptr->points.push_back(p);
            }
        }

        sensor_msgs::msg::PointCloud2::SharedPtr no_ground_cloud_msg_ptr(
            new sensor_msgs::msg::PointCloud2);
        pcl::toROSMsg(*no_ground_cloud_ptr, *no_ground_cloud_msg_ptr);
        no_ground_cloud_msg_ptr->header = input.header;
        output = *no_ground_cloud_msg_ptr;
        no_ground_points_pub_->publish(output);
    }

    void passthrough_filter(
        const sensor_msgs::msg::PointCloud2 &input, const pcl::IndicesPtr &indices, sensor_msgs::msg::PointCloud2 &output)
    {

        pcl::PCLPointCloud2::Ptr pcl_input(new pcl::PCLPointCloud2);
        pcl_conversions::toPCL(input, *(pcl_input));
        impl_.setInputCloud(pcl_input);
        impl_.setIndices(indices);
        pcl::PCLPointCloud2 pcl_output;
        impl_.filter(pcl_output);
        pcl_conversions::moveFromPCL(pcl_output, output);
        output.header = input.header;
        // ring_passthrough_point_cloud_pub_->publish(output);
    }

    void crop_box_filter(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &input, [[maybe_unused]] const pcl::IndicesPtr &indices,
        sensor_msgs::msg::PointCloud2 &output, CropBoxParam param)
    {
        output.data.resize(input->data.size());
        Eigen::Vector3f pt(Eigen::Vector3f::Zero());
        size_t j = 0;
        const auto data_size = input->data.size();
        const auto point_step = input->point_step;
        // If inside the cropbox
        if (!param.negative)
        {
            for (size_t i = 0; i + point_step < data_size; i += point_step)
            {
                memcpy(pt.data(), &input->data[i], sizeof(float) * 3);
                if (
                    param.min_z < pt.z() && pt.z() < param.max_z && param.min_y < pt.y() &&
                    pt.y() < param.max_y && param.min_x < pt.x() && pt.x() < param.max_x)
                {
                    memcpy(&output.data[j], &input->data[i], point_step);
                    j += point_step;
                }
            }
            // If outside the cropbox
        }
        else
        {
            for (size_t i = 0; i + point_step < data_size; i += point_step)
            {
                memcpy(pt.data(), &input->data[i], sizeof(float) * 3);
                if (
                    param.min_z > pt.z() || pt.z() > param.max_z || param.min_y > pt.y() ||
                    pt.y() > param.max_y || param.min_x > pt.x() || pt.x() > param.max_x)
                {
                    memcpy(&output.data[j], &input->data[i], point_step);
                    j += point_step;
                }
            }
        }

        output.data.resize(j);
        output.header = input->header;
        output.height = 1;
        output.fields = input->fields;
        output.is_bigendian = input->is_bigendian;
        output.point_step = input->point_step;
        output.is_dense = input->is_dense;
        output.width = static_cast<uint32_t>(output.data.size() / output.height / output.point_step);
        output.row_step = static_cast<uint32_t>(output.data.size() / output.height);
        // crop_box_point_cloud_pub_->publish(output);
        // publishCropBoxPolygon(param);
    }

    void publishCropBoxPolygon(CropBoxParam param)
    {
        auto generatePoint = [](double x, double y, double z)
        {
            geometry_msgs::msg::Point32 point;
            point.x = x;
            point.y = y;
            point.z = z;
            return point;
        };

        const double x1 = param.max_x;
        const double x2 = param.min_x;
        const double x3 = param.min_x;
        const double x4 = param.max_x;

        const double y1 = param.max_y;
        const double y2 = param.max_y;
        const double y3 = param.min_y;
        const double y4 = param.min_y;

        const double z1 = param.min_z;
        const double z2 = param.max_z;

        geometry_msgs::msg::PolygonStamped polygon_msg;
        polygon_msg.header.frame_id = "vanjee_lidar";
        polygon_msg.header.stamp = get_clock()->now();
        polygon_msg.polygon.points.push_back(generatePoint(x1, y1, z1));
        polygon_msg.polygon.points.push_back(generatePoint(x2, y2, z1));
        polygon_msg.polygon.points.push_back(generatePoint(x3, y3, z1));
        polygon_msg.polygon.points.push_back(generatePoint(x4, y4, z1));
        polygon_msg.polygon.points.push_back(generatePoint(x1, y1, z1));

        polygon_msg.polygon.points.push_back(generatePoint(x1, y1, z2));

        polygon_msg.polygon.points.push_back(generatePoint(x2, y2, z2));
        polygon_msg.polygon.points.push_back(generatePoint(x2, y2, z1));
        polygon_msg.polygon.points.push_back(generatePoint(x2, y2, z2));

        polygon_msg.polygon.points.push_back(generatePoint(x3, y3, z2));
        polygon_msg.polygon.points.push_back(generatePoint(x3, y3, z1));
        polygon_msg.polygon.points.push_back(generatePoint(x3, y3, z2));

        polygon_msg.polygon.points.push_back(generatePoint(x4, y4, z2));
        polygon_msg.polygon.points.push_back(generatePoint(x4, y4, z1));
        polygon_msg.polygon.points.push_back(generatePoint(x4, y4, z2));

        polygon_msg.polygon.points.push_back(generatePoint(x1, y1, z2));

        crop_box_polygon_pub_->publish(polygon_msg);
    }

    nav_msgs::msg::OccupancyGrid::UniquePtr OccupancyGridMapToMsgPtr(
        const std::string &frame_id, const rclcpp::Time &stamp, const float &robot_pose_z,
        const nav2_costmap_2d::Costmap2D &occupancy_grid_map)
    {
        auto msg_ptr = std::make_unique<nav_msgs::msg::OccupancyGrid>();

        msg_ptr->header.frame_id = frame_id;
        msg_ptr->header.stamp = stamp;
        msg_ptr->info.resolution = occupancy_grid_map.getResolution();

        msg_ptr->info.width = occupancy_grid_map.getSizeInCellsX();
        msg_ptr->info.height = occupancy_grid_map.getSizeInCellsY();

        double wx{};
        double wy{};
        occupancy_grid_map.mapToWorld(0, 0, wx, wy);
        msg_ptr->info.origin.position.x = occupancy_grid_map.getOriginX();
        msg_ptr->info.origin.position.y = occupancy_grid_map.getOriginY();
        msg_ptr->info.origin.position.z = robot_pose_z;
        msg_ptr->info.origin.orientation.w = 1.0;

        msg_ptr->data.resize(msg_ptr->info.width * msg_ptr->info.height);

        unsigned char *data = occupancy_grid_map.getCharMap();
        for (unsigned int i = 0; i < msg_ptr->data.size(); ++i)
        {
            msg_ptr->data[i] = cost_translation_table[data[i]];
        }
        return msg_ptr;
    }

    bool isOccupied(const pcl::PointXYZ &pt)
    {
        int x_idx = static_cast<int>((pt.x - occupancy_grid_map_.info.origin.position.x) / 0.05);
        int y_idx = static_cast<int>((pt.y - occupancy_grid_map_.info.origin.position.y) / 0.05);
        int width = occupancy_grid_map_.info.width;
        int height = occupancy_grid_map_.info.height;
        if (x_idx >= 0 && x_idx < width && y_idx >= 0 && y_idx < height)
        {
            int index = y_idx * width + x_idx;

            if (occupancy_grid_map_.data[index] == 100)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        return false;
    }

    /**
     * @brief perform background subtraction
     * @param src_cloud
     * @return src_cloud - bg_cloud
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr backgroundFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud)
    {
        geometry_msgs::msg::TransformStamped transform_stamped;
        // 设置 TransformStamped 的时间戳和 frame_id
        transform_stamped.header = pose_.header; // 使用当前时间戳

        // 设置平移部分 (pose.position -> transform.translation)
        transform_stamped.transform.translation.x = pose_.pose.pose.position.x;
        transform_stamped.transform.translation.y = pose_.pose.pose.position.y;
        transform_stamped.transform.translation.z = pose_.pose.pose.position.z;

        // 设置旋转部分 (pose.orientation -> transform.rotation)
        transform_stamped.transform.rotation.x = pose_.pose.pose.orientation.x;
        transform_stamped.transform.rotation.y = pose_.pose.pose.orientation.y;
        transform_stamped.transform.rotation.z = pose_.pose.pose.orientation.z;
        transform_stamped.transform.rotation.w = pose_.pose.pose.orientation.w;

        // 应用变换，将点云从 "lidar_frame" 坐标系转换到 "base_link" 坐标系
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        pcl_ros::transformPointCloud(*src_cloud, *transformed_cloud, transform_stamped);

        pcl::PointCloud<pcl::PointXYZ>::Ptr dst_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        dst_cloud->reserve(src_cloud->size() / 4);

        for (auto pt = transformed_cloud->begin(); pt != transformed_cloud->end(); pt++)
        {
            if (isOccupied(*pt) == false)
            {
                dst_cloud->push_back(*pt);
            }
        }
        dst_cloud->header = src_cloud->header;
        dst_cloud->width = dst_cloud->points.size();
        dst_cloud->height = 1;
        dst_cloud->is_dense = false;

        // pcl::io::savePCDFileASCII("/home/dyf/project/lidar_camera_perception_ws/src/my_point_cloud_preprocessor/data", *dst_cloud);
        // std::cout << "pcd saved" << std::endl;
        return dst_cloud;
    }

private:
    int pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr points)
    {

        pcl::IndicesPtr indice;

        sensor_msgs::msg::PointCloud2::Ptr output(new sensor_msgs::msg::PointCloud2);
        crop_box_filter(points, indice, *output, crop_outer_param_);

        sensor_msgs::msg::PointCloud2 output1;
        crop_box_filter(output, indice, output1, crop_inner_param_);

        sensor_msgs::msg::PointCloud2 output2;
        passthrough_filter(output1, indice, output2);

        /************************* point cloud undistort *********************/

        // if (output2.data.empty() || twist_queue_.empty())
        // {
        //     RCLCPP_WARN_STREAM_THROTTLE(
        //         get_logger(), *get_clock(), 10000 /* ms */,
        //         "input_pointcloud->points or twist_queue_ is empty.");
        //     return false;
        // }
        // // auto time_stamp_field_it = std::find_if(
        // //     std::cbegin(points->fields), std::cend(points->fields),
        // //     [this](const sensor_msgs::msg::PointField &field)
        // //     {
        // //         return field.name == "time_stamp";
        // //     });
        // // if (time_stamp_field_it == points->fields.cend())
        // // {
        // //     RCLCPP_WARN_STREAM_THROTTLE(
        // //         get_logger(), *get_clock(), 10000 /* ms */,
        // //         "Required field time stamp doesn't exist in the point cloud.");
        // //     return false;
        // // }

        // sensor_msgs::PointCloud2Iterator<float> it_x(output2, "x");
        // sensor_msgs::PointCloud2Iterator<float> it_y(output2, "y");
        // sensor_msgs::PointCloud2Iterator<float> it_z(output2, "z");
        // // sensor_msgs::PointCloud2ConstIterator<double> it_time_stamp(*points, "time_stamp");

        // float theta{0.0f};
        // float x{0.0f};
        // float y{0.0f};
        // // double prev_time_stamp_sec{*it_time_stamp};
        // const double first_point_time_stamp_sec{rclcpp::Time(output2.header.stamp).seconds()};
        // double *it_time_stamp = new double;
        // *it_time_stamp = first_point_time_stamp_sec;

        // auto twist_it = std::lower_bound(
        //     std::begin(twist_queue_), std::end(twist_queue_), first_point_time_stamp_sec,
        //     [](const geometry_msgs::msg::TwistStamped &x, const double t)
        //     {
        //         return rclcpp::Time(x.header.stamp).seconds() < t;
        //     });
        // twist_it = twist_it == std::end(twist_queue_) ? std::end(twist_queue_) - 1 : twist_it;

        // decltype(angular_velocity_queue_)::iterator imu_it;
        // if (!angular_velocity_queue_.empty())
        // {
        //     imu_it = std::lower_bound(
        //         std::begin(angular_velocity_queue_), std::end(angular_velocity_queue_),
        //         first_point_time_stamp_sec, [](const geometry_msgs::msg::Vector3Stamped &x, const double t)
        //         { return rclcpp::Time(x.header.stamp).seconds() < t; });
        //     imu_it =
        //         imu_it == std::end(angular_velocity_queue_) ? std::end(angular_velocity_queue_) - 1 : imu_it;
        // }

        // for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z /*, ++it_time_stamp*/)
        // {
        //     *it_time_stamp += 0.1 / 38400;
        //     for (;
        //          (twist_it != std::end(twist_queue_) - 1 &&
        //           *it_time_stamp > rclcpp::Time(twist_it->header.stamp).seconds());
        //          ++twist_it)
        //     {
        //     }

        //     float v{static_cast<float>(twist_it->twist.linear.x)};
        //     float w{static_cast<float>(twist_it->twist.angular.z)};

        //     if (std::abs(*it_time_stamp - rclcpp::Time(twist_it->header.stamp).seconds()) > 0.1)
        //     {
        //         RCLCPP_WARN_STREAM_THROTTLE(
        //             get_logger(), *get_clock(), 10000 /* ms */,
        //             "twist time_stamp is too late. Could not interpolate.");
        //         v = 0.0f;
        //         w = 0.0f;
        //     }

        //     if (!angular_velocity_queue_.empty())
        //     {
        //         for (;
        //              (imu_it != std::end(angular_velocity_queue_) - 1 &&
        //               *it_time_stamp > rclcpp::Time(imu_it->header.stamp).seconds());
        //              ++imu_it)
        //         {
        //         }
        //         if (std::abs(*it_time_stamp - rclcpp::Time(imu_it->header.stamp).seconds()) > 0.1)
        //         {
        //             RCLCPP_WARN_STREAM_THROTTLE(
        //                 get_logger(), *get_clock(), 10000 /* ms */,
        //                 "imu time_stamp is too late. Could not interpolate.");
        //         }
        //         else
        //         {
        //             w = static_cast<float>(imu_it->vector.z);
        //         }
        //     }

        //     // const float time_offset = static_cast<float>(*it_time_stamp - prev_time_stamp_sec);
        //     const float time_offset = 0.1 / 38400; // 假设100ms采集完38400个点

        //     const tf2::Vector3 lidarTF_point{*it_x, *it_y, *it_z};

        //     theta += w * time_offset;
        //     tf2::Quaternion lidar_quat{};
        //     lidar_quat.setRPY(0.0, 0.0, theta);

        //     const float dis = v * time_offset;
        //     x += dis * std::cos(theta);
        //     y += dis * std::sin(theta);

        //     tf2::Transform lidarTF_odom{};
        //     lidarTF_odom.setOrigin(tf2::Vector3(x, y, 0.0));
        //     lidarTF_odom.setRotation(lidar_quat);

        //     const tf2::Vector3 lidarTF_trans_point{lidarTF_odom * lidarTF_point};

        //     *it_x = lidarTF_trans_point.getX();
        //     *it_y = lidarTF_trans_point.getY();
        //     *it_z = lidarTF_trans_point.getZ();

        //     // prev_time_stamp_sec = *it_time_stamp;
        // }
        // undistorted_points_pub_->publish(output2);
        /************************* point cloud undistort *********************/

        sensor_msgs::msg::PointCloud2 output3;
        ground_filter(output2, indice, output3);

        /**** occupancy grip map ****/
        // geometry_msgs::msg::Pose pose; // lidar2map ,目前没有使用定位
        // pose.position.x = 0;
        // pose.position.y = 0;
        // pose.position.z = 0;
        // pose.orientation.w = 1;
        // pose.orientation.x = 0;
        // pose.orientation.y = 0;
        // pose.orientation.z = 0;

        // Create single frame occupancy grid map
        OccupancyGridMap single_frame_occupancy_grid_map(
            200,
            200,
            0.05, -5, -5); // 200 个栅格 ，每个栅格0.05m，起始点为-5,-5,因为cropbox 保留了10x10

        single_frame_occupancy_grid_map.updateWithPointCloud(output3, pose_.pose.pose);
        occupancy_grid_map_pub_->publish(OccupancyGridMapToMsgPtr(
            "odom", points->header.stamp, 0, single_frame_occupancy_grid_map));
        /**** occupancy grip map ****/

        pcl::PointCloud<pcl::PointXYZ>::Ptr output3_pcl(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(output3, *output3_pcl);

        auto output4_pcl = backgroundFilter(output3_pcl);

        sensor_msgs::msg::PointCloud2::SharedPtr output4(
            new sensor_msgs::msg::PointCloud2);
        pcl::toROSMsg(*output4_pcl, *output4);
        output4->header.frame_id = "odom";
        no_static_points_pub_->publish(*output4);
    }

    void onTwistWithCovarianceStamped(const nav_msgs::msg::Odometry::ConstSharedPtr twist_msg)
    {
        geometry_msgs::msg::TwistStamped msg;
        msg.header = twist_msg->header;
        msg.twist = twist_msg->twist.twist;
        twist_queue_.push_back(msg);

        while (!twist_queue_.empty())
        {
            // for replay rosbag
            if (rclcpp::Time(twist_queue_.front().header.stamp) > rclcpp::Time(twist_msg->header.stamp))
            {
                twist_queue_.pop_front();
            }
            else if ( // NOLINT
                rclcpp::Time(twist_queue_.front().header.stamp) <
                rclcpp::Time(twist_msg->header.stamp) - rclcpp::Duration::from_seconds(1.0))
            {
                twist_queue_.pop_front();
            }
            break;
        }
        // 为什么这里没有将线速度转换到lidar坐标系下
        // 因为线速度只需要进行旋转变换，不涉及位移，而当前的lidar和base_link只有位移没有旋转。
    }

    void onImu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg)
    {
        geometry_msgs::msg::Vector3Stamped angular_velocity;
        angular_velocity.vector = imu_msg->angular_velocity;

        geometry_msgs::msg::TransformStamped::SharedPtr tf_imu2lidar_ptr = std::make_shared<geometry_msgs::msg::TransformStamped>();
        tf_imu2lidar_ptr->transform.rotation.w = 1;
        tf_imu2lidar_ptr->transform.rotation.x = 0;
        tf_imu2lidar_ptr->transform.rotation.y = 0;
        tf_imu2lidar_ptr->transform.rotation.z = 0; // 无旋转
        tf_imu2lidar_ptr->transform.translation.x = 0.03;
        tf_imu2lidar_ptr->transform.translation.y = 0.0;
        tf_imu2lidar_ptr->transform.translation.z = -0.38;

        geometry_msgs::msg::Vector3Stamped transformed_angular_velocity;

        tf2::doTransform(angular_velocity, transformed_angular_velocity, *tf_imu2lidar_ptr);
        transformed_angular_velocity.header = imu_msg->header;
        angular_velocity_queue_.push_back(transformed_angular_velocity);

        while (!angular_velocity_queue_.empty())
        {
            // for replay rosbag
            if (
                rclcpp::Time(angular_velocity_queue_.front().header.stamp) >
                rclcpp::Time(imu_msg->header.stamp))
            {
                angular_velocity_queue_.pop_front();
            }
            else if ( // NOLINT
                rclcpp::Time(angular_velocity_queue_.front().header.stamp) <
                rclcpp::Time(imu_msg->header.stamp) - rclcpp::Duration::from_seconds(1.0))
            {
                angular_velocity_queue_.pop_front();
            }
            break;
        }
    }

    void poseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr pose_in)
    {
        pose_ = *pose_in;
    }

    void gridMaCallback(const nav_msgs::msg::OccupancyGrid::UniquePtr map_in)
    {
        occupancy_grid_map_ = *map_in;
        std::cout << "grid_map reccived" << std::endl;
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr crop_box_polygon_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr crop_box_point_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ring_passthrough_point_cloud_pub_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr twist_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;

    std::deque<geometry_msgs::msg::TwistStamped> twist_queue_;
    std::deque<geometry_msgs::msg::Vector3Stamped> angular_velocity_queue_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr undistorted_points_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr no_ground_points_pub_;
    pcl::PassThroughUInt16<pcl::PCLPointCloud2> impl_;
    Eigen::Vector3d unit_vec_ = Eigen::Vector3d::UnitZ();

    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_map_pub_;
    geometry_msgs::msg::PoseWithCovarianceStamped pose_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_map_sub_;
    nav_msgs::msg::OccupancyGrid occupancy_grid_map_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr no_static_points_pub_;

    CropBoxParam crop_outer_param_, crop_inner_param_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    std::shared_ptr<PointCloudSubscriber> node = std::make_shared<PointCloudSubscriber>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin(); // Uses multiple threads to process callbacks
    rclcpp::shutdown();
    return 0;
}
