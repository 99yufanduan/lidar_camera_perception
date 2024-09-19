#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <cv_bridge/cv_bridge.h>

class LidarCameraAligner : public rclcpp::Node
{
public:
    LidarCameraAligner() : Node("lidar_camera_aligner")
    {
        depth_image_ = cv::Mat(329, 645, CV_32FC1);
        // Camera intrinsic parameters (assumed to be known)
        fx_ = 500.0; // Focal length in x (pixels)
        fy_ = 500.0; // Focal length in y (pixels)
        cx_ = 320.0; // Optical center x (pixels)
        cy_ = 240.0; // Optical center y (pixels)

        // Extrinsic parameters: rotation (R) and translation (t) from LiDAR to camera frame
        R_ << 1, 0, 0,
            0, 1, 0,
            0, 0, 1; // Identity matrix for no rotation (replace with actual values)

        t_ << 0, 0, 0; // No translation (replace with actual values)

        // Subscribe to the camera and LiDAR topics
        camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color_image", 10, std::bind(&LidarCameraAligner::cameraCallback, this, std::placeholders::_1));

        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar_points", 10, std::bind(&LidarCameraAligner::lidarCallback, this, std::placeholders::_1));

        // Publisher for the aligned LiDAR data
        aligned_lidar_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/aligned_lidar/point_cloud", 10);
        depth_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/depth_image", 10);
    }

private:
    void cameraCallback(const sensor_msgs::msg::Image::SharedPtr image_msg)
    {
        // Store the latest image for alignment (you can perform any image processing here)
        latest_image_ = image_msg;
    }

    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr lidar_msg)
    {
        // if (!latest_image_)
        // {
        //     RCLCPP_WARN(this->get_logger(), "No image received yet.");
        //     return;
        // }

        // Convert PointCloud2 to PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*lidar_msg, *pcl_cloud);

        // Extract the relevant LiDAR data that corresponds to the camera's field of view
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        for (const auto &point : pcl_cloud->points)
        {
            // Convert the point to Eigen for easier manipulation
            Eigen::Vector3f lidar_point(point.x, point.y, point.z);

            // Transform the LiDAR point to the camera coordinate system
            Eigen::Vector3f camera_point = R_ * lidar_point + t_;

            // Project the transformed point onto the 2D image plane
            if (isPointInCameraFOV(camera_point))
            {
                aligned_pcl_cloud->points.push_back(point);
            }
        }

        // Convert the aligned PCL PointCloud back to PointCloud2
        sensor_msgs::msg::PointCloud2 aligned_msg;
        pcl::toROSMsg(*aligned_pcl_cloud, aligned_msg);
        aligned_msg.header = lidar_msg->header;

        // Convert cv::Mat to sensor_msgs::msg::Image using cv_bridge
        auto depth_image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", depth_image_).toImageMsg();

        // Publish the image
        depth_image_pub_->publish(*depth_image_msg);
        depth_image_ = cv::Mat::zeros(329, 645, CV_32FC1);

        // Publish the aligned LiDAR data
        aligned_lidar_pub_->publish(aligned_msg);
    }

    bool isPointInCameraFOV(const Eigen::Vector3f &point)
    {
        if (point[0] < 0)
        {
            return false;
        }
        // Project the 3D point in the camera frame to the 2D image plane
        float u = (fx_ * point[1] / point[0]) + cx_;
        float v = (fy_ * point[2] / point[0]) + cy_;

        // Check if the projected point lies within the image bounds
        if (u >= 0 && u < 640 && v >= 0 && v < 320)
        {
            cv::Rect roi(std::round(u), std::round(v), 4, 8);
            // Access the ROI in the matrix
            cv::Mat mat_roi = depth_image_(roi);
            mat_roi.setTo(point[0]);
            return true;
        }
        return false;
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_lidar_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub_;

    sensor_msgs::msg::Image::SharedPtr latest_image_;

    // Camera intrinsic parameters
    float fx_, fy_, cx_, cy_;

    // Extrinsic parameters (rotation and translation from LiDAR to camera frame)
    Eigen::Matrix3f R_; // Rotation matrix
    Eigen::Vector3f t_; // Translation vector
    // 创建一个 640x320 的深度图
    cv::Mat depth_image_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarCameraAligner>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
