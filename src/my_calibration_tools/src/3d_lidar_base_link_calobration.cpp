#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl/filters/passthrough.h>

#include <pcl_conversions/pcl_conversions.h>

Eigen::Matrix3d computeRotationMatrix(double a, double b, double c)
{
    // 归一化法向量
    Eigen::Vector3d n(a, b, c);
    n.normalize();

    // 如果 n 已经是 z 轴方向，则返回单位矩阵
    if (n.isApprox(Eigen::Vector3d(0, 0, 1)))
    {
        return Eigen::Matrix3d::Identity();
    }

    // 标准向量 z 轴
    Eigen::Vector3d z(0, 0, 1);

    // 计算旋转轴
    Eigen::Vector3d v = z.cross(n);
    v.normalize();

    // 计算旋转角度
    double cosTheta = n.z(); // 点积 n 和 z 轴 (0, 0, 1)
    double theta = std::acos(cosTheta);

    // Rodrigues 公式计算旋转矩阵
    Eigen::Matrix3d K;
    K << 0, -v.z(), v.y(),
        v.z(), 0, -v.x(),
        -v.y(), v.x(), 0;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1 - std::cos(theta)) * K * K;

    return R;
}

double distanceFromOriginToPlane(double a, double b, double c, double d)
{
    // 计算法向量的模长
    double norm = std::sqrt(a * a + b * b + c * c);

    // 计算距离
    return std::abs(d) / norm;
}

class PointCloudSubscriber : public rclcpp::Node
{
public:
    PointCloudSubscriber()
        : Node("lidar_base_link_calobration")
    {
        // 创建一个订阅者，订阅 /pointcloud 话题
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar_points", 10,
            std::bind(&PointCloudSubscriber::pointcloud_callback, this, std::placeholders::_1));
    }

    void applyRANSAC(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &input, pcl::PointIndices::Ptr &output_inliers,
        pcl::ModelCoefficients::Ptr &output_coefficients)
    {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setRadiusLimits(0.3, std::numeric_limits<double>::max());
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.005);
        seg.setInputCloud(input);
        seg.setMaxIterations(5000);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.segment(*output_inliers, *output_coefficients);
    }

    int pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr points)
    {
        std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        pcl::fromROSMsg(*points, *pcl_cloud);

        std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(pcl_cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-1.47, -1.07); // 雷达离地面约1.27m
        pass.filter(*cloud_filtered);
        // apply ransac
        std::shared_ptr<pcl::PointIndices> inliers = std::make_shared<pcl::PointIndices>();
        std::shared_ptr<pcl::ModelCoefficients> coefficients = std::make_shared<pcl::ModelCoefficients>();
        applyRANSAC(cloud_filtered, inliers, coefficients);
        double a = coefficients->values[0];
        double b = coefficients->values[1];
        double c = coefficients->values[2];
        double d = coefficients->values[3];

        if (c < 0) // 保证平面法向量始终朝上
        {
            a = -a;
            b = -b;
            c = -c;
            d = -d;
        }

        std::cout << "a: " << a << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << "c: " << c << std::endl;
        std::cout << "d: " << d << std::endl;

        double height = distanceFromOriginToPlane(a, b, c, d); // 平面到原点的距离，即雷达到平面的高度
        std::cout << "height " << height << std::endl;

        Eigen::Matrix3d R = computeRotationMatrix(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        std::cout << "R " << R << std::endl;
        // TODO 2024/09/11 Release 下会出现segment fault ，debug不会
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
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
