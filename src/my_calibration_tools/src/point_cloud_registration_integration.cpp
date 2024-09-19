#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thread>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <cstdlib> // 包含 system 函数的头文件

bool captrue_flag = 1;

// 鼠标回调函数，获取鼠标点击位置的坐标
void onMouse(int event, int x, int y, int, void *userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        // 输出点击的坐标
        std::cout << "Mouse Clicked at: (" << x << ", " << y << ")" << std::endl;

        // 获取传入的图像
        cv::Mat *img = reinterpret_cast<cv::Mat *>(userdata);

        // 确保坐标在图像范围内
        if (x >= 0 && x < img->cols && y >= 0 && y < img->rows)
        {
            std::cout << "piexl point at (" << x << ", " << y << "): ";

            // 在图像中绘制圆形标记点击位置
            cv::circle(*img, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1); // 红色圆，半径为5
            // 更新显示图像
            cv::imshow("Image", *img);
        }
    }
}

void executeCommand(const std::string &command)
{
    std::system(command.c_str());
}

class PointCloudStitcher : public rclcpp::Node
{
public:
    PointCloudStitcher()
        : Node("point_cloud_registration_integration")
    {
        // 订阅来自传感器的点云
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar_points", 10, std::bind(&PointCloudStitcher::pointcloud_callback, this, std::placeholders::_1));

        // 发布拼接后的点云
        pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/stitched_pointcloud", 10);

        // 初始化拼接的点云
        stitched_cloud_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

        // Subscribe to the camera and LiDAR topics
        camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color_image", 10, std::bind(&PointCloudStitcher::cameraCallback, this, std::placeholders::_1));

        depth_image_ = cv::Mat(489, 645, CV_32FC1);
        aligned_pcl_cloud_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

        // Camera intrinsic parameters (assumed to be known)
        fx_ = 470.78426; // Focal length in x (pixels)
        fy_ = 469.63534; // Focal length in y (pixels)
        cx_ = 317.81025; // Optical center x (pixels)
        cy_ = 257.05152; // Optical center y (pixels)
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
    sensor_msgs::msg::Image::SharedPtr first_image_;

    cv::Mat depth_image_;
    float fx_, fy_, cx_, cy_;
    // Extract the relevant LiDAR data that corresponds to the camera's field of view
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_pcl_cloud_;

    void cameraCallback(const sensor_msgs::msg::Image::SharedPtr image_msg)
    {
        static bool first_flag = 1;
        if (first_flag == 1)
        {
            first_flag = 0;
            // Store the latest image for alignment (you can perform any image processing here)
            first_image_ = image_msg;
            std::cout << "first image saved" << std::endl;
        }
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
        if (u >= 0 && u < 640 && v >= 0 && v < 480)
        {
            cv::Rect roi(std::round(u), std::round(v), 4, 8);
            // Access the ROI in the matrix
            cv::Mat mat_roi = depth_image_(roi);
            mat_roi.setTo(point[0]);
            // depth_image_.at<float>(v, u) = point[0] * 25;
            return true;
        }
        return false;
    }

    // static void mouseCallback(const pcl::visualization::PointPickingEvent &event, void *viewer_void)
    // {
    //     pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);
    //     if (event.getPointIndex() != -1)
    //     {
    //         pcl::PointXYZ selected_point;
    //         event.getPoint(selected_point.x, selected_point.y, selected_point.z);

    //         std::cout << "Clicked point coordinates: "
    //                   << "x = " << selected_point.x
    //                   << ", y = " << selected_point.y
    //                   << ", z = " << selected_point.z
    //                   << std::endl;
    //     }
    // }

    // 订阅回调函数
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        static int captrue_num = 0;
        if (captrue_num == 10)
        {
            for (const auto &point : stitched_cloud_->points)
            {
                Eigen::Vector3f lidar_point(point.x, point.y, point.z);
                if (isPointInCameraFOV(lidar_point))
                {
                    aligned_pcl_cloud_->points.push_back(point);
                }
            }
            // 设置点云的宽度和高度（无序点云高度为 1，宽度为点的数量）
            aligned_pcl_cloud_->width = aligned_pcl_cloud_->points.size();
            aligned_pcl_cloud_->height = 1;
            aligned_pcl_cloud_->is_dense = true; // 设置点云是否是密集的
            // 保存点云到文件
            pcl::io::savePCDFileASCII("src/my_calibration_tools/data/output.pcd", *aligned_pcl_cloud_);
            std::cout << "Saved " << aligned_pcl_cloud_->points.size() << " data points to output.pcd" << std::endl;
            system("pcl_pcd2ply src/my_calibration_tools/data/output.pcd src/my_calibration_tools/data/output.ply");
            std::thread thread1(executeCommand, "CloudCompare src/my_calibration_tools/data/output.ply");

            // 从 ROS 2 图像消息转换为 cv::Mat
            cv::Mat image = cv_bridge::toCvShare(first_image_, "bgr8")->image;

            // 定义新尺寸为原尺寸的两倍
            cv::Size newSize(image.cols * 2, image.rows * 2);

            // 使用插值方法放大图像 ，亚像素化
            cv::Mat resizedImage;
            cv::resize(image, resizedImage, newSize, 0, 0, cv::INTER_LANCZOS4);

            // 将图像转换为灰度图像
            cv::Mat grayImg;
            cv::cvtColor(resizedImage, grayImg, cv::COLOR_BGR2GRAY);

            // 执行 Canny 边缘检测
            cv::Mat edges;
            double lowThreshold = 50;
            double highThreshold = 150;
            int kernelSize = 3; // Sobel 算子大小
            cv::Canny(grayImg, edges, lowThreshold, highThreshold, kernelSize);

            // cv::imwrite("src/my_calibration_tools/data/output_raw.jpg", resizedImage);
            // cv::imwrite("src/my_calibration_tools/data/output_edges.jpg", edges);
            // system("gimp src/my_calibration_tools/data/output_raw.jpg");
            // system("gimp src/my_calibration_tools/data/output_edges.jpg");

            // 创建窗口并显示图像
            cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
            cv::imshow("Image", image);

            // 设置鼠标回调函数
            cv::setMouseCallback("Image", onMouse, &image);

            // 等待按键，按任意键退出
            cv::waitKey(0);
            exit(0);
        }

        if (captrue_flag)
        {
            std::cout << "16:39:19: " << std::endl;
            captrue_flag = 0;
            captrue_num++;
            // 将 ROS 2 点云转换为 PCL 点云格式
            pcl ::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *cloud);
            // 创建一个存储过滤后点云的容器
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

            // 使用 PCL 的 removeNaNFromPointCloud 函数来去除 NaN 点
            std::vector<int> indices; // 过滤掉的点的索引
            pcl::removeNaNFromPointCloud(*cloud, *filtered_cloud, indices);

            // 现在 filtered_cloud 中已经去除了 NaN 点，可以进一步使用或拼接
            std::cout << "Original cloud size: " << cloud->points.size() << std::endl;
            std::cout << "Filtered cloud size: " << filtered_cloud->points.size() << std::endl;

            // 如果需要更新原始点云
            *cloud = *filtered_cloud;

            static pcl ::PointCloud<pcl::PointXYZ>::Ptr first_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            if (stitched_cloud_->empty())
            {
                *stitched_cloud_ = *cloud;
                *first_cloud = *cloud;
            }
            else
            {
                // 使用 ICP 进行点云配准
                pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
                // 设置 ICP 参数

                icp.setTransformationEpsilon(1e-10);          // 更严格的收敛条件
                icp.setMaxCorrespondenceDistance(0.12);       // 减小最大对应点距离
                icp.setMaximumIterations(1000);               // 设置较高的迭代次数
                icp.setRANSACOutlierRejectionThreshold(0.01); // 降低 RANSAC 阈值
                icp.setEuclideanFitnessEpsilon(1e-8);         // 提高最终收敛的拟合精度

                icp.setInputSource(cloud);
                icp.setInputTarget(first_cloud);

                pcl::PointCloud<pcl::PointXYZ> final_cloud;
                pcl::PointCloud<pcl::PointXYZ> final_cloud2;
                auto initial_transform = Eigen::Matrix4f::Identity();
                icp.align(final_cloud, initial_transform);

                // 将该变换矩阵应用到原始点云上
                pcl::transformPointCloud(*cloud, final_cloud2, icp.getFinalTransformation());

                if (icp.hasConverged())
                {
                    RCLCPP_INFO(this->get_logger(), "ICP has converged with score: %f", icp.getFitnessScore());
                    std::cout << "17:18:06: " << icp.getFinalTransformation() << std::endl;

                    // 拼接点云
                    *stitched_cloud_ += final_cloud2;
                    std::cout << "14:34:28: " << stitched_cloud_->size() << std::endl;

                    // 发布拼接后的点云
                    sensor_msgs::msg::PointCloud2 output_msg;
                    pcl::toROSMsg(*stitched_cloud_, output_msg);
                    output_msg.header.frame_id = "odom"; // 保持时间戳和帧 ID
                    pointcloud_pub_->publish(output_msg);
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "ICP did not converge.");
                }
            }
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr stitched_cloud_;
};

// 处理键盘输入的线程函数
void keyboardListener()
{
    char command;
    while (1)
    {
        std::cin >> command;
        if (command == 'c')
        {
            captrue_flag = 1;
            std::cout << "captrue" << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    std::shared_ptr<PointCloudStitcher> node = std::make_shared<PointCloudStitcher>();

    // 启动键盘监听线程
    std::thread keyboard_thread(keyboardListener);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin(); // Uses multiple threads to process callbacks

    rclcpp::shutdown();
    return 0;
}
