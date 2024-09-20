#pragma once
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <climits> 

// void passThroughFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
//                     pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
//                     double low, double high)
// {
//     pcl::PassThrough<pcl::PointXYZ> pass;
//     pass.setInputCloud(input_cloud);
//     pass.setFilterFieldName("z");  
//     pass.setFilterLimits(low, high);  
//     pass.filter(*filtered_cloud);
// }

void erode(const nav_msgs::msg::OccupancyGrid& occupancy_grid_before, 
            nav_msgs::msg::OccupancyGrid& occupancy_grid_after,
            int kernel_size = 3,
            int kernel_num_threshold = 1){

    int height = occupancy_grid_before.info.height;
    int width = occupancy_grid_before.info.width;

    int kCenterY = kernel_size / 2;
    int kCenterX = kernel_size / 2;

     // 遍历图像每个像素（跳过边界）
    for (int y = kCenterY; y < height - kCenterY; y++) {
        for (int x = kCenterX; x < width - kCenterX; x++) {
            // 遍历核
            int kernel_num = 0;
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    // 获取当前核覆盖的像素值
                    int index = (y + ky - kCenterY) * width + (x + kx - kCenterX);
                    int pixelValue = occupancy_grid_before.data[index];
                    if(pixelValue == 100){
                        kernel_num++;
                    }
                    if(kernel_num >= kernel_num_threshold){
                        occupancy_grid_after.data[y*width+x] = 100;
                        break;
                    }
                }
            }
        }
    }
}
