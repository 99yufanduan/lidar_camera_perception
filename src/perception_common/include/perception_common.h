#pragma once
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <climits> 
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>     
#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <pcl/filters/voxel_grid.h>

struct PlaneBasis
{
    Eigen::Vector3d e_x;
    Eigen::Vector3d e_y;
    Eigen::Vector3d e_z;
};

void passThroughFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
                    double low, double high);

void erode(const nav_msgs::msg::OccupancyGrid& occupancy_grid_before,
            nav_msgs::msg::OccupancyGrid& occupancy_grid_after,
            int kernel_size = 3,
            int kernel_num_threshold = 1,
             int flag = 1);


void applyRANSAC(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, pcl::PointIndices::Ptr &filtered_cloud,
    pcl::ModelCoefficients::Ptr &output_coefficients);

Eigen::Vector3d getArbitraryOrthogonalVector(const Eigen::Vector3d &input);

PlaneBasis getPlaneBasis(const Eigen::Vector3d &plane_normal);

Eigen::Affine3d getPlaneAffine(
const pcl::PointCloud<pcl::PointXYZ> segment_ground_cloud, const Eigen::Vector3d &plane_normal);

void extractPointsIndices(
const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, const pcl::PointIndices &in_indices,
pcl::PointCloud<pcl::PointXYZ>::Ptr out_only_indices_cloud_ptr,
pcl::PointCloud<pcl::PointXYZ>::Ptr out_removed_indices_cloud_ptr);

void voxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
                    float leaf_size = 0.05);


void RadiusOutlierFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
                        double radius_search,
                        int min_neighbors_inRadius);

void statisticalOutlierRemovalFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                                    pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
                                    int mean_k,
                                    double stddev_mul_thresh);



