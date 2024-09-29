#include "perception_common.h"

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

void erode(const nav_msgs::msg::OccupancyGrid &occupancy_grid_before,
           nav_msgs::msg::OccupancyGrid &occupancy_grid_after,
           int kernel_size,
           int kernel_num_threshold,
           int flag)
{
    int fill_value = 100;
    if (flag == 0)
    {
        fill_value = 0;
    }
    int height = occupancy_grid_before.info.height;
    int width = occupancy_grid_before.info.width;

    int kCenterY = kernel_size / 2;
    int kCenterX = kernel_size / 2;

    // 遍历图像每个像素（跳过边界）
    for (int y = kCenterY; y < height - kCenterY; y++)
    {
        for (int x = kCenterX; x < width - kCenterX; x++)
        {
            // 遍历核
            int kernel_num = 0;
            for (int ky = 0; ky < kernel_size; ky++)
            {
                for (int kx = 0; kx < kernel_size; kx++)
                {
                    // 获取当前核覆盖的像素值
                    int index = (y + ky - kCenterY) * width + (x + kx - kCenterX);
                    int pixelValue = occupancy_grid_before.data[index];
                    if (pixelValue == fill_value)
                    {
                        kernel_num++;
                    }
                    if (kernel_num >= kernel_num_threshold)
                    {
                        occupancy_grid_after.data[y * width + x] = fill_value;
                        break;
                    }
                }
            }
        }
    }
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

void voxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
                     float leaf_size)
{
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(input_cloud);
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    sor.filter(*filtered_cloud);
}

void RadiusOutlierFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
                         double radius_search,
                         int min_neighbors_inRadius)
{

    pcl::RadiusOutlierRemoval<pcl::PointXYZ> radiusoutlier;

    radiusoutlier.setInputCloud(input_cloud);
    radiusoutlier.setRadiusSearch(radius_search);
    radiusoutlier.setMinNeighborsInRadius(min_neighbors_inRadius);
    radiusoutlier.filter(*filtered_cloud);
}

void statisticalOutlierRemovalFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
                                     int mean_k,
                                     double stddev_mul_thresh)
{
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(input_cloud);
    sor.setMeanK(mean_k);                      // 邻域点的数量
    sor.setStddevMulThresh(stddev_mul_thresh); // 离群点的标准差阈值
    sor.filter(*filtered_cloud);
}
