#include <opencv2/opencv.hpp>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// PnP 误差模型
struct PnPResidual
{
    PnPResidual(const cv::Point3f &point3D, const cv::Point2f &point2D)
        : point3D_(point3D), point2D_(point2D) {}

    template <typename T>
    bool operator()(const T *const rvec, const T *const tvec, T *residual) const
    {
        // 相机内参矩阵
        Eigen::Matrix3d K;

        K << 470.78426, 0.0, 317.81025, // 焦距和主点
            0.0, 469.63534, 257.05152,
            0.0, 0.0, 1.0;
        // 转换旋转向量为旋转矩阵
        Eigen::Matrix<T, 3, 3> R;
        ceres::AngleAxisToRotationMatrix(rvec, R.data());

        // 计算投影
        Eigen::Matrix<T, 3, 1> P;
        P << (T)(point3D_.x), (T)(point3D_.y), (T)(point3D_.z);
        Eigen::Matrix<T, 3, 1> Tvec;
        Tvec << tvec[0], tvec[1], tvec[2];
        Eigen::Matrix<T, 3, 1> projected = (R * P + Tvec);
        auto projected1 = projected / projected[2];
        auto projected2 = K * projected1;
        // 计算残差
        residual[0] = T(point2D_.x) - projected2[0];
        residual[1] = T(point2D_.y) - projected2[1];
        return true;
    }

    cv::Point3f point3D_;
    cv::Point2f point2D_;
};

int main()
{
    // 3D 点在雷达坐标系下
    std::vector<cv::Point3f> objectPoints = {
        cv::Point3f(-0.477964, -0.404409, 1.870150),
        cv::Point3f(0.936607, -0.403547, 1.884733),
        cv::Point3f(-0.495926, 0.357302, 1.891702),
        cv::Point3f(0.931839, 0.412912, 1.926659),
        cv::Point3f(0.746390, -0.407215, 2.127130),
        cv::Point3f(0.710706, 0.453193, 2.163942),
        cv::Point3f(-0.492833, 0.469743, 1.338681),
        cv::Point3f(-0.649516, -0.439572, 2.355342),
        cv::Point3f(0.619771, -0.393251, 1.697563),
        cv::Point3f(-0.644704, 0.420201, 2.408443),
        cv::Point3f(0.611521, 0.451660, 1.757808)};

    // 2D 点在图像平面上的映射 /2 是因为亚像素化为原来的两倍了
    std::vector<cv::Point2f>
        imagePoints = {
            cv::Point2f(426 / 2, 139 / 2),
            cv::Point2f(1137 / 2, 143 / 2),
            cv::Point2f(430 / 2, 564 / 2),
            cv::Point2f(1170 / 2, 572 / 2),
            cv::Point2f(991 / 2, 204 / 2),
            cv::Point2f(1006 / 2, 586 / 2),
            cv::Point2f(308 / 2, 543 / 2),
            cv::Point2f(403 / 2, 243 / 2),
            cv::Point2f(1007 / 2, 110 / 2),
            cv::Point2f(392 / 2, 587 / 2),
            cv::Point2f(1037 / 2, 575 / 2)};

    // 相机内参矩阵
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 470.78426, 0.0, 317.81025, // 焦距和主点
                            0.0, 469.63534, 257.05152,
                            0.0, 0.0, 1.0);

    // 相机内参矩阵
    Eigen::Matrix3d K;

    // 畸变系数（假设无畸变）
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    // 旋转向量和位移向量
    cv::Mat rvec, tvec;

    // 使用 solvePnP 计算外参，选择算法为 EPnP, EPNP 最少需要四个点来求解线性方程，并且避免共面点，这会导致算法出现奇异性从而数值解不稳定
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);

    std::cout << "Rotation Vector (rvec):\n"
              << rvec << std::endl;
    std::cout << "Translation Vector (tvec):\n"
              << tvec << std::endl;

    // 计算旋转向量和位移向量 ，迭代计算pnp
    bool useExtrinsicGuess = true; // 设置为 true 以使用初始值
    bool success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, cv::SOLVEPNP_ITERATIVE);

    std::cout << "Rotation Vector (rvec):\n"
              << rvec << std::endl;
    std::cout << "Translation Vector (tvec):\n"
              << tvec << std::endl;

    // 初始化旋转向量和位移向量
    double rvec_array[3] = {rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0)};
    double tvec_array[3] = {tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)};

    // double rvec_array[3] = {0, 0.2, 0};
    // double tvec_array[3] = {0, 0, 0.64};

    // 创建问题
    ceres::Problem problem;
    for (size_t i = 0; i < objectPoints.size(); ++i)
    {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PnPResidual, 2, 3, 3>(
                new PnPResidual(objectPoints[i], imagePoints[i])),
            nullptr, rvec_array, tvec_array);
    }

    // 配置求解器
    ceres::Solver::Options options;
    // 最大迭代次数
    options.max_num_iterations = 1000;

    // 梯度容忍度
    options.gradient_tolerance = 1e-12;

    // 目标函数值容忍度
    options.function_tolerance = 1e-8;

    // 步长容忍度
    options.parameter_tolerance = 1e-8;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    std::cout << "Rotation Vector:\n"
              << std::endl;
    std::cout << rvec_array[0] << std::endl;
    std::cout << rvec_array[1] << std::endl;
    std::cout << rvec_array[2] << std::endl;

    std::cout << "Translation Vector:\n"
              << std::endl;
    std::cout << tvec_array[0] << std::endl;
    std::cout << tvec_array[1] << std::endl;
    std::cout << tvec_array[2] << std::endl;

    return 0;
}
