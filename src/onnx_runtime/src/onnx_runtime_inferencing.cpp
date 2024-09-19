#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <cv_bridge/cv_bridge.h>

#include "onnxruntime_cxx_api.h"
#include <vector>

std::vector<std::string> class_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

// 将 OpenCV Mat 转换为 ONNX 推理所需的输入 tensor
std::vector<float> convertMatToTensor(const cv::Mat &img)
{
    std::vector<float> input_tensor_values;

    input_tensor_values.assign((float *)img.data, (float *)img.data + img.total() * img.channels());
    return input_tensor_values;
}

// 图像预处理函数
cv::Mat preprocessImage(const cv::Mat &img, const std::vector<int64_t> &input_shape)
{
    cv::Mat resized_img, float_img;
    // 将图像调整为模型的输入尺寸
    cv::resize(img, resized_img, cv::Size(input_shape[3], input_shape[2]));
    // 转换为 float32 类型
    resized_img.convertTo(float_img, CV_32F, 1.0 / 255);
    // 转置为 (C, H, W) 格式
    cv::dnn::blobFromImage(float_img, float_img);
    return float_img;
}

void drawBoxes(cv::Mat &img, const std::vector<cv::Rect> &boxes, const std::vector<int> &class_ids, std::vector<int> indices)
{
    for (auto i : indices)
    {
        // 获取边界框和类ID
        const cv::Rect &box = boxes[i];
        int class_id = class_ids[i];

        // 绘制边界框
        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2); // 使用绿色绘制矩形框

        // 绘制类别标签
        std::string label = class_names[class_id];
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(img, cv::Point(box.x, box.y - label_size.height - baseline), cv::Point(box.x + label_size.width, box.y), cv::Scalar(0, 255, 0), -1); // 背景框
        cv::putText(img, label, cv::Point(box.x, box.y - baseline), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);                          // 标签文本
    }
}

class OnnxRuntimeInferencing : public rclcpp::Node
{
private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr raw_img)
    {
        cv::Mat cv_image = cv_bridge::toCvCopy(raw_img, "bgr8")->image; // opencv 的格式是bgr
        // cv::Mat cv_image = cv::imread("/home/dyf/project/lidar_camera_perception_ws/src/onnx_runtime/data/bus.jpg"); // test_image

        std::vector<int64_t> onnx_input_shape = {1, 3, 640, 640}; // NCHW RGB格式
        cv::Mat preprocessed_image = preprocessImage(image, onnx_input_shape);

        // 将图像数据转换为一维数组,onnx输入的张量
        std::vector<float> onnx_input_tensor_values(preprocessed_image.begin<float>(), preprocessed_image.end<float>());

        // 4. 创建输入张量shape
        std::vector<int64_t> onnx_input_tensor_shape = {1, 3, input_shape[2], input_shape[3]};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // 创建张量
        Ort::Value onnx_input_tensor = Ort::Value::CreateTensor<float>(memory_info, onnx_input_tensor_values.data(), onnx_input_tensor_values.size(), onnx_input_tensor_shape.data(), onnx_input_tensor_shape.size());

        const std::array<const char *, 3> outNames = {output_node_names_[0].c_str(), output_node_names_[1].c_str(), output_node_names_[2].c_str()};

        const std::array<const char *, 1> inputNames = {input_node_names_[0].c_str()};

        auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor, 1, outNames.data(), 1);
        float *output_data = output_tensors.front().GetTensorMutableData<float>();

        auto output_shape = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        // 假设模型输出为 [num_boxes, 85]，其中 85 = 4 (bbox) + 1 (confidence) + 80 (classes)
        int num_boxes = output_shape[1];
        int num_classes = 80;
        float conf_threshold = 0.8;
        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        float nms_threshold = 0.5;
        std::vector<int> indices;
        std::vector<float> scores;

        for (int i = 0; i < num_boxes; ++i)
        {
            float *box_data = output_data + i * (4 + 1 + num_classes);
            float x = box_data[0];
            float y = box_data[1];
            float w = box_data[2];
            float h = box_data[3];
            float conf = box_data[4];

            if (conf > conf_threshold)
            {
                int class_id = std::max_element(box_data + 5, box_data + 5 + num_classes) - (box_data + 5);
                float class_conf = box_data[5 + class_id];
                if (class_conf > conf_threshold)
                {
                    int x1 = static_cast<int>((x - w / 2));
                    int y1 = static_cast<int>((y - h / 2));
                    int x2 = static_cast<int>((x + w / 2));
                    int y2 = static_cast<int>((y + h / 2));
                    std::cout << "15:15:18: " << conf << std::endl;
                    boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
                    scores.push_back(class_conf);
                    class_ids.push_back(class_id);
                }
            }
        }
        for (auto class_id : class_ids)
        {
            std::cout << "class_id: " << class_id << std::endl;
        }

        cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);

        cv::Mat resized_img;
        cv::resize(image, resized_img, cv::Size(input_shape[3], input_shape[2]));

        draw_boxes(resized_img, boxes, class_ids, indices);
        // // 显示检测后的图像
        cv::imshow("onnx", resized_img);
        cv::waitKey(1);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

    Ort::Session session_;
    // 获取模型输入输出信息
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;

public:
    onnx_runtime_inferencing(std::shared_ptr<Ort::Env> env, const char *model_path, std::shared_ptr<Ort::SessionOptions> session_options) : Node("onnx_runtime_inferencing"), session_(*env, model_path, *session_options)
    {
        cv::namedWindow("onnx", cv::WINDOW_AUTOSIZE);
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>("/camera/color_image", 10, std::bind(&onnx_runtime_inferencing::imageCallback, this, std::placeholders::_1));

        int input_nodes_num = session_.GetInputCount();
        int output_nodes_num = session_.GetOutputCount();

        // 获得输入信息
        for (int i = 0; i < input_nodes_num; i++)
        {
            auto input_name = session_.GetInputNameAllocated(i, allocator_);
            input_node_names_.push_back(input_name.get());
            auto inputShapeInfo = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            int ch = inputShapeInfo[1];
            int input_h = inputShapeInfo[2];
            int input_w = inputShapeInfo[3];
            std::cout << "input format: " << ch << "x" << input_h << "x" << input_w << std::endl;
        }

        // 获得输出信息 多输出
        for (int i = 0; i < output_nodes_num; i++)
        {
            auto output_name = session_.GetOutputNameAllocated(i, allocator_);
            output_node_names_.push_back(output_name.get());
            auto outShapeInfo = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            int ch = outShapeInfo[1];
            int output_h = outShapeInfo[2];
            int output_w = outShapeInfo[3];
            std::cout << "output format: " << ch << "x" << output_h << "x" << output_w << std::endl;
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    // 加载 ONNX 模型路径
    const char *model_path = "/home/dyf/project/lidar_camera_perception_ws/src/onnx_runtime/data/yolov5s.onnx";

    // 初始化 ONNX Runtime
    std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ImageRecognition");
    std::shared_ptr<Ort::SessionOptions> session_options = std::make_shared<Ort::SessionOptions>();
    rclcpp::spin(std::make_shared<onnx_runtime_inferencing>(env, model_path, session_options));
    rclcpp::shutdown();
    return 0;
}