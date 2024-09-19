#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

using namespace std;
using namespace cv;

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

// 图像预处理函数
cv::Mat preprocess_image(const cv::Mat &img, const std::vector<int64_t> &input_shape)
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

// 加载标签映射
std::map<int, std::string> load_labels(const std::string &yaml_file)
{
    YAML::Node config = YAML::LoadFile(yaml_file);
    std::map<int, std::string> labels;
    for (const auto &item : config["names"])
    {
        labels[item.first.as<int>()] = item.second.as<std::string>();
    }
    return labels;
}

// 后处理，解析推理结果，提取边界框和类别
std::vector<std::tuple<cv::Rect, int>> postprocess(const std::vector<float> &boxes, const std::vector<float> &scores,
                                                   const std::vector<int> &class_ids, float conf_threshold)
{
    std::vector<std::tuple<cv::Rect, int>> output_boxes;
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > conf_threshold)
        {
            int x = static_cast<int>(boxes[i * 4 + 0]);
            int y = static_cast<int>(boxes[i * 4 + 1]);
            int width = static_cast<int>(boxes[i * 4 + 2] - x);
            int height = static_cast<int>(boxes[i * 4 + 3] - y);
            output_boxes.emplace_back(cv::Rect(x, y, width, height), class_ids[i]);
        }
    }
    return output_boxes;
}

// 绘制边界框和标签
void draw_boxes(cv::Mat &img, const std::vector<std::tuple<cv::Rect, int>> &boxes, const std::map<int, std::string> &labels)
{
    for (const auto &[box, class_id] : boxes)
    {
        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);
        std::string label = labels.at(class_id);
        cv::putText(img, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <onnx_model> <image_file> <coco_yaml>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string coco_yaml_path = argv[3];

    // 1. 加载 COCO 标签
    std::map<int, std::string> labels = load_labels(coco_yaml_path);

    // 2. 加载 ONNX 模型
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_inference");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    // 获取模型输入输出信息
    Ort::AllocatorWithDefaultOptions allocator;
    const Ort::Value &input_tensor_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> input_shape = input_tensor_shape.GetShape();

    // 3. 加载和预处理图像
    cv::Mat img = cv::imread(image_path);
    if (img.empty())
    {
        std::cerr << "Image not found: " << image_path << std::endl;
        return -1;
    }
    cv::Mat preprocessed_img = preprocess_image(img, input_shape);

    // 将图像数据转换为一维数组
    std::vector<float> input_tensor_values(preprocessed_img.begin<float>(), preprocessed_img.end<float>());

    // 4. 创建输入张量
    std::vector<int64_t> input_tensor_shape_vec = {1, 3, input_shape[2], input_shape[3]};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape_vec.data(), input_tensor_shape_vec.size());

    // 5. 推理执行
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &session.GetInputName(0, allocator), &input_tensor, 1, session.GetOutputName(0, allocator), 1);

    // 获取推理结果
    float *boxes = output_tensors[0].GetTensorMutableData<float>();
    float *scores = output_tensors[1].GetTensorMutableData<float>();
    int *class_ids = output_tensors[2].GetTensorMutableData<int>();

    // 6. 后处理，解析输出，获取边界框和类别
    float conf_threshold = 0.5;
    std::vector<std::tuple<cv::Rect, int>> detected_boxes = postprocess(std::vector<float>(boxes, boxes + 4), std::vector<float>(scores, scores + 1), std::vector<int>(class_ids, class_ids + 1), conf_threshold);

    // 7. 绘制边界框和标签
    draw_boxes(img, detected_boxes, labels);

    // 显示结果
    cv::imshow("Detected Image", img);
    cv::waitKey(0);

    return 0;
}
#include <iostream>
#include <opencv2/opencv.hpp>

// NMS 实现
std::vector<int> non_max_suppression(const std::vector<cv::Rect> &boxes, const std::vector<float> &scores, float iou_threshold)
{
    std::vector<int> indices;
    if (boxes.empty())
    {
        return indices;
    }

    // 排序，根据置信度从高到低
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0); // 初始化为 [0, 1, 2, ..., N-1]
    std::sort(order.begin(), order.end(), [&scores](int i1, int i2)
              { return scores[i1] > scores[i2]; });

    std::vector<bool> suppressed(boxes.size(), false);

    // 迭代边框，进行 NMS
    for (size_t i = 0; i < order.size(); ++i)
    {
        int idx = order[i];
        if (suppressed[idx])
        {
            continue;
        }
        indices.push_back(idx);
        for (size_t j = i + 1; j < order.size(); ++j)
        {
            int idx_j = order[j];
            if (suppressed[idx_j])
            {
                continue;
            }

            // 计算 IoU（Intersection over Union）
            float interArea = (boxes[idx] & boxes[idx_j]).area();
            float unionArea = boxes[idx].area() + boxes[idx_j].area() - interArea;
            float iou = interArea / unionArea;

            // 如果 IoU 大于阈值，抑制当前框
            if (iou > iou_threshold)
            {
                suppressed[idx_j] = true;
            }
        }
    }

    return indices;
}

// 绘制 NMS 处理后的边框
void draw_nms_boxes(cv::Mat &img, const std::vector<cv::Rect> &boxes, const std::vector<int> &nms_indices, const std::map<int, std::string> &labels, const std::vector<int> &class_ids)
{
    for (int idx : nms_indices)
    {
        cv::rectangle(img, boxes[idx], cv::Scalar(0, 255, 0), 2);
        std::string label = labels.at(class_ids[idx]);
        cv::putText(img, label, cv::Point(boxes[idx].x, boxes[idx].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
    }
}

int main()
{
    // 假设有一组边界框和对应的置信度得分
    std::vector<cv::Rect> boxes = {
        cv::Rect(100, 100, 50, 50),
        cv::Rect(105, 105, 50, 50),
        cv::Rect(200, 200, 50, 50),
    };
    std::vector<float> scores = {0.9, 0.8, 0.7};
    std::vector<int> class_ids = {0, 0, 1}; // 假设类 ID

    // 假设有标签
    std::map<int, std::string> labels = {
        {0, "person"},
        {1, "car"}};

    // 设定 IoU 阈值
    float iou_threshold = 0.4;

    // 运行 NMS
    std::vector<int> nms_indices = non_max_suppression(boxes, scores, iou_threshold);

    // 加载图像并绘制边界框
    cv::Mat img = cv::Mat::zeros(400, 400, CV_8UC3); // 空白图像
    draw_nms_boxes(img, boxes, nms_indices, labels, class_ids);

    // 显示图像
    cv::imshow("NMS Results", img);
    cv::waitKey(0);

    return 0;
}