#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <print>

namespace translation_estimation::utils
{
    template <typename T> torch::Tensor cv_image_to_torch_tensor(const cv::Mat& image)
    {
        return torch::from_blob(image.data, {image.rows, image.cols},
                                image.depth() == CV_64F ? torch::kFloat64 : torch::kFloat32)
            .clone();
    }

    template <typename T> cv::Mat torch_tensor_to_cv_image(const torch::Tensor& tensor)
    {
        // Ensure tensor is on CPU and contiguous
        torch::Tensor cpuTensor = tensor.detach().to(torch::kCPU).contiguous();

        int height = cpuTensor.size(0);
        int width = cpuTensor.size(1);

        return cv::Mat(height, width, std::is_same_v<T, double> ? CV_64F : CV_32F, cpuTensor.data_ptr()).clone();
    }

    template <typename T> torch::Tensor load_image_to_torch(std::string const& filepath)
    {
        cv::Mat img = cv::imread(filepath, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
        if (img.empty()) throw std::runtime_error("Could not load image: " + filepath);

        if (img.channels() != 1)
        {
            std::println("Image has more than one channel. Convert it to grayscale.");
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }

        constexpr auto cv_t = std::is_same_v<T, double> ? CV_64F : CV_32F;
        if (img.type() != cv_t) img.convertTo(img, cv_t);
        if (!img.isContinuous()) img = img.clone();

        const size_t rows = img.rows;
        const size_t cols = img.cols;

        return cv_image_to_torch_tensor<T>(img);
    }
} // namespace translation_estimation::utils

template <> struct std::formatter<torch::Tensor>: std::formatter<std::string>
{
    auto format(torch::Tensor const& arr, format_context& ctx) const
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(8);
        oss << arr;
        return std::formatter<std::string>::format(oss.str(), ctx);
    }
};