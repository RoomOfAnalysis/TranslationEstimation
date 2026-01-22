#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <opencv2/opencv.hpp>

#include <print>

namespace translation_estimation::utils
{
    template <typename T> cv::Mat load_image_to_mat(std::string const& filepath)
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

        return img;
    }

    template <typename T> xt::xarray<T> load_image_to_xtensor(std::string const& filepath)
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

        xt::xarray<T> result = xt::zeros<T>({rows, cols});
        std::memcpy(result.data(), img.ptr<T>(), rows * cols * sizeof(T));

        return result;
    }
} // namespace translation_estimation::utils

template <typename T> struct std::formatter<xt::xarray<T>>: std::formatter<std::string>
{
    auto format(xt::xarray<T> const& arr, format_context& ctx) const
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(8);
        oss << arr;
        return std::formatter<std::string>::format(oss.str(), ctx);
    }
};

template <> struct std::formatter<cv::Mat>: std::formatter<std::string>
{
    auto format(cv::Mat const& mat, format_context& ctx) const
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(8);
        /*oss << cv::format(mat, cv::Formatter::FMT_NUMPY);*/
        if (mat.type() == CV_32F)
        {
            auto arr = xt::adapt((float*)mat.data, mat.total(), xt::no_ownership(),
                                 std::vector<size_t>{static_cast<size_t>(mat.rows), static_cast<size_t>(mat.cols)});
            oss << arr;
        }
        else if (mat.type() == CV_64F)
        {
            auto arr = xt::adapt((double*)mat.data, mat.total(), xt::no_ownership(),
                                 std::vector<size_t>{static_cast<size_t>(mat.rows), static_cast<size_t>(mat.cols)});
            oss << arr;
        }
        else
            throw std::exception("Unsupported matrix type.");
        return std::formatter<std::string>::format(oss.str(), ctx);
    }
};