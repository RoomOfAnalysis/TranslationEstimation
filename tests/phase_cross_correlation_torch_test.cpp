#include "phase_cross_correlation_torch.hpp"
#include "utils_torch.hpp"
#include <argparse/argparse.hpp>
#include <chrono>

int main(int argc, char* argv[])
{
    using namespace translation_estimation;
    using namespace translation_estimation::utils;

    argparse::ArgumentParser program("phase_cross_correlation_test");
    program.add_argument("--image1").help("Path to image1").required().metavar("FILE");
    program.add_argument("--image2").help("Path to image2").required().metavar("FILE");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err)
    {
        std::println("{}", err.what());
        std::exit(1);
    }

    auto image1 = load_image_to_torch<double>(program.get<std::string>("--image1"));
    auto image2 = load_image_to_torch<double>(program.get<std::string>("--image2"));

    // std::println("Image1: {}", image1);
    // std::println("Image2: {}", image2);

    auto [shift, error, diff_phase] = phase_cross_correlation(image1, image2, 5.);
    std::println("Shift: {}", shift); // y, x
    std::println("Error: {}", error);
    std::println("Diff phase: {}", diff_phase);

    auto shape1 = image1.sizes();
    auto shape2 = image2.sizes();
    auto mat1 = torch_tensor_to_cv_image<double>(image1);
    auto mat2 = torch_tensor_to_cv_image<double>(image2);
    auto shift_cv = cv::phaseCorrelate(mat1, mat2); // x, y
    std::println("OpenCV Shift: ({}, {})", shift_cv.y, shift_cv.x);

    auto start = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < 30; i++)
        if (i % 2 == 0)
            std::tie(shift, error, diff_phase) = phase_cross_correlation(image1, image2, 5.);
        else
            std::tie(shift, error, diff_phase) = phase_cross_correlation(image2, image1, 5.);
    auto end = std::chrono::high_resolution_clock::now();
    std::println("Time taken: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    return 0;
}