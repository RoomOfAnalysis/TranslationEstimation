#include "phase_cross_correlation.hpp"
#include "utils.hpp"
#include <argparse/argparse.hpp>

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

    auto image1 = load_image_to_xtensor<double>(program.get<std::string>("--image1"));
    auto image2 = load_image_to_xtensor<double>(program.get<std::string>("--image2"));

    std::println("Image1: {}", image1);
    std::println("Image2: {}", image2);

    auto [shift, error, diff_phase] = phase_cross_correlation(image1, image2, 5.);
    std::println("Shift: {}", shift); // y, x
    std::println("Error: {}", error);
    std::println("Diff phase: {}", diff_phase);

    auto shape1 = image1.shape();
    auto shape2 = image2.shape();
    cv::Mat mat1(shape1[0], shape1[1], CV_64F, const_cast<double*>(image1.data()));
    cv::Mat mat2(shape2[0], shape2[1], CV_64F, const_cast<double*>(image2.data()));
    auto shift_cv = cv::phaseCorrelate(mat1, mat2); // x, y
    std::println("OpenCV Shift: ({}, {})", shift_cv.y, shift_cv.x);

    return 0;
}