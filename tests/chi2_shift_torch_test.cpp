#include "chi2_shift_torch.hpp"
#include "utils_torch.hpp"
#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <chrono>

int main(int argc, char* argv[])
{
    using namespace translation_estimation;
    using namespace translation_estimation::utils;

    argparse::ArgumentParser program("chi2_shift_test");
    program.add_argument("--image1").help("Path to image1").required().metavar("FILE");
    program.add_argument("--image2").help("Path to image2").required().metavar("FILE");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err)
    {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    auto image1 = load_image_to_torch<double>(program.get<std::string>("--image1"));
    auto image2 = load_image_to_torch<double>(program.get<std::string>("--image2"));

    // std::println("Image1: {}", image1);
    // std::println("Image2: {}", image2);

    auto corr = correlate2d(image1, image2);
    // std::println("Correlation: {}", corr);
    std::println("Shape: {}", corr.sizes());
    std::println("Mean: {}", torch::mean(corr));

    auto [chi2, term1, term2, term3] = chi2n_map(image1, image2, true, true, false);
    // std::println("chi2: {}", chi2);
    std::println("Mean of chi2: {}", torch::mean(chi2));
    std::println("term1: {}", term1);
    // std::println("term2: {}", term2);
    std::println("Mean of term2: {}", torch::mean(term2));
    std::println("term3: {}", term3);

    auto [hv_shifts, errs, chi2ups] = chi2_shift(image1, image2, 2.0, true, true, true, true);
    auto [h_shift, v_shift] = hv_shifts;
    std::println("h_shift: {}, v_shift: {}", h_shift, v_shift);
    std::println("errs: {}", errs);
    // std::println("chi2ups: {}", chi2ups);
    std::println("Mean of chi2ups: {}", torch::mean(chi2ups));

    auto corrected_image = shift2d(image2, -h_shift, -v_shift);
    // std::println("Corrected image: {}", corrected_image);
    std::println("Mean of corrected image: {}", torch::mean(corrected_image));

    auto start = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < 10; i++)
        if (i % 2 == 0)
            std::tie(hv_shifts, errs, chi2ups) = chi2_shift(image1, image2, 2.0, true, true, true, false);
        else
            std::tie(hv_shifts, errs, chi2ups) = chi2_shift(image2, image1, 2.0, true, true, true, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::println("Time taken: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    return 0;
}