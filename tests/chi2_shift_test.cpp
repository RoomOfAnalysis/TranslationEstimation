#include "chi2_shift.hpp"
#include <argparse/argparse.hpp>
#include <xtensor/xcsv.hpp>
#include <filesystem>
#include <fstream>

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

    auto image1 = load_image_to_xtensor<double>(program.get<std::string>("--image1"));
    auto image2 = load_image_to_xtensor<double>(program.get<std::string>("--image2"));

    std::println("Image1: {}", image1);
    std::println("Image2: {}", image2);

    auto corr = correlate2d(image1, image2);
    std::println("Correlation: {}", corr);
    std::println("Shape: {}", corr.shape());
    std::println("Mean: {}", xt::mean(corr)());

    auto [chi2, term1, term2, term3] = chi2n_map(image1, image2, true, true, false);
    std::println("chi2: {}", chi2);
    std::println("term1: {}", term1);
    std::println("term2: {}", term2);
    std::println("term3: {}", term3);

    //const auto chi2m{std::filesystem::current_path() / "chi2.txt"};
    ////if (std::FILE * stream{std::fopen(chi2m.string().c_str(), "w")})
    ////{
    ////    std::print(stream, "{}", chi2);
    ////    std::fclose(stream);
    ////}
    //std::ofstream stream{chi2m.string()};
    //stream << std::fixed << std::setprecision(16);
    //xt::dump_csv(stream, chi2);

    auto [hv_shifts, errs, chi2ups] = chi2_shift(image1, image2, 2.0, true, true, true, true);
    auto [h_shift, v_shift] = hv_shifts;
    std::println("h_shift: {}, v_shift: {}", h_shift, v_shift);
    std::println("errs: {}", errs);
    std::println("chi2ups: {}", chi2ups);

    return 0;
}