#pragma once

#define _USE_MATH_DEFINES
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xutils.hpp>
#include <xtensor/xlayout.hpp>
#include <xtensor-fftw/basic.hpp>
#include <xtensor-fftw/helper.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <opencv2/core.hpp>

#include "utils.hpp"

namespace translation_estimation
{
    void Rearrange(cv::Mat& out)
    {
        int isXodd = out.cols % 2 == 1;
        int isYodd = out.rows % 2 == 1;
        int xMid = out.cols >> 1;
        int yMid = out.rows >> 1;
        cv::Mat q0(out, cv::Rect(0, 0, xMid + isXodd, yMid + isYodd));
        cv::Mat q1(out, cv::Rect(xMid + isXodd, 0, xMid, yMid + isYodd));
        cv::Mat q2(out, cv::Rect(0, yMid + isYodd, xMid + isXodd, yMid));
        cv::Mat q3(out, cv::Rect(xMid + isXodd, yMid + isYodd, xMid, yMid));
        //std::println("q0: {}\nq1: {}\nq2: {}\nq3: {}", q0, q1, q2, q3);

        if (!(isXodd || isYodd))
        {
            cv::Mat tmp;
            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);

            q1.copyTo(tmp);
            q2.copyTo(q1);
            tmp.copyTo(q2);
        }
        else
        {
            cv::Mat tmp0, tmp1, tmp2, tmp3;
            q0.copyTo(tmp0);
            q1.copyTo(tmp1);
            q2.copyTo(tmp2);
            q3.copyTo(tmp3);

            tmp0.copyTo(out(cv::Rect(xMid, yMid, xMid + isXodd, yMid + isYodd)));
            tmp3.copyTo(out(cv::Rect(0, 0, xMid, yMid)));

            tmp1.copyTo(out(cv::Rect(0, yMid, xMid, yMid + isYodd)));
            tmp2.copyTo(out(cv::Rect(xMid, 0, xMid + isXodd, yMid)));

            // to keep aligned with `image_registration.correlate2d`
            if (isYodd)
            {
                cv::Mat col0;
                out.col(0).copyTo(col0);
                out.colRange(1, out.cols).copyTo(out.colRange(0, out.cols - 1));
                col0.copyTo(out.col(out.cols - 1));
            }
            if (isXodd)
            {
                cv::Mat row0;
                out.row(0).copyTo(row0);
                out.rowRange(1, out.rows).copyTo(out.rowRange(0, out.rows - 1));
                row0.copyTo(out.row(out.rows - 1));
            }
        }
    }

    cv::Mat XCorrelation(cv::Mat const& I, cv::Mat const& I1)
    {
        cv::Mat fft1;
        cv::Mat fft2;

        cv::dft(I, fft1);
        cv::dft(I1, fft2);

        cv::mulSpectrums(fft1, fft2, fft1, 0, true);
        cv::idft(fft1, fft1, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
        Rearrange(fft1);
        return fft1;
    }

    // Helper function for 2D correlation using FFT
    template <typename T> xt::xarray<T> correlate2d(const xt::xarray<T>& img1, const xt::xarray<T>& img2)
    {
        // Determine output size (full correlation)
        auto shape1 = img1.shape();
        auto shape2 = img2.shape();

        auto img1_c = img1;
        auto img2_c = img2;

        // Perform correlation
        cv::Mat mat1(shape1[0], shape1[1], std::is_same_v<T, double> ? CV_64F : CV_32F, const_cast<T*>(img1_c.data()));
        cv::Mat mat2(shape2[0], shape2[1], std::is_same_v<T, double> ? CV_64F : CV_32F, const_cast<T*>(img2_c.data()));

        // FIXME: slightly precision diff compared with `image_registration.correlate2d`
        auto corr_mat = XCorrelation(mat1, mat2);

        // Create a copy to ensure memory safety
        xt::xarray<T> result = xt::eval(xt::view(
            xt::adapt(corr_mat.ptr<T>(), corr_mat.total(), xt::no_ownership(),
                      std::vector<size_t>{static_cast<size_t>(corr_mat.rows), static_cast<size_t>(corr_mat.cols)}),
            xt::all(), xt::all()));
        //std::println("result: \n{}", result);

        return result;
    }

    // Main chi2n_map function
    template <typename T>
    std::tuple<xt::xarray<T>, T, xt::xarray<T>, xt::xarray<T>> chi2n_map(const xt::xarray<T>& im1,
                                                                         const xt::xarray<T>& im2,
                                                                         bool zeromean = false, bool return_all = true,
                                                                         bool reduced = false)
    {
        if (im1.shape()[0] != im2.shape()[0] || im1.shape()[1] != im2.shape()[1])
            throw std::invalid_argument("Images must have same shape.");

        auto work_im1 = im1;
        auto work_im2 = im2;

        if (zeromean)
        {
            work_im1 -= xt::mean(work_im1)();
            work_im2 -= xt::mean(work_im2)();
        }
        work_im1 = xt::nan_to_num(work_im1);
        work_im2 = xt::nan_to_num(work_im2);
        //std::println("work_im1: {}", work_im1);
        //std::println("work_im2: {}", work_im2);

        // Error is a scalar: simply sum im1^2 / err^2
        auto err_val = 1.0;
        auto err_sq_scalar = err_val * err_val;
        auto term3 = xt::sum(work_im1 * work_im1 / err_sq_scalar)();
        //std::println("term3: {}", term3);

        // Calculate term1: sum of (im2^2 / err^2)
        auto term1 = xt::sum(work_im2 * work_im2 / err_sq_scalar)();
        //std::println("term1: {}", term1);

        // Calculate term2: -2 * correlation of (im1) with (im2/err^2)
        xt::xarray<T> term2 = -2 * correlate2d<T>(work_im1, work_im2 / err_sq_scalar);
        //std::println("term2: {}", term2);

        // Calculate chi2: term1 + term2 + term3
        xt::xarray<T> chi2 = term1 + term2 + term3;

        // Apply reduction if requested
        if (reduced) chi2 = chi2 / T(work_im2.size() - 2);

        if (return_all)
            return std::make_tuple(chi2, term1, term2, term3);
        else
            return std::make_tuple(chi2, T(0), xt::xarray<T>{}, xt::xarray<T>{});
    }

    // (-ex,+ex,-ey,+ey) where ex/ey are the x and y errors
    template <typename T>
    std::array<T, 4> chi2map_to_errors(xt::xarray<T> const& chi2_map, double zoom_factor = 1., int nsigma = 1)
    {
        auto sigma_to_chi2 = [](int x) {
            static constexpr double chi2[]{0.,
                                           2.295748928898636,
                                           6.180074306244173,
                                           11.829158081900795,
                                           19.333908611934685,
                                           28.743702426935496,
                                           40.08724353908164,
                                           53.38232525007259};
            assert(x >= 1); // x must be >= 1
            return x < 8 ? chi2[x] : 1.59358435 * std::pow(x, 1.80468278);
        };

        // Create coordinate grids (centered on image center)
        auto chi2_map_shape = chi2_map.shape();
        size_t height = chi2_map_shape[0];
        size_t width = chi2_map_shape[1];
        T height_center = height / 2.0;
        T width_center = width / 2.0;
        T zoom_inv = 1.0 / zoom_factor;

        // Create yy and xx coordinate grids
        xt::xarray<T> yy = xt::zeros<T>(chi2_map_shape);
        xt::xarray<T> xx = xt::zeros<T>(chi2_map_shape);
        for (size_t i = 0; i < height; ++i)
        {
            for (size_t j = 0; j < width; ++j)
            {
                yy(i, j) = (static_cast<T>(i) - height_center) * zoom_inv;
                xx(i, j) = (static_cast<T>(j) - width_center) * zoom_inv;
            }
        }
        //std::println("yy: {}", yy);
        //std::println("xx: {}", xx);

        // Center coordinates at minimum position
        auto min_idx = xt::argmin(chi2_map)();
        auto xcen = xx.flat(min_idx);
        auto ycen = yy.flat(min_idx);
        std::println("xcen: {}, ycen: {}", xcen, ycen);

        // Compute delta-chi² and find region within nsigma
        auto chi2_min = chi2_map(min_idx);
        // FIXME: chi2_min is slightly different from image_registration version due to precision diff in correlate2d
        std::println("min_idx: {}, chi2_min: {}", min_idx, chi2_min);
        xt::xarray<T> deltachi2 = chi2_map - chi2_min;
        //std::println("deltachi2: {}", deltachi2);
        auto sigma1_area = deltachi2 < static_cast<T>(sigma_to_chi2(nsigma));
        auto x_sigma1 = xt::filter(xx, sigma1_area);
        auto y_sigma1 = xt::filter(yy, sigma1_area);
        std::println("x_sigma1: {}, y_sigma1: {}", x_sigma1, y_sigma1);

        // Compute error bounds
        T errx_low = xcen - xt::amin(x_sigma1)();
        T errx_high = xt::amax(x_sigma1)() - xcen;
        T erry_low = ycen - xt::amin(y_sigma1)();
        T erry_high = xt::amax(y_sigma1)() - ycen;

        return {errx_low, errx_high, erry_low, erry_high};
    }

    template <typename T> xt::xarray<T> fourier_interp2d(xt::xarray<T> const& data, xt::xarray<T> const& outinds)
    {
        if (data.dimension() != 2) throw std::invalid_argument("Data must be 2D");
        if (outinds.dimension() != 3 || outinds.shape()[0] != 2)
            throw std::invalid_argument("3D outinds must have shape [2, height, width]");

        // Compute inverse FFT of input data
        xt::xarray<std::complex<T>> data_complex = xt::cast<std::complex<T>>(data);
        auto imfft = xt::fftw::ifft2(data_complex);

        // Determine how outinds is structured and process accordingly
        xt::xarray<T> outind_y = xt::view(outinds, 0, xt::all(), xt::all());
        xt::xarray<T> outind_x = xt::view(outinds, 1, xt::all(), xt::all());

        // Compute frequency arrays
        auto freqY = xt::fftw::fftfreq<T>(data.shape()[0]);
        auto freqX = xt::fftw::fftfreq<T>(data.shape()[1]);

        auto y_slice = xt::view(outind_y, xt::all(), 0); // shape: (height,) - take first column
        // freqY[np.newaxis,:] * outinds[0,:,0][:,np.newaxis]
        xt::xarray<T> expanded_freqY = xt::expand_dims(freqY, 0);     // shape: (1, height)
        xt::xarray<T> expanded_y_slice = xt::expand_dims(y_slice, 1); // shape: (height, 1)
        auto indsY = expanded_freqY * expanded_y_slice;               // shape: (height, height)
        xt::xarray<std::complex<T>> kerny = xt::exp(std::complex<T>(0, -2 * M_PI) * indsY);

        auto x_slice = xt::row(outind_x, 0); // shape: (width,) - take first row
        // freqX[:,np.newaxis] * outinds[1,0,:][np.newaxis,:]
        xt::xarray<T> expanded_freqX = xt::expand_dims(freqX, 1);     // shape: (width, 1)
        xt::xarray<T> expanded_x_slice = xt::expand_dims(x_slice, 0); // shape: (1, width)
        auto indsX = expanded_freqX * expanded_x_slice;               // shape: (width, width)
        xt::xarray<std::complex<T>> kernx = xt::exp(std::complex<T>(0, -2 * M_PI) * indsX);

        // Perform matrix multiplication: result = kerny @ imfft @ kernx
        auto result_complex = xt::linalg::dot(xt::linalg::dot(kerny, imfft), kernx);

        return xt::real(result_complex);
    }

    template <typename T>
    std::pair<xt::xarray<T>, xt::xarray<T>> zoom_on_pixel(xt::xarray<T> const& inp, std::array<T, 2> coordinates,
                                                          double usfac = 1.0)
    {
        auto outshape = inp.shape();
        xt::xarray<double> outarr = xt::zeros<double>(std::vector<std::size_t>{2, outshape[0], outshape[1]});
        //std::println("outarr shape: {}", outarr.shape());
        xt::xarray<double> r_coords =
            xt::linspace<double>(coordinates[0] - (outshape[0] - 1.) / usfac / 2.0,
                                 coordinates[0] + (outshape[0] - 1.) / usfac / 2.0, outshape[0]);
        //std::println("r_coords: {}", r_coords);
        xt::xarray<double> c_coords =
            xt::linspace<double>(coordinates[1] - (outshape[1] - 1.) / usfac / 2.0,
                                 coordinates[1] + (outshape[1] - 1.) / usfac / 2.0, outshape[1]);
        //std::println("c_coords: {}", c_coords);
        xt::view(outarr, 0, xt::all(), xt::all()) =
            xt::broadcast(xt::view(r_coords, xt::all(), xt::newaxis()), {outshape[0], outshape[1]});
        xt::view(outarr, 1, xt::all(), xt::all()) =
            xt::broadcast(xt::view(c_coords, xt::newaxis(), xt::all()), {outshape[0], outshape[1]});
        //std::println("outarr: {}", outarr);

        auto fid2 = fourier_interp2d(inp, outarr);
        //std::println("fid2: {}", fid2);
        return std::make_pair(outarr, fid2);
    }

    template <typename T>
    std::pair<xt::xarray<T>, xt::xarray<T>> zoom(xt::xarray<T> const& inp, std::array<T, 2> offsets, double usfac = 1.0)
    {
        auto shape = inp.shape();
        T two = 2.0;
        std::array<T, 2> middlepix{(shape[0] - 1) / two + offsets[0], (shape[1] - 1) / two + offsets[1]};
        //std::println("middlepix: {}", middlepix);
        return zoom_on_pixel(inp, middlepix, usfac);
    }

    // <<-xshift_corr, -yshift_corr>, <errx, erry>, chi2_ups>
    template <typename T>
    std::tuple<std::pair<T, T>, std::pair<T, T>, xt::xarray<T>> chi2_shift(
        xt::xarray<T> const& img1, xt::xarray<T> const& img2, double upsample_factor = 2., bool zeromean = true,
        bool return_error = false, bool return_chi2array = false, bool verbose = false)
    {
        // Call chi2n_map to get the chi2 map
        auto [chi2_map, term1, term2, term3] = chi2n_map(img1, img2, zeromean, zeromean, false);

        // Find minimum chi2 position
        auto min_pos = xt::unravel_index(xt::argmin(chi2_map)(), chi2_map.shape());
        //std::println("min_pos: {}", min_pos);

        auto ymax = min_pos[0];
        auto xmax = min_pos[1];
        auto ylen = img1.shape()[0];
        auto xlen = img1.shape()[1];
        //std::println("ylen/xlen = {},{}", ylen, xlen);

        // This is the center pixel - it's an integer pixel ID (not the center coordinate)
        auto ycen = ylen / 2. - (ylen % 2 == 0 ? 1. : 0.5);
        auto xcen = xlen / 2. - (xlen % 2 == 0 ? 1. : 0.5);
        //std::println("ycen/xcen = {},{}", ycen, xcen);

        // Original shift calculation
        std::array<double, 2> shift{ymax - ycen, xmax - xcen}; // shift img2 by these numbers to get img1
        std::println("Coarse xmax/ymax = {},{}, for offset {},{}", xmax, ymax, shift[1], shift[0]);

        // Sub-pixel zoom-in
        auto [shifts_correction, chi2_ups] = zoom(chi2_map, shift, upsample_factor);
        //std::println("shifts_correction: {}", shifts_correction);
        //std::println("chi2_ups: {}", chi2_ups);

        // deltachi2 is not reduced deltachi2
        auto chi2_up_min = xt::amin(chi2_ups)();
        auto deltachi2_ups = (chi2_ups - chi2_up_min);
        if (verbose)
            std::print("Minimum chi2_ups: {}   Max delta-chi2 (highres): {}  Min delta-chi2 (highres): {}", chi2_up_min,
                       xt::amax(deltachi2_ups)(), xt::amin(xt::filter(deltachi2_ups, deltachi2_ups > 0))());

        auto yshifts_corrections = xt::view(shifts_correction, 0, xt::all(), xt::all());
        auto xshifts_corrections = xt::view(shifts_correction, 1, xt::all(), xt::all());
        auto chi2_ups_min_idx = xt::argmin(chi2_ups)();
        T yshift_corr = yshifts_corrections.flat(chi2_ups_min_idx) - ycen;
        T xshift_corr = xshifts_corrections.flat(chi2_ups_min_idx) - xcen;
        //std::println("Shift correction: {},{}", xshift_corr, yshift_corr);

        if (return_error && return_chi2array)
        {
            auto [errx_low, errx_high, erry_low, erry_high] = chi2map_to_errors(chi2_ups, upsample_factor);
            //std::println("Error x: [{}, {}]", errx_low, errx_high);
            //std::println("Error y: [{}, {}]", erry_low, erry_high);
            return std::make_tuple(std::make_pair(-xshift_corr, -yshift_corr),
                                   std::make_pair((errx_low + errx_high) / 2., (erry_low + erry_high) / 2.), chi2_ups);
        }
        else if (return_error)
        {
            auto [errx_low, errx_high, erry_low, erry_high] = chi2map_to_errors(chi2_ups, upsample_factor);
            return std::make_tuple(std::make_pair(-xshift_corr, -yshift_corr),
                                   std::make_pair((errx_low + errx_high) / 2., (erry_low + erry_high) / 2.),
                                   xt::xarray<T>{});
        }
        else if (return_chi2array)
            return std::make_tuple(std::make_pair(-xshift_corr, -yshift_corr), std::make_pair(T{}, T{}), chi2_ups);
        else
            return std::make_tuple(std::make_pair(-xshift_corr, -yshift_corr), std::make_pair(T{}, T{}),
                                   xt::xarray<T>{});
    }
} // namespace translation_estimation