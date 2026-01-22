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
    // Pure xtensor 2D cross-correlation using FFT
    template <typename T> xt::xarray<T> correlate2d(const xt::xarray<T>& img1, const xt::xarray<T>& img2)
    {
        auto shape = img1.shape();

        // Convert to complex and compute FFTs
        xt::xarray<std::complex<T>> img1_complex = xt::cast<std::complex<T>>(img1);
        xt::xarray<std::complex<T>> img2_complex = xt::cast<std::complex<T>>(img2);

        auto fft1 = xt::fftw::fft2(img1_complex);
        auto fft2 = xt::fftw::fft2(img2_complex);

        // Cross-correlation in frequency domain: FFT1 * conj(FFT2)
        xt::xarray<std::complex<T>> fft_corr = fft1 * xt::conj(fft2);

        // Inverse FFT
        xt::xarray<std::complex<T>> corr_complex = xt::fftw::ifft2(fft_corr);
        xt::xarray<T> corr = xt::real(corr_complex);

        // Fftshift to center the zero-lag correlation
        xt::xarray<T> shifted_corr = xt::roll(xt::roll(corr, shape[0] / 2, 0), shape[1] / 2, 1);

        int isYodd = shape[0] % 2 == 1;
        int isXodd = shape[1] % 2 == 1;
        // to keep aligned with `image_registration.correlate2d`
        if (isYodd) return xt::roll(shifted_corr, -1, 1);
        if (isXodd) return xt::roll(shifted_corr, -1, 0);

        return shifted_corr;
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

        // Error is a scalar: simply sum im1^2 / err^2
        auto err_val = 1.0;
        auto err_sq_scalar = err_val * err_val;
        auto term3 = xt::sum(work_im1 * work_im1 / err_sq_scalar)();

        // Calculate term1: sum of (im2^2 / err^2)
        auto term1 = xt::sum(work_im2 * work_im2 / err_sq_scalar)();

        // Calculate term2: -2 * correlation of (im1) with (im2/err^2)
        xt::xarray<T> term2 = -2 * correlate2d<T>(work_im1, work_im2 / err_sq_scalar);

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

        // Center coordinates at minimum position
        auto min_idx = xt::argmin(chi2_map)();
        auto xcen = xx.flat(min_idx);
        auto ycen = yy.flat(min_idx);

        // Compute delta-chi² and find region within nsigma
        auto chi2_min = chi2_map(min_idx);
        // FIXME: chi2_min is slightly different from image_registration version due to precision diff in correlate2d
        xt::xarray<T> deltachi2 = chi2_map - chi2_min;
        auto sigma1_area = deltachi2 < static_cast<T>(sigma_to_chi2(nsigma));
        auto x_sigma1 = xt::filter(xx, sigma1_area);
        auto y_sigma1 = xt::filter(yy, sigma1_area);

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

        auto y_slice = xt::view(outind_y, xt::all(), 0);              // shape: (height,) - take first column
        xt::xarray<T> expanded_freqY = xt::expand_dims(freqY, 0);     // shape: (1, height)
        xt::xarray<T> expanded_y_slice = xt::expand_dims(y_slice, 1); // shape: (height, 1)
        auto indsY = expanded_freqY * expanded_y_slice;               // shape: (height, height)
        xt::xarray<std::complex<T>> kerny = xt::exp(std::complex<T>(0, -2 * M_PI) * indsY);

        auto x_slice = xt::row(outind_x, 0);                          // shape: (width,) - take first row
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
        xt::xarray<double> r_coords =
            xt::linspace<double>(coordinates[0] - (outshape[0] - 1.) / usfac / 2.0,
                                 coordinates[0] + (outshape[0] - 1.) / usfac / 2.0, outshape[0]);
        xt::xarray<double> c_coords =
            xt::linspace<double>(coordinates[1] - (outshape[1] - 1.) / usfac / 2.0,
                                 coordinates[1] + (outshape[1] - 1.) / usfac / 2.0, outshape[1]);
        xt::view(outarr, 0, xt::all(), xt::all()) =
            xt::broadcast(xt::view(r_coords, xt::all(), xt::newaxis()), {outshape[0], outshape[1]});
        xt::view(outarr, 1, xt::all(), xt::all()) =
            xt::broadcast(xt::view(c_coords, xt::newaxis(), xt::all()), {outshape[0], outshape[1]});

        auto fid2 = fourier_interp2d(inp, outarr);
        return std::make_pair(outarr, fid2);
    }

    template <typename T>
    std::pair<xt::xarray<T>, xt::xarray<T>> zoom(xt::xarray<T> const& inp, std::array<T, 2> offsets, double usfac = 1.0)
    {
        auto shape = inp.shape();
        T two = 2.0;
        std::array<T, 2> middlepix{(shape[0] - 1) / two + offsets[0], (shape[1] - 1) / two + offsets[1]};
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

        auto ymax = min_pos[0];
        auto xmax = min_pos[1];
        auto ylen = img1.shape()[0];
        auto xlen = img1.shape()[1];

        // This is the center pixel - it's an integer pixel ID (not the center coordinate)
        auto ycen = ylen / 2. - (ylen % 2 == 0 ? 1. : 0.5);
        auto xcen = xlen / 2. - (xlen % 2 == 0 ? 1. : 0.5);

        // Original shift calculation
        std::array<double, 2> shift{ymax - ycen, xmax - xcen}; // shift img2 by these numbers to get img1
        if (verbose) std::println("Coarse xmax/ymax = {},{}, for offset {},{}", xmax, ymax, shift[1], shift[0]);

        // Sub-pixel zoom-in
        auto [shifts_correction, chi2_ups] = zoom(chi2_map, shift, upsample_factor);

        // deltachi2 is not reduced deltachi2
        auto chi2_up_min = xt::amin(chi2_ups)();
        auto deltachi2_ups = (chi2_ups - chi2_up_min);
        if (verbose)
            std::println("Minimum chi2_ups: {}   Max delta-chi2 (highres): {}  Min delta-chi2 (highres): {}",
                         chi2_up_min, xt::amax(deltachi2_ups)(),
                         xt::amin(xt::filter(deltachi2_ups, deltachi2_ups > 0))());

        auto yshifts_corrections = xt::view(shifts_correction, 0, xt::all(), xt::all());
        auto xshifts_corrections = xt::view(shifts_correction, 1, xt::all(), xt::all());
        auto chi2_ups_min_idx = xt::argmin(chi2_ups)();
        T yshift_corr = yshifts_corrections.flat(chi2_ups_min_idx) - ycen;
        T xshift_corr = xshifts_corrections.flat(chi2_ups_min_idx) - xcen;

        if (return_error && return_chi2array)
        {
            auto [errx_low, errx_high, erry_low, erry_high] = chi2map_to_errors(chi2_ups, upsample_factor);
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