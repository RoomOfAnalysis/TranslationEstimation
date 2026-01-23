#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-fftw/basic.hpp>
#include <xtensor-fftw/helper.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <tuple>
#include <algorithm>
#include <numeric>

#include "utils.hpp"

namespace translation_estimation
{

    namespace detail
    {
        // Upsampled DFT
        template <typename T>
        xt::xarray<std::complex<T>> _upsampled_dft(xt::xarray<std::complex<T>> data,
                                                   const std::vector<size_t>& upsampled_region_size, T upsample_factor,
                                                   const std::vector<T>& axis_offsets)
        {
            if (data.dimension() != 2) throw std::invalid_argument("data must be 2D");

            if (upsampled_region_size.size() != 2)
                throw std::invalid_argument("upsampled_region_size must match data dimensionality");

            if (axis_offsets.size() != 2) throw std::invalid_argument("axis_offsets must match data dimensionality");

            const T pi2 = 2.0 * M_PI;

            // Process dimensions in reverse order (last to first)
            // For 2D: process columns first (axis 1), then rows (axis 0)
            for (int ax = 1; ax >= 0; --ax)
            {
                size_t n_items = data.shape()[ax];
                size_t ups_size = upsampled_region_size[ax];
                T ax_offset = axis_offsets[ax];

                // Create frequency array: fftfreq(n_items, upsample_factor)
                xt::xarray<T> freq = xt::fftw::fftfreq(n_items, upsample_factor);

                // Create kernel: (ups_size, n_items)
                // kernel[i, j] = exp(-1i * 2π * (i - ax_offset) * freq[j])
                xt::xarray<T> kernel_indices = xt::arange<T>(0, static_cast<T>(ups_size)) - ax_offset;
                xt::xarray<T> expanded_indices = xt::expand_dims(kernel_indices, 1); // (ups_size, 1)
                xt::xarray<T> expanded_freq = xt::expand_dims(freq, 0);              // (1, n_items)

                auto phase = -pi2 * (expanded_indices * expanded_freq); // (ups_size, n_items)
                xt::xarray<std::complex<T>> kernel = xt::exp(std::complex<T>(0, 1) * phase);

                //// Apply kernel using tensordot along last axis
                //// Equivalent to: data = kernel @ data along axis -1
                //// For 2D data with axis=1: result[i, k] = sum_j kernel[i, j] * data[k, j]
                //// For 2D data with axis=0: result[i, k] = sum_j kernel[i, j] * data[j, k]

                if (ax == 1)
                { // Last axis (columns)
                    // data: (n_rows, n_cols), kernel: (ups_size, n_cols)
                    // result: (n_rows, ups_size)
                    data = xt::linalg::dot(data, xt::transpose(kernel));
                }
                else if (ax == 0)
                { // First axis (rows)
                    // data: (n_rows, ups_size), kernel: (ups_size, n_rows)
                    // result: (ups_size, ups_size)
                    data = xt::linalg::dot(kernel, data);
                }
            }

            return data;
        }

        // Compute registration error
        template <typename T> T _compute_error(std::complex<T> CCmax, T src_power, T target_power)
        {
            auto amp = src_power * target_power;
            if (amp < std::numeric_limits<T>::epsilon()) return std::numeric_limits<T>::quiet_NaN();
            auto error = 1.0 - CCmax * std::conj(CCmax) / amp;
            return std::sqrt(std::abs(error));
        }

        // Compute phase difference
        template <typename T> T constexpr _compute_phasediff(std::complex<T> CCmax)
        {
            return std::arg(CCmax);
        }
    } // namespace detail

    template <typename T>
    std::tuple<std::array<T, 2>, T, T> phase_cross_correlation(const xt::xarray<T>& reference_image,
                                                               const xt::xarray<T>& moving_image,
                                                               T upsample_factor = 1., bool normalization = true)
    {
        constexpr T eps = std::numeric_limits<float>::epsilon() * 100;

        // Check that images have the same shape
        if (reference_image.shape() != moving_image.shape())
            throw std::invalid_argument("Images must be the same shape");

        if (reference_image.dimension() != 2) throw std::invalid_argument("Images must be 2D");

        auto shape = reference_image.shape();

        xt::xarray<std::complex<T>> reference_image_complex = xt::cast<std::complex<T>>(reference_image);
        xt::xarray<std::complex<T>> moving_image_complex = xt::cast<std::complex<T>>(moving_image);

        // Transform to frequency domain
        // FIXME: slightly different results compared with `scipy.fft.fftn`
        xt::xarray<std::complex<T>> src_freq = xt::fftw::fft2(reference_image_complex);
        xt::xarray<std::complex<T>> target_freq = xt::fftw::fft2(moving_image_complex);
        //std::println("src_freq: {}", src_freq);
        //std::println("target_freq: {}", target_freq);

        // Compute cross-correlation in frequency domain
        xt::xarray<std::complex<T>> image_product = src_freq * xt::conj(target_freq);
        //std::println("image_product: {}", image_product);

        // Apply phase normalization
        if (normalization)
        {
            // Phase normalization: divide by magnitude
            xt::xarray<T> abs_product = xt::abs(image_product);
            // Avoid division by zero
            xt::filtration(abs_product, abs_product < eps) = eps;
            image_product /= abs_product;
        }
        //std::println("image_product: {}", image_product);

        // Inverse FFT to get cross-correlation
        xt::xarray<std::complex<T>> cross_correlation = xt::fftw::ifft2(image_product);

        // Locate maximum using absolute value
        auto maxima = xt::unravel_index(xt::argmax(xt::abs(cross_correlation))(), cross_correlation.shape());

        // Calculate shift from maximum location
        // Shift is relative to image center
        std::array<T, 2> shift_vec{};
        for (auto i = 0; i < 2; i++)
        {
            T shift_val = static_cast<T>(maxima[i]);
            T mid = shape[i] / 2;

            // Convert to [-size/2, size/2] range
            if (shift_val > mid) shift_val -= static_cast<T>(shape[i]);
            shift_vec[i] = shift_val;
        }

        // Calculate power/amplitude for error computation
        T src_power = xt::sum(xt::real(src_freq * xt::conj(src_freq)))();
        T target_power = xt::sum(xt::real(target_freq * xt::conj(target_freq)))();

        std::complex<T> CCmax;

        if (std::abs(upsample_factor - T{1}) < eps)
        {
            // No upsampling, use the value at the found maximum
            CCmax = cross_correlation(maxima[0], maxima[1]);
        }
        else
        {
            // Refine shift with upsampling

            // Round initial shift to nearest pixel on upsampled grid
            for (auto i = 0; i < 2; i++)
                shift_vec[i] = std::round(shift_vec[i] * upsample_factor) / upsample_factor;

            // Define upsampled region size
            size_t dft_size = static_cast<size_t>(std::ceil(upsample_factor * 1.5));

            // Compute upsampled DFT in neighborhood of peak
            T row_shift = shift_vec[0];
            T col_shift = shift_vec[1];

            T dftshift = dft_size / 2;

            xt::xarray<std::complex<T>> image_product_conj = xt::conj(image_product);
            xt::xarray<std::complex<T>> refined_corr = xt::conj(detail::_upsampled_dft(
                image_product_conj, {dft_size, dft_size}, upsample_factor,
                {dftshift - row_shift * upsample_factor, dftshift - col_shift * upsample_factor}));
            //std::println("refined_corr: {}", refined_corr);

            // Find maximum in refined correlation
            auto refined_maxima = xt::unravel_index(xt::argmax(xt::abs(refined_corr))(), refined_corr.shape());

            // Update shift with refined values
            T refined_row = static_cast<T>(refined_maxima[0]) - dftshift;
            T refined_col = static_cast<T>(refined_maxima[1]) - dftshift;

            shift_vec[0] = row_shift + refined_row / upsample_factor;
            shift_vec[1] = col_shift + refined_col / upsample_factor;

            // Get the refined correlation maximum
            CCmax = refined_corr(refined_maxima[0], refined_maxima[1]);
        }

        // Adjust shifts for single-element dimensions
        for (auto dim = 0; dim < 2; dim++)
            if (shape[dim] == 1) shift_vec[dim] = 0;

        //std::println("CCmax: {} {}", CCmax.real(), CCmax.imag());

        T error = detail::_compute_error(CCmax, src_power, target_power);
        T phasediff = detail::_compute_phasediff(CCmax);

        return std::make_tuple(shift_vec, error, phasediff);
    }
} // namespace translation_estimation