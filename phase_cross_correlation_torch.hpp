#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <torch/torch.h>

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace translation_estimation
{

    namespace detail
    {

        // -----------------------------------------------------------------------
        // Upsampled DFT (2‑D) – returns a 2D complex tensor
        // -----------------------------------------------------------------------
        inline torch::Tensor _upsampled_dft(const torch::Tensor& data, // complex 2D
                                            const std::vector<int64_t>& upsampled_region_size, double upsample_factor,
                                            const std::vector<double>& axis_offsets)
        {
            // data must be 2D complex
            TORCH_CHECK(data.dim() == 2, "data must be 2D");
            TORCH_CHECK(upsampled_region_size.size() == 2, "upsampled_region_size must have size 2");
            TORCH_CHECK(axis_offsets.size() == 2, "axis_offsets must have size 2");

            const double pi2 = 2.0 * M_PI;
            auto work = data.clone();

            // Process dimensions in reverse order (last to first)
            for (int ax = 1; ax >= 0; --ax)
            {
                int64_t n_items = work.size(ax);
                int64_t ups_size = upsampled_region_size[ax];
                double ax_offset = axis_offsets[ax];

                // Frequency vector with spacing = upsample_factor
                auto freq =
                    torch::fft::fftfreq(n_items, upsample_factor, torch::dtype(torch::kFloat64).device(work.device()));

                // Kernel indices: [0 .. ups_size-1] - offset
                auto kernel_indices = torch::arange(0, ups_size, torch::dtype(torch::kFloat64).device(work.device()))
                                          .sub(ax_offset); // (ups_size,)
                // Phase: -2π * indices * freq
                auto inds = kernel_indices.view({-1, 1}) * freq.view({1, -1});             // (ups_size, n_items)
                auto phase = -pi2 * inds;                                                  // real tensor
                auto kernel = torch::exp(torch::complex(torch::zeros_like(phase), phase)); // (ups_size, n_items)

                // Apply kernel via matrix multiplication
                if (ax == 1)
                { // last axis (columns)
                    // data : (n_rows, n_cols)  ->  matmul(data, kernel^T) -> (n_rows, ups_size)
                    work = torch::matmul(work, kernel.transpose(0, 1));
                }
                else
                { // ax == 0 (rows)
                    // data : (n_rows, ups_size)  ->  matmul(kernel, data) -> (ups_size, ups_size)
                    work = torch::matmul(kernel, work);
                }
            }
            return work;
        }

        // -----------------------------------------------------------------------
        // Compute registration error from CCmax and image powers
        // -----------------------------------------------------------------------
        template <typename T> T _compute_error(std::complex<T> CCmax, T src_power, T target_power)
        {
            T amp = src_power * target_power;
            if (amp < std::numeric_limits<T>::epsilon()) return std::numeric_limits<T>::quiet_NaN();
            auto val = std::complex<T>(1.0, 0.0) - (CCmax * std::conj(CCmax)) / amp;
            return std::sqrt(std::abs(val));
        }

        // -----------------------------------------------------------------------
        // Phase difference from CCmax
        // -----------------------------------------------------------------------
        template <typename T> T _compute_phasediff(std::complex<T> CCmax)
        {
            return std::arg(CCmax);
        }

    } // namespace detail

    // -----------------------------------------------------------------------
    // Phase‑based cross‑correlation with optional upsampling and normalisation
    // -----------------------------------------------------------------------
    template <typename T = double>
    std::tuple<std::array<T, 2>, T, T> phase_cross_correlation(const torch::Tensor& reference_image,
                                                               const torch::Tensor& moving_image,
                                                               T upsample_factor = 1.0, bool normalization = true)
    {
        constexpr T eps = static_cast<T>(std::numeric_limits<float>::epsilon()) * 100;

        TORCH_CHECK(reference_image.sizes() == moving_image.sizes(), "Images must have the same shape");
        TORCH_CHECK(reference_image.dim() == 2, "Images must be 2D");

        auto shape = reference_image.sizes();
        auto rows = shape[0];
        auto cols = shape[1];

        // Convert to complex double for FFT
        auto ref_complex = reference_image.to(torch::kComplexDouble);
        auto mov_complex = moving_image.to(torch::kComplexDouble);

        // FFT
        auto src_freq = torch::fft::fft2(ref_complex);
        auto target_freq = torch::fft::fft2(mov_complex);

        // Cross‑power spectrum
        auto image_product = src_freq * torch::conj(target_freq);

        // Optional phase normalisation
        if (normalization)
        {
            auto abs_prod = torch::abs(image_product);
            abs_prod = torch::clamp(abs_prod, eps, std::numeric_limits<T>::max());
            image_product = image_product / abs_prod;
        }

        // Inverse FFT -> spatial cross‑correlation
        auto cross_correlation = torch::fft::ifft2(image_product);
        auto abs_cc = torch::abs(cross_correlation);

        // Find integer peak
        auto flat_abs = abs_cc.view({-1});
        auto max_idx = torch::argmax(flat_abs).template item<int64_t>();
        int64_t row_max = max_idx / cols;
        int64_t col_max = max_idx % cols;

        // Convert to shift relative to image centre
        std::array<T, 2> shift_vec{};
        for (int i = 0; i < 2; ++i)
        {
            T shift_val = static_cast<T>(i == 0 ? row_max : col_max);
            T mid = static_cast<T>(shape[i]) / 2.0;
            if (shift_val > mid) shift_val -= static_cast<T>(shape[i]);
            shift_vec[i] = shift_val;
        }

        // Powers for error calculation
        T src_power = torch::sum(torch::real(src_freq * torch::conj(src_freq))).template item<T>();
        T target_power = torch::sum(torch::real(target_freq * torch::conj(target_freq))).template item<T>();

        std::complex<T> CCmax;

        if (std::abs(upsample_factor - T{1.0}) < eps)
        {
            // No upsampling – take value at the integer peak
            auto cc_val = cross_correlation.index({row_max, col_max});
            CCmax = std::complex<T>(torch::real(cc_val).template item<T>(), torch::imag(cc_val).template item<T>());
        }
        else
        {
            // Sub‑pixel refinement via upsampled DFT

            // Round initial shift to the nearest pixel on the upsampled grid
            for (int i = 0; i < 2; ++i)
                shift_vec[i] = std::round(shift_vec[i] * upsample_factor) / upsample_factor;

            int64_t dft_size = static_cast<int64_t>(std::ceil(upsample_factor * 1.5));
            double dftshift = dft_size / 2.0;

            std::vector<double> offsets = {dftshift - shift_vec[0] * upsample_factor,
                                           dftshift - shift_vec[1] * upsample_factor};

            // The original algorithm processes the conjugate of image_product
            // and conjugates the result to obtain the correct correlation.
            auto image_product_conj = torch::conj(image_product);
            auto refined_complex =
                detail::_upsampled_dft(image_product_conj, {dft_size, dft_size}, upsample_factor, offsets);
            auto refined_corr = torch::conj(refined_complex);

            // Locate maximum in the upsampled region
            auto refined_abs = torch::abs(refined_corr).view({-1});
            auto refined_max_idx = torch::argmax(refined_abs).template item<int64_t>();
            int64_t refined_row = refined_max_idx / dft_size;
            int64_t refined_col = refined_max_idx % dft_size;

            // Update shifts with refined values
            shift_vec[0] = shift_vec[0] + (static_cast<T>(refined_row) - dftshift) / upsample_factor;
            shift_vec[1] = shift_vec[1] + (static_cast<T>(refined_col) - dftshift) / upsample_factor;

            // Extract the complex value at the refined peak
            auto cc_peak = refined_corr.index({refined_row, refined_col});
            CCmax = std::complex<T>(torch::real(cc_peak).template item<T>(), torch::imag(cc_peak).template item<T>());
        }

        // Zero shift for singleton dimensions
        for (int i = 0; i < 2; ++i)
            if (shape[i] == 1) shift_vec[i] = T{0};

        T error = detail::_compute_error(CCmax, src_power, target_power);
        T phasediff = detail::_compute_phasediff(CCmax);

        return std::make_tuple(shift_vec, error, phasediff);
    }

} // namespace translation_estimation