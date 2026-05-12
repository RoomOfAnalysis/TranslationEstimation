#pragma once

#include <torch/torch.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace translation_estimation
{

    // -----------------------------------------------------------------------
    // 2D cross‑correlation via FFT
    // -----------------------------------------------------------------------
    inline torch::Tensor correlate2d(const torch::Tensor& img1, const torch::Tensor& img2)
    {
        // Input must be real 2D tensors
        auto fft1 = torch::fft::fft2(img1.to(torch::kComplexDouble));
        auto fft2 = torch::fft::fft2(img2.to(torch::kComplexDouble));
        auto fft_corr = fft1 * torch::conj(fft2);
        auto corr_complex = torch::fft::ifft2(fft_corr);
        auto corr = torch::real(corr_complex);

        // Roll so that zero shift is at the centre
        int64_t shift0 = (img1.size(0) - 1) / 2;
        int64_t shift1 = (img1.size(1) - 1) / 2;
        corr = torch::roll(corr, {shift0, shift1}, {0, 1});
        return corr;
    }

    // -----------------------------------------------------------------------
    // χ² map (full map of test statistic for translation registration)
    // -----------------------------------------------------------------------
    inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> chi2n_map(const torch::Tensor& im1,
                                                                                            const torch::Tensor& im2,
                                                                                            bool zeromean = false,
                                                                                            bool return_all = true,
                                                                                            bool reduced = false)
    {
        if (im1.size(0) != im2.size(0) || im1.size(1) != im2.size(1))
            throw std::invalid_argument("Images must have the same shape.");

        auto work_im1 = im1.clone();
        auto work_im2 = im2.clone();

        if (zeromean)
        {
            work_im1 = work_im1 - torch::mean(work_im1);
            work_im2 = work_im2 - torch::mean(work_im2);
        }
        work_im1 = torch::nan_to_num(work_im1);
        work_im2 = torch::nan_to_num(work_im2);

        // Hard‑coded error of 1.0
        const double err_val = 1.0;
        const double err_sq_scalar = err_val * err_val;

        // term1: sum(im2^2)
        auto term1 = torch::sum(work_im2 * work_im2 / err_sq_scalar);
        // term3: sum(im1^2)
        auto term3 = torch::sum(work_im1 * work_im1 / err_sq_scalar);
        // term2: -2 * correlation(im1, im2)
        auto term2 = -2.0 * correlate2d(work_im1, work_im2 / err_sq_scalar);

        // χ² = term1 + term2 + term3 (scalars broadcast over the 2D map)
        auto chi2 = term1 + term2 + term3;

        if (reduced) chi2 = chi2 / (work_im2.numel() - 2);

        if (return_all)
            return {chi2, term1, term2, term3};
        else
            return {chi2, torch::tensor(0.0), torch::Tensor{}, torch::Tensor{}};
    }

    // -----------------------------------------------------------------------
    // Convert a χ² map to error bounds for a given confidence level
    // -----------------------------------------------------------------------
    template <typename T = double>
    std::array<T, 4> chi2map_to_errors(const torch::Tensor& chi2_map, double zoom_factor = 1.0, int nsigma = 1)
    {
        // Pre‑computed χ² values for 1 … 7 σ (2‑D)
        auto sigma_to_chi2 = [](int x) -> double {
            static constexpr double chi2[] = {0.,
                                              2.295748928898636,
                                              6.180074306244173,
                                              11.829158081900795,
                                              19.333908611934685,
                                              28.743702426935496,
                                              40.08724353908164,
                                              53.38232525007259};
            return x < 8 ? chi2[x] : 1.59358435 * std::pow(double(x), 1.80468278);
        };

        const auto shape = chi2_map.sizes();
        int64_t H = shape[0], W = shape[1];
        T height_center = H / 2.0;
        T width_center = W / 2.0;
        T zoom_inv = 1.0 / zoom_factor;

        // Coordinate grids centred on the image centre
        auto y_lin = torch::arange(H, chi2_map.options().dtype(torch::kFloat64));
        auto x_lin = torch::arange(W, chi2_map.options().dtype(torch::kFloat64));
        auto grids = torch::meshgrid({y_lin, x_lin}, "ij");
        auto yy = grids[0], xx = grids[1];
        yy = (yy - height_center) * zoom_inv;
        xx = (xx - width_center) * zoom_inv;

        // Global minimum position
        auto flat_map = chi2_map.view({-1});
        auto min_idx = torch::argmin(flat_map).item<int64_t>();
        int64_t ycen_idx = min_idx / W, xcen_idx = min_idx % W;

        T xcen = xx[ycen_idx][xcen_idx].item<T>();
        T ycen = yy[ycen_idx][xcen_idx].item<T>();
        T chi2_min = chi2_map[ycen_idx][xcen_idx].item<T>();

        // Region inside the desired confidence contour
        auto deltachi2 = chi2_map - chi2_min;
        auto mask = deltachi2 < sigma_to_chi2(nsigma);
        auto x_vals = xx.masked_select(mask);
        auto y_vals = yy.masked_select(mask);

        T errx_low = xcen - x_vals.min().item<T>();
        T errx_high = x_vals.max().item<T>() - xcen;
        T erry_low = ycen - y_vals.min().item<T>();
        T erry_high = y_vals.max().item<T>() - ycen;

        return {errx_low, errx_high, erry_low, erry_high};
    }

    // -----------------------------------------------------------------------
    // 2D Fourier interpolation (arbitrary coordinates)
    // -----------------------------------------------------------------------
    inline torch::Tensor fourier_interp2d(const torch::Tensor& data, const torch::Tensor& outinds)
    {
        if (data.dim() != 2) throw std::invalid_argument("Data must be 2D");
        if (outinds.dim() != 3 || outinds.size(0) != 2) throw std::invalid_argument("outinds must be [2, H, W]");

        // Convert to complex and go to the frequency domain with ifft2
        auto imfft = torch::fft::ifft2(data.to(torch::kComplexDouble));

        // Separate coordinate arrays
        auto outind_y = outinds[0]; // (H, W)
        auto outind_x = outinds[1]; // (H, W)

        int64_t H = data.size(0), W = data.size(1);
        auto freqY = torch::fft::fftfreq(H, 1.0, torch::dtype(torch::kFloat64));
        auto freqX = torch::fft::fftfreq(W, 1.0, torch::dtype(torch::kFloat64));

        // ----- Kernel in y ------------------------------------------------
        auto y_slice = outind_y.index({torch::indexing::Slice(), 0}); // (H,)
        auto indsY = freqY.view({1, -1}) * y_slice.view({-1, 1});     // (H, H)
        auto kerny = torch::exp(torch::complex(torch::zeros_like(indsY), -2.0 * M_PI * indsY));

        // ----- Kernel in x ------------------------------------------------
        auto x_slice = outind_x.index({0, torch::indexing::Slice()}); // (W,)
        auto indsX = freqX.view({-1, 1}) * x_slice.view({1, -1});     // (W, W)
        auto kernx = torch::exp(torch::complex(torch::zeros_like(indsX), -2.0 * M_PI * indsX));

        // ----- Matrix products (kerny @ imfft @ kernx) --------------------
        auto mid = torch::matmul(kerny, imfft);  // (H, W)
        auto result = torch::matmul(mid, kernx); // (H, W)

        return torch::real(result);
    }

    // -----------------------------------------------------------------------
    // Zoom in on a pixel using Fourier interpolation
    // -----------------------------------------------------------------------
    template <typename T = double>
    std::pair<torch::Tensor, torch::Tensor> zoom_on_pixel(const torch::Tensor& inp, std::array<T, 2> coordinates,
                                                          double usfac = 1.0)
    {
        auto outshape = inp.sizes();
        int64_t H = outshape[0], W = outshape[1];

        // Build the (2, H, W) array of target coordinates
        auto outarr = torch::zeros({2, H, W}, inp.options().dtype(torch::kFloat64));

        auto r_coords = torch::linspace(coordinates[0] - (H - 1.0) / usfac / 2.0,
                                        coordinates[0] + (H - 1.0) / usfac / 2.0, H, torch::dtype(torch::kFloat64));
        auto c_coords = torch::linspace(coordinates[1] - (W - 1.0) / usfac / 2.0,
                                        coordinates[1] + (W - 1.0) / usfac / 2.0, W, torch::dtype(torch::kFloat64));

        outarr[0] = r_coords.view({-1, 1}).expand({H, W});
        outarr[1] = c_coords.view({1, -1}).expand({H, W});

        auto fid2 = fourier_interp2d(inp, outarr);
        return {outarr, fid2};
    }

    // -----------------------------------------------------------------------
    // Zoom with a user‑provided offset
    // -----------------------------------------------------------------------
    template <typename T = double>
    std::pair<torch::Tensor, torch::Tensor> zoom(const torch::Tensor& inp, std::array<T, 2> offsets, double usfac = 1.0)
    {
        auto shape = inp.sizes();
        T two = 2.0;
        std::array<T, 2> middlepix{(shape[0] - 1) / two + offsets[0], (shape[1] - 1) / two + offsets[1]};
        return zoom_on_pixel(inp, middlepix, usfac);
    }

    // -----------------------------------------------------------------------
    // χ² shift estimation with sub‑pixel refinement
    // -----------------------------------------------------------------------
    template <typename T = double>
    std::tuple<std::pair<T, T>, std::pair<T, T>, torch::Tensor> chi2_shift(
        const torch::Tensor& img1, const torch::Tensor& img2, double upsample_factor = 2.0, bool zeromean = true,
        bool return_error = false, bool return_chi2array = false, bool verbose = false)
    {
        auto [chi2_map, term1, term2, term3] = chi2n_map(img1, img2, zeromean, true, false);

        int64_t H = chi2_map.size(0), W = chi2_map.size(1);
        auto min_idx = torch::argmin(chi2_map.view({-1})).item<int64_t>();
        int64_t ymax = min_idx / W;
        int64_t xmax = min_idx % W;

        T ylen = H, xlen = W;
        T ycen = ylen / 2.0 - (H % 2 == 0 ? 1.0 : 0.5);
        T xcen = xlen / 2.0 - (W % 2 == 0 ? 1.0 : 0.5);

        std::array<T, 2> shift{T(ymax) - ycen, T(xmax) - xcen};
        if (verbose)
            std::cout << "Coarse xmax/ymax = " << xmax << "," << ymax << " for offset " << shift[1] << "," << shift[0]
                      << std::endl;

        auto [shifts_correction, chi2_ups] = zoom(chi2_map, shift, upsample_factor);

        T chi2_up_min = torch::amin(chi2_ups).item<T>();
        auto deltachi2_ups = chi2_ups - chi2_up_min;
        if (verbose)
        {
            auto max_d = torch::amax(deltachi2_ups).item<T>();
            auto min_d = torch::amin(deltachi2_ups.masked_select(deltachi2_ups > 0));
            std::cout << "Minimum chi2_ups: " << chi2_up_min << "  Max delta-chi2: " << max_d
                      << "  Min delta-chi2 (highres): " << min_d.item<T>() << std::endl;
        }

        auto yshifts = shifts_correction[0]; // (H, W)
        auto xshifts = shifts_correction[1]; // (H, W)

        auto up_min_idx = torch::argmin(chi2_ups.view({-1})).item<int64_t>();
        T yshift_corr = yshifts.view({-1})[up_min_idx].item<T>() - ycen;
        T xshift_corr = xshifts.view({-1})[up_min_idx].item<T>() - xcen;

        // Build the return triple
        T zero = T(0);
        if (return_error && return_chi2array)
        {
            auto errs = chi2map_to_errors<T>(chi2_ups, upsample_factor);
            return {{-xshift_corr, -yshift_corr}, {(errs[0] + errs[1]) / 2, (errs[2] + errs[3]) / 2}, chi2_ups};
        }
        else if (return_error)
        {
            auto errs = chi2map_to_errors<T>(chi2_ups, upsample_factor);
            return {{-xshift_corr, -yshift_corr}, {(errs[0] + errs[1]) / 2, (errs[2] + errs[3]) / 2}, torch::Tensor{}};
        }
        else if (return_chi2array)
        {
            return {{-xshift_corr, -yshift_corr}, {zero, zero}, chi2_ups};
        }
        else
        {
            return {{-xshift_corr, -yshift_corr}, {zero, zero}, torch::Tensor{}};
        }
    }

    // -----------------------------------------------------------------------
    // Sub‑pixel image shift via phase ramp in frequency space
    // -----------------------------------------------------------------------
    inline torch::Tensor shift2d(const torch::Tensor& img, double deltax, double deltay, double phase = 0.0)
    {
        auto img_fft = torch::fft::fft2(torch::nan_to_num(img).to(torch::kComplexDouble));

        int64_t H = img.size(0), W = img.size(1);
        auto freqY = torch::fft::fftfreq(H, 1.0, torch::dtype(torch::kFloat64)).view({-1, 1});
        auto freqX = torch::fft::fftfreq(W, 1.0, torch::dtype(torch::kFloat64)).view({1, -1});

        auto phase_arg = -2.0 * M_PI * (freqX * deltax + freqY * deltay) - phase;
        auto phase_shift = torch::exp(torch::complex(torch::zeros_like(phase_arg), phase_arg));

        auto shifted = torch::fft::ifft2(img_fft * phase_shift);
        return torch::real(shifted);
    }

} // namespace translation_estimation