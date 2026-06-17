#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <stdexcept>
#include <vector>

#include "utils_torch.hpp"

namespace translation_estimation
{

    namespace detail
    {

        // -------------------------------------------------------------------
        // Laplacian pyramid blend — works with any number of channels.
        //
        // img1, img2  – float images (1- or 3-channel), same size & type
        // mask         – single-channel float mask, same spatial size, 0–1
        // levels       – number of pyramid levels (≥ 1)
        //
        // cv::pyrDown / pyrUp / subtract / add / multiply all operate
        // natively on multi-channel Mats, so we build a single pyramid
        // per image.  The mask is broadcast to match the channel count
        // via cv::merge at the blend step.
        // -------------------------------------------------------------------
        inline cv::Mat _laplacian_blend_impl(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask, int levels)
        {
            CV_Assert(img1.size() == img2.size());
            CV_Assert(img1.size() == mask.size());
            CV_Assert(img1.type() == img2.type());
            CV_Assert(mask.channels() == 1);
            CV_Assert(mask.type() == CV_32F || mask.type() == CV_64F);
            CV_Assert(levels >= 1);

            const int dtype = img1.type();
            const int nchan = img1.channels();

            // Ensure mask has the same depth as the images
            cv::Mat mask_typed;
            if (mask.depth() != img1.depth())
                mask.convertTo(mask_typed, dtype);
            else
                mask_typed = mask;

            // ---------- Gaussian pyramid for the mask (always single-channel) ----------
            std::vector<cv::Mat> g_mask;
            {
                cv::Mat cur = mask_typed.clone();
                g_mask.push_back(cur);
                for (int i = 0; i < levels; ++i)
                {
                    cv::Mat down;
                    cv::pyrDown(cur, down);
                    g_mask.push_back(down);
                    cur = down;
                }
            }

            // ---------- Laplacian pyramid for img1 (multi-channel natively) ----------
            std::vector<cv::Mat> lp1;
            {
                cv::Mat g = img1.clone();
                for (int i = 0; i < levels; ++i)
                {
                    cv::Mat g_down;
                    cv::pyrDown(g, g_down);
                    cv::Mat g_up;
                    cv::pyrUp(g_down, g_up, g.size());
                    cv::Mat laplace;
                    cv::subtract(g, g_up, laplace);
                    lp1.push_back(laplace);
                    g = g_down;
                }
                lp1.push_back(g); // top-level residual Gaussian
            }

            // ---------- Laplacian pyramid for img2 ----------
            std::vector<cv::Mat> lp2;
            {
                cv::Mat g = img2.clone();
                for (int i = 0; i < levels; ++i)
                {
                    cv::Mat g_down;
                    cv::pyrDown(g, g_down);
                    cv::Mat g_up;
                    cv::pyrUp(g_down, g_up, g.size());
                    cv::Mat laplace;
                    cv::subtract(g, g_up, laplace);
                    lp2.push_back(laplace);
                    g = g_down;
                }
                lp2.push_back(g);
            }

            // ---------- Blend each pyramid level independently ----------
            std::vector<cv::Mat> blended_pyr;
            blended_pyr.reserve(levels + 1);
            for (int i = 0; i <= levels; ++i)
            {
                // Match mask spatial size to this level
                cv::Mat m = g_mask[i];
                if (m.size() != lp1[i].size()) cv::resize(m, m, lp1[i].size());

                // Broadcast single-channel mask → n-channel for element-wise multiply
                cv::Mat m_bc;
                if (nchan > 1)
                {
                    std::vector<cv::Mat> chs(static_cast<size_t>(nchan), m);
                    cv::merge(chs, m_bc);
                }
                else
                {
                    m_bc = m;
                }

                // blended = lp1 * mask + lp2 * (1 - mask)
                cv::Mat inv_m = cv::Scalar::all(1.0) - m_bc;
                cv::Mat term1 = lp1[i].mul(m_bc);
                cv::Mat term2 = lp2[i].mul(inv_m);
                blended_pyr.push_back(term1 + term2);
            }

            // ---------- Collapse the pyramid ----------
            cv::Mat result = blended_pyr[levels].clone(); // top level
            for (int i = levels - 1; i >= 0; --i)
            {
                cv::pyrUp(result, result, blended_pyr[i].size());
                cv::add(result, blended_pyr[i], result);
            }

            return result;
        }

    } // namespace detail

    // -----------------------------------------------------------------------
    // Multi-band Laplacian pyramid blending.
    //
    // Blends two images using a soft mask via multi-resolution pyramid
    // decomposition.  Low frequencies are blended over a wide transition zone
    // while high frequencies (edges, texture) are blended over a narrow zone,
    // avoiding visible seam artefacts.
    //
    //   img1, img2  – input images, both (H,W) or (H,W,C), float32/float64
    //   mask         – blending weight (H,W), float,  0 = img2,  1 = img1
    //   levels       – number of pyramid levels (default 6);  clamped
    //                  automatically so the coarsest level is at least 2×2.
    //
    // Returns a torch::Tensor of the same shape and dtype as img1.
    //
    // Example use with seam_find:
    //   auto seam = dijkstra_seam(overlap1, overlap2, SeamDirection::Horizontal);
    //   auto soft_mask = gaussian_blur(seam.mask.to(kFloat32), 15, 5.0);
    //   auto blended = laplacian_blend(overlap1, overlap2, soft_mask, 6);
    // -----------------------------------------------------------------------
    template <typename T = float>
    torch::Tensor laplacian_blend(const torch::Tensor& img1, const torch::Tensor& img2, const torch::Tensor& mask,
                                  int levels = 6)
    {
        // ---------- validation ----------
        TORCH_CHECK(img1.sizes() == img2.sizes(), "img1 and img2 must have the same shape, got ", img1.sizes(), " vs ",
                    img2.sizes());
        TORCH_CHECK(img1.dim() == 2 || img1.dim() == 3, "Images must be 2D (H,W) or 3D (H,W,C), got ", img1.dim(), "D");
        TORCH_CHECK(mask.dim() == 2, "Mask must be 2D (H,W), got ", mask.dim(), "D");
        TORCH_CHECK(mask.size(0) == img1.size(0) && mask.size(1) == img1.size(1), "Mask spatial size ", mask.sizes(),
                    " must match image spatial size ", std::vector<int64_t>{img1.size(0), img1.size(1)});
        TORCH_CHECK(levels >= 1, "levels must be ≥ 1, got ", levels);

        // Clamp levels so that the coarsest image isn't too small for pyrDown
        {
            int64_t min_dim = std::min(img1.size(0), img1.size(1));
            int max_levels = 0;
            int64_t d = min_dim;
            while (d >= 2)
            {
                d = (d + 1) / 2;
                ++max_levels;
            }
            if (levels > max_levels)
            {
                levels = max_levels;
                TORCH_WARN("levels clamped to ", levels, " (image too small for requested pyramid depth)");
            }
            if (levels < 1) levels = 1;
        }

        // ---------- move to CPU, ensure contiguous ----------
        auto img1_cpu = img1.detach().to(torch::kCPU).contiguous();
        auto img2_cpu = img2.detach().to(torch::kCPU).contiguous();
        auto mask_cpu = mask.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();

        const bool is_double = (img1_cpu.scalar_type() == torch::kFloat64);
        const int64_t H = img1_cpu.size(0);
        const int64_t W = img1_cpu.size(1);

        // Build cv::Mat wrappers — cv::pyrDown/pyrUp handle multi-channel natively
        cv::Mat m(static_cast<int>(H), static_cast<int>(W), CV_32F, mask_cpu.data_ptr());

        cv::Mat mat1, mat2;
        if (img1_cpu.dim() == 2)
        {
            const int cv_type = is_double ? CV_64F : CV_32F;
            mat1 = cv::Mat(static_cast<int>(H), static_cast<int>(W), cv_type, img1_cpu.data_ptr());
            mat2 = cv::Mat(static_cast<int>(H), static_cast<int>(W), cv_type, img2_cpu.data_ptr());
        }
        else
        {
            const int64_t C = img1_cpu.size(2);
            // PyTorch (H,W,C) contiguous ≡ OpenCV H×W×C interleaved
            const int cv_type = is_double ? CV_64FC(static_cast<int>(C)) : CV_32FC(static_cast<int>(C));
            mat1 = cv::Mat(static_cast<int>(H), static_cast<int>(W), cv_type, img1_cpu.data_ptr());
            mat2 = cv::Mat(static_cast<int>(H), static_cast<int>(W), cv_type, img2_cpu.data_ptr());
        }

        cv::Mat blended = detail::_laplacian_blend_impl(mat1, mat2, m, levels);

        // Convert back to torch::Tensor
        if (blended.channels() == 1)
        {
            if (is_double)
                return utils::cv_image_to_torch_tensor<double>(blended);
            else
                return utils::cv_image_to_torch_tensor<float>(blended);
        }
        else
        {
            // Multi-channel: cv::Mat (H,W,CV_32FC3) → torch::Tensor (H,W,C)
            const int C = blended.channels();
            auto result = torch::empty({H, W, C}, torch::dtype(is_double ? torch::kFloat64 : torch::kFloat32));
            std::vector<cv::Mat> chs;
            cv::split(blended, chs);
            for (int c = 0; c < C; ++c)
            {
                torch::Tensor ch_tensor;
                if (is_double)
                    ch_tensor = utils::cv_image_to_torch_tensor<double>(chs[static_cast<size_t>(c)]);
                else
                    ch_tensor = utils::cv_image_to_torch_tensor<float>(chs[static_cast<size_t>(c)]);
                result.slice(2, c, c + 1).copy_(ch_tensor.unsqueeze(2));
            }
            return result;
        }
    }

} // namespace translation_estimation
