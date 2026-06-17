#include "laplacian_blend.hpp"
#include "seam_find.hpp"
#include "utils_torch.hpp"

#include <argparse/argparse.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <print>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// -----------------------------------------------------------------------
// Helper: create a 2D tensor from a flat initializer list
// -----------------------------------------------------------------------
template <typename T> torch::Tensor make_tensor(const std::vector<T>& data, int64_t H, int64_t W)
{
    auto t = torch::from_blob(const_cast<T*>(data.data()), {H, W},
                              std::is_same_v<T, double> ? torch::kFloat64 : torch::kFloat32)
                 .clone();
    return t;
}

// -----------------------------------------------------------------------
// Helper: create a 3D (H,W,C) tensor from a flat initializer list
// -----------------------------------------------------------------------
template <typename T> torch::Tensor make_tensor3(const std::vector<T>& data, int64_t H, int64_t W, int64_t C)
{
    auto t = torch::from_blob(const_cast<T*>(data.data()), {H, W, C},
                              std::is_same_v<T, double> ? torch::kFloat64 : torch::kFloat32)
                 .clone();
    return t;
}

// -----------------------------------------------------------------------
// Test 1 — Identical images + uniform 0.5 mask → output equals input
// -----------------------------------------------------------------------
bool test_identical_images()
{
    std::println("--- Test 1: Identical images, uniform 0.5 mask ---");

    auto img = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, 3, 3);
    auto mask = torch::full({3, 3}, 0.5f, torch::kFloat32);

    auto result = translation_estimation::laplacian_blend<float>(img, img, mask, 3);

    // Since img1 == img2, blending with any mask should produce the original image
    bool ok = torch::allclose(result, img, 1e-5, 1e-5);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 2 — White + black + 0.5 mask → uniform gray
// -----------------------------------------------------------------------
bool test_white_black_half_mask()
{
    std::println("--- Test 2: White + black, mask=0.5 → gray ---");

    auto white = torch::ones({5, 5}, torch::kFloat32);
    auto black = torch::zeros({5, 5}, torch::kFloat32);
    auto mask = torch::full({5, 5}, 0.5f, torch::kFloat32);

    auto result = translation_estimation::laplacian_blend<float>(white, black, mask, 3);

    // Every pixel should be 0.5
    bool ok = torch::allclose(result, torch::full({5, 5}, 0.5f, torch::kFloat32), 1e-4, 1e-4);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 3 — White + black + mask=1.0 → white (img1 dominates)
// -----------------------------------------------------------------------
bool test_mask_all_img1()
{
    std::println("--- Test 3: mask=1.0 → img1 ---");

    auto white = torch::ones({4, 4}, torch::kFloat32);
    auto black = torch::zeros({4, 4}, torch::kFloat32);
    auto mask = torch::ones({4, 4}, torch::kFloat32);

    auto result = translation_estimation::laplacian_blend<float>(white, black, mask, 3);

    bool ok = torch::allclose(result, white, 1e-5, 1e-5);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 4 — White + black + mask=0.0 → black (img2 dominates)
// -----------------------------------------------------------------------
bool test_mask_all_img2()
{
    std::println("--- Test 4: mask=0.0 → img2 ---");

    auto white = torch::ones({4, 4}, torch::kFloat32);
    auto black = torch::zeros({4, 4}, torch::kFloat32);
    auto mask = torch::zeros({4, 4}, torch::kFloat32);

    auto result = translation_estimation::laplacian_blend<float>(white, black, mask, 3);

    bool ok = torch::allclose(result, black, 1e-5, 1e-5);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 5 — Gradient mask: left half white, right half black, soft edge
// -----------------------------------------------------------------------
bool test_gradient_mask()
{
    std::println("--- Test 5: Gradient mask (left→right transition) ---");

    constexpr int64_t H = 32;
    constexpr int64_t W = 64;

    auto white = torch::ones({H, W}, torch::kFloat32);
    auto black = torch::zeros({H, W}, torch::kFloat32);

    // Mask: smooth horizontal ramp 0→1
    std::vector<float> mask_data(static_cast<size_t>(H * W));
    for (int64_t r = 0; r < H; ++r)
        for (int64_t c = 0; c < W; ++c)
            mask_data[static_cast<size_t>(r * W + c)] = static_cast<float>(c) / static_cast<float>(W - 1);
    auto mask = make_tensor(mask_data, H, W);

    auto result = translation_estimation::laplacian_blend<float>(white, black, mask, 4);

    // For constant images, the Laplacian blend smooths the mask through the pyramid.
    // Check that left side ≈ black, right side ≈ white, smooth transition in between.
    auto left_vals = result.slice(1, 0, 4).contiguous();                // first 4 columns
    auto right_vals = result.slice(1, W - 4, W).contiguous();           // last 4 columns
    auto mid_vals = result.slice(1, W / 2 - 2, W / 2 + 2).contiguous(); // middle 4 columns

    float left_mean = torch::mean(left_vals).item<float>();
    float right_mean = torch::mean(right_vals).item<float>();
    float mid_mean = torch::mean(mid_vals).item<float>();

    std::println("  left mean={:.4f}  mid mean={:.4f}  right mean={:.4f}", left_mean, mid_mean, right_mean);

    // With pyramid smoothing, extremes are compressed but left < mid < right must hold
    bool ok = (left_mean < mid_mean - 0.05f) && (right_mean > mid_mean + 0.05f);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 6 — Varying pyramid levels produce different results (multi-band effect)
// -----------------------------------------------------------------------
bool test_levels_parameter()
{
    std::println("--- Test 6: Different pyramid levels ---");

    constexpr int64_t H = 32;
    constexpr int64_t W = 64;

    // Create two images with different texture (high-frequency) content
    auto img1 = torch::rand({H, W}, torch::kFloat32);
    auto img2 = torch::rand({H, W}, torch::kFloat32);

    // Sharp mask: step function at column W/2
    auto mask = torch::zeros({H, W}, torch::kFloat32);
    mask.slice(1, W / 2, W).fill_(1.0f);

    auto result_1 = translation_estimation::laplacian_blend<float>(img1, img2, mask, 1);
    auto result_4 = translation_estimation::laplacian_blend<float>(img1, img2, mask, 4);

    // Single-level pyramid vs multi-level should produce different results
    auto diff = torch::max(torch::abs(result_1 - result_4)).item<float>();
    bool ok = (diff > 1e-5f);
    std::println("  max |level1 - level4| = {:.6f}  {}", diff, ok ? "PASS" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 7 — 3-channel color images
// -----------------------------------------------------------------------
bool test_color_images()
{
    std::println("--- Test 7: Color (3-channel) images ---");

    constexpr int64_t H = 4;
    constexpr int64_t W = 4;
    constexpr int64_t C = 3;

    // Create a "red" image (R=1, G=0, B=0) and "blue" image (R=0, G=0, B=1)
    std::vector<float> red_data(static_cast<size_t>(H * W * C), 0.0f);
    std::vector<float> blue_data(static_cast<size_t>(H * W * C), 0.0f);
    for (size_t i = 0; i < static_cast<size_t>(H * W); ++i)
    {
        red_data[i * 3 + 0] = 1.0f;  // R
        blue_data[i * 3 + 2] = 1.0f; // B
    }
    auto red = make_tensor3(red_data, H, W, C);
    auto blue = make_tensor3(blue_data, H, W, C);

    // 50% blend → each pixel should be (0.5, 0, 0.5)
    auto mask = torch::full({H, W}, 0.5f, torch::kFloat32);
    auto result = translation_estimation::laplacian_blend<float>(red, blue, mask, 3);

    // Check channel 0 (R): should be 0.5
    auto r_ch = result.slice(2, 0, 1).squeeze(2);
    bool ok_r = torch::allclose(r_ch, torch::full({H, W}, 0.5f, torch::kFloat32), 1e-4, 1e-4);

    // Check channel 1 (G): should be 0.0
    auto g_ch = result.slice(2, 1, 2).squeeze(2);
    bool ok_g = torch::allclose(g_ch, torch::zeros({H, W}, torch::kFloat32), 1e-4, 1e-4);

    // Check channel 2 (B): should be 0.5
    auto b_ch = result.slice(2, 2, 3).squeeze(2);
    bool ok_b = torch::allclose(b_ch, torch::full({H, W}, 0.5f, torch::kFloat32), 1e-4, 1e-4);

    bool ok = ok_r && ok_g && ok_b;
    std::println("  R channel: {}  G channel: {}  B channel: {}  {}", ok_r ? "OK" : "FAIL", ok_g ? "OK" : "FAIL",
                 ok_b ? "OK" : "FAIL", ok ? "PASS" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 8 — Shape mismatch → throws exception
// -----------------------------------------------------------------------
bool test_shape_mismatch()
{
    std::println("--- Test 8: Shape mismatch ---");

    auto img1 = torch::ones({3, 5}, torch::kFloat32);
    auto img2 = torch::ones({4, 5}, torch::kFloat32); // different H
    auto mask = torch::ones({3, 5}, torch::kFloat32);

    bool ok = false;
    try
    {
        translation_estimation::laplacian_blend<float>(img1, img2, mask, 3);
        std::println("  FAIL: expected exception but none was thrown");
    }
    catch (const std::exception& e)
    {
        std::println("  Correctly threw: {}", e.what());
        ok = true;
    }

    std::println("  {}", ok ? "PASS" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 9 — Mask spatial size mismatch
// -----------------------------------------------------------------------
bool test_mask_size_mismatch()
{
    std::println("--- Test 9: Mask size mismatch ---");

    auto img1 = torch::ones({3, 5}, torch::kFloat32);
    auto img2 = torch::ones({3, 5}, torch::kFloat32);
    auto mask = torch::ones({2, 5}, torch::kFloat32); // wrong H

    bool ok = false;
    try
    {
        translation_estimation::laplacian_blend<float>(img1, img2, mask, 3);
        std::println("  FAIL: expected exception but none was thrown");
    }
    catch (const std::exception& e)
    {
        std::println("  Correctly threw: {}", e.what());
        ok = true;
    }

    std::println("  {}", ok ? "PASS" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 10 — Double precision support
// -----------------------------------------------------------------------
bool test_double_precision()
{
    std::println("--- Test 10: Double precision ---");

    auto white = torch::ones({3, 3}, torch::kFloat64);
    auto black = torch::zeros({3, 3}, torch::kFloat64);
    auto mask = torch::full({3, 3}, 0.5, torch::kFloat32); // mask can be float32

    auto result = translation_estimation::laplacian_blend<double>(white, black, mask, 3);

    bool ok = (result.scalar_type() == torch::kFloat64) &&
              torch::allclose(result, torch::full({3, 3}, 0.5, torch::kFloat64), 1e-10, 1e-10);
    std::println("  dtype: {}  {}", (result.scalar_type() == torch::kFloat64 ? "float64" : "other"),
                 ok ? "PASS" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 11 — Benchmark
// -----------------------------------------------------------------------
bool test_benchmark()
{
    std::println("--- Test 11: Benchmark ---");

    constexpr int64_t H = 500;
    constexpr int64_t W = 500;

    auto img1 = torch::rand({H, W}, torch::kFloat32);
    auto img2 = torch::rand({H, W}, torch::kFloat32);
    auto mask = torch::rand({H, W}, torch::kFloat32);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = translation_estimation::laplacian_blend<float>(img1, img2, mask, 6);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    bool ok = (ms < 5000); // generous threshold; typical should be < 200 ms
    std::println("  500×500 blend (6 levels): {} ms  result shape: [{},{}]  {}", ms, result.size(0), result.size(1),
                 ok ? "PASS" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 12 — Color benchmark
// -----------------------------------------------------------------------
bool test_color_benchmark()
{
    std::println("--- Test 12: Color benchmark ---");

    constexpr int64_t H = 500;
    constexpr int64_t W = 500;
    constexpr int64_t C = 3;

    auto img1 = torch::rand({H, W, C}, torch::kFloat32);
    auto img2 = torch::rand({H, W, C}, torch::kFloat32);
    auto mask = torch::rand({H, W}, torch::kFloat32);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = translation_estimation::laplacian_blend<float>(img1, img2, mask, 6);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    bool ok = (ms < 5000);
    std::println("  500×500×3 blend (6 levels): {} ms  result shape: [{},{},{}]  {}", ms, result.size(0),
                 result.size(1), result.size(2), ok ? "PASS" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Helper: parse ImageSet.txt → map { "rrrr_cccc.jpg" → (dx, dy) }
// -----------------------------------------------------------------------
std::unordered_map<std::string, std::pair<int64_t, int64_t>> parse_image_set(const std::string& dir_path)
{
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> offsets;
    std::ifstream file(dir_path + "/ImageSet.txt");
    if (!file.is_open()) return offsets;

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '\r') continue;
        std::istringstream iss(line);
        std::string name;
        int64_t dx, dy;
        if (iss >> name >> dx >> dy) offsets[name] = {dx, dy};
    }
    return offsets;
}

// -----------------------------------------------------------------------
// Helper: extract overlap region given the shift of img2 relative to img1
// -----------------------------------------------------------------------
std::pair<torch::Tensor, torch::Tensor> extract_overlap(const torch::Tensor& img1, const torch::Tensor& img2,
                                                        int64_t dx, int64_t dy)
{
    int64_t H1 = img1.size(0), W1 = img1.size(1);
    int64_t H2 = img2.size(0), W2 = img2.size(1);

    int64_t r1_start = std::max(int64_t{0}, dy);
    int64_t r1_end = std::min(H1, H2 + dy);
    int64_t c1_start = std::max(int64_t{0}, dx);
    int64_t c1_end = std::min(W1, W2 + dx);

    int64_t r2_start = std::max(int64_t{0}, -dy);
    int64_t r2_end = std::min(H2, H1 - dy);
    int64_t c2_start = std::max(int64_t{0}, -dx);
    int64_t c2_end = std::min(W2, W1 - dx);

    auto crop1 = img1.slice(0, r1_start, r1_end).slice(1, c1_start, c1_end);
    auto crop2 = img2.slice(0, r2_start, r2_end).slice(1, c2_start, c2_end);

    return {crop1, crop2};
}

// -----------------------------------------------------------------------
// Test 13 — Real-image seam + blend pipeline (color)
// -----------------------------------------------------------------------
bool test_real_blend(const std::string& image_dir)
{
    if (image_dir.empty())
    {
        std::println("--- Test 13: Real blend pipeline — SKIPPED (no --image-dir) ---");
        return true;
    }

    std::println("--- Test 13: Real blend pipeline ---");
    std::println("  Dir: {}", image_dir);

    using namespace translation_estimation;
    using namespace translation_estimation::utils;

    auto offsets = parse_image_set(image_dir);
    if (offsets.empty())
    {
        std::println("  FAIL: could not parse ImageSet.txt");
        return false;
    }

    bool all_ok = true;

    // ---------- horizontal pair: (0,0) ↔ (0,1) ----------
    {
        std::println("\n  -- Horizontal pair: 0000_0000 ↔ 0000_0001 --");

        auto ref_path = image_dir + "/0000_0000.jpg";
        auto mov_path = image_dir + "/0000_0001.jpg";

        // Load as color (BGR, 3-channel, 8-bit → float32 [0,1])
        cv::Mat ref_bgr = cv::imread(ref_path, cv::IMREAD_COLOR);
        cv::Mat mov_bgr = cv::imread(mov_path, cv::IMREAD_COLOR);
        ref_bgr.convertTo(ref_bgr, CV_32FC3, 1.0 / 255.0);
        mov_bgr.convertTo(mov_bgr, CV_32FC3, 1.0 / 255.0);

        // Create grayscale copies for seam finding
        cv::Mat ref_gray, mov_gray;
        cv::cvtColor(ref_bgr, ref_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(mov_bgr, mov_gray, cv::COLOR_BGR2GRAY);

        // → torch tensors
        auto ref_color_t = torch::from_blob(ref_bgr.data, {ref_bgr.rows, ref_bgr.cols, 3}, torch::kFloat32).clone();
        auto mov_color_t = torch::from_blob(mov_bgr.data, {mov_bgr.rows, mov_bgr.cols, 3}, torch::kFloat32).clone();
        auto ref_gray_t = cv_image_to_torch_tensor<float>(ref_gray);
        auto mov_gray_t = cv_image_to_torch_tensor<float>(mov_gray);

        std::println("  Ref size:  {}×{}", ref_color_t.size(0), ref_color_t.size(1));
        std::println("  Mov size:  {}×{}", mov_color_t.size(0), mov_color_t.size(1));

        auto it = offsets.find("0000_0001.jpg");
        if (it == offsets.end())
        {
            std::println("  FAIL: 0000_0001.jpg not in ImageSet.txt");
            return false;
        }
        int64_t gt_dx = it->second.first;
        int64_t gt_dy = it->second.second;
        std::println("  GT shift:  dx={}, dy={}", gt_dx, gt_dy);

        // Extract overlaps (grayscale for seam, color for blend)
        auto [overlap1_gray, overlap2_gray] = extract_overlap(ref_gray_t, mov_gray_t, gt_dx, gt_dy);
        auto [overlap1_color, overlap2_color] = extract_overlap(ref_color_t, mov_color_t, gt_dx, gt_dy);
        int64_t ov_H = overlap1_color.size(0), ov_W = overlap1_color.size(1);
        std::println("  Overlap:     {}×{}", ov_H, ov_W);

        // ---- step 1: find seam (on grayscale) ----
        auto t0 = std::chrono::high_resolution_clock::now();
        auto seam = dijkstra_seam<float>(overlap1_gray, overlap2_gray, SeamDirection::Horizontal);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto seam_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        std::println("  Seam:        {} px  cost={:.2f}  time={}ms", seam.path.size(), seam.total_cost, seam_ms);

        // ---- step 2: soften the binary seam mask ----
        auto mask_t = seam.mask.to(torch::kFloat32).contiguous();
        cv::Mat mask_cv(static_cast<int>(ov_H), static_cast<int>(ov_W), CV_32F, mask_t.data_ptr());
        cv::Mat mask_soft;
        int kernel_size = std::max(3, static_cast<int>(std::min(ov_H, ov_W) / 16) * 2 + 1);
        cv::GaussianBlur(mask_cv, mask_soft, cv::Size(kernel_size, kernel_size), 0);
        auto soft_mask = cv_image_to_torch_tensor<float>(mask_soft);

        // ---- step 3: Laplacian blend (color) ----
        auto t2 = std::chrono::high_resolution_clock::now();
        auto blended = laplacian_blend<float>(overlap1_color, overlap2_color, soft_mask, 6);
        auto t3 = std::chrono::high_resolution_clock::now();
        auto blend_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

        std::println("  Blend time:  {}ms", blend_ms);

        // ---- validation ----
        bool ok = true;

        if (blended.size(0) != ov_H || blended.size(1) != ov_W || blended.size(2) != 3)
        {
            std::println("  FAIL: blended shape [{},{},{}] != expected [{},{},3]", blended.size(0), blended.size(1),
                         blended.size(2), ov_H, ov_W);
            ok = false;
        }

        if (torch::any(torch::isnan(blended)).item<bool>())
        {
            std::println("  FAIL: NaN values in blended result");
            ok = false;
        }
        if (torch::any(torch::isinf(blended)).item<bool>())
        {
            std::println("  FAIL: Inf values in blended result");
            ok = false;
        }

        auto bmin = torch::min(blended).item<float>();
        auto bmax = torch::max(blended).item<float>();
        auto bmean = torch::mean(blended).item<float>();
        auto in1_min = torch::min(overlap1_color).item<float>();
        auto in1_max = torch::max(overlap1_color).item<float>();
        auto in2_min = torch::min(overlap2_color).item<float>();
        auto in2_max = torch::max(overlap2_color).item<float>();
        auto global_min = std::min(in1_min, in2_min);
        auto global_max = std::max(in1_max, in2_max);
        std::println("  Input range: [{:.2f}, {:.2f}]  Blended: min={:.2f}  max={:.2f}  mean={:.2f}", global_min,
                     global_max, bmin, bmax, bmean);

        float tol = 0.05f * (global_max - global_min);
        if (bmin < global_min - tol || bmax > global_max + tol)
        {
            std::println("  FAIL: blended values outside input image range (tol={:.2f})", tol);
            ok = false;
        }

        if (!ok) all_ok = false;
        std::println("  {}", ok ? "PASS" : "FAIL");

        // ---- step 4: save color blended result ----
        {
            namespace fs = std::filesystem;
            auto result_dir = fs::path(image_dir) / "result";
            fs::create_directories(result_dir);

            // blended is (H,W,3) float32 [0,1] in BGR order
            cv::Mat blend_cv(static_cast<int>(ov_H), static_cast<int>(ov_W), CV_32FC3, blended.data_ptr<float>());
            cv::Mat blend_8u;
            blend_cv.convertTo(blend_8u, CV_8UC3, 255.0);
            std::string out_path = (result_dir / "horizontal.jpg").string();
            cv::imwrite(out_path, blend_8u);
            std::println("  Saved: {}", out_path);
        }
    }

    // ---------- vertical pair: (0,0) ↔ (1,0) ----------
    {
        std::println("\n  -- Vertical pair: 0000_0000 ↔ 0001_0000 --");

        auto ref_path = image_dir + "/0000_0000.jpg";
        auto mov_path = image_dir + "/0001_0000.jpg";

        cv::Mat ref_bgr = cv::imread(ref_path, cv::IMREAD_COLOR);
        cv::Mat mov_bgr = cv::imread(mov_path, cv::IMREAD_COLOR);
        ref_bgr.convertTo(ref_bgr, CV_32FC3, 1.0 / 255.0);
        mov_bgr.convertTo(mov_bgr, CV_32FC3, 1.0 / 255.0);

        cv::Mat ref_gray, mov_gray;
        cv::cvtColor(ref_bgr, ref_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(mov_bgr, mov_gray, cv::COLOR_BGR2GRAY);

        auto ref_color_t = torch::from_blob(ref_bgr.data, {ref_bgr.rows, ref_bgr.cols, 3}, torch::kFloat32).clone();
        auto mov_color_t = torch::from_blob(mov_bgr.data, {mov_bgr.rows, mov_bgr.cols, 3}, torch::kFloat32).clone();
        auto ref_gray_t = cv_image_to_torch_tensor<float>(ref_gray);
        auto mov_gray_t = cv_image_to_torch_tensor<float>(mov_gray);

        auto it = offsets.find("0001_0000.jpg");
        if (it == offsets.end())
        {
            std::println("  FAIL: 0001_0000.jpg not in ImageSet.txt");
            return false;
        }
        int64_t gt_dx = it->second.first;
        int64_t gt_dy = it->second.second;
        std::println("  GT shift:  dx={}, dy={}", gt_dx, gt_dy);

        auto [overlap1_gray, overlap2_gray] = extract_overlap(ref_gray_t, mov_gray_t, gt_dx, gt_dy);
        auto [overlap1_color, overlap2_color] = extract_overlap(ref_color_t, mov_color_t, gt_dx, gt_dy);
        int64_t ov_H = overlap1_color.size(0), ov_W = overlap1_color.size(1);
        std::println("  Overlap:     {}×{}", ov_H, ov_W);

        // ---- step 1: find seam ----
        auto t0 = std::chrono::high_resolution_clock::now();
        auto seam = dijkstra_seam<float>(overlap1_gray, overlap2_gray, SeamDirection::Vertical);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto seam_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        std::println("  Seam:        {} px  cost={:.2f}  time={}ms", seam.path.size(), seam.total_cost, seam_ms);

        // ---- step 2: soften mask ----
        auto mask_t = seam.mask.to(torch::kFloat32).contiguous();
        cv::Mat mask_cv(static_cast<int>(ov_H), static_cast<int>(ov_W), CV_32F, mask_t.data_ptr());
        cv::Mat mask_soft;
        int kernel_size = std::max(3, static_cast<int>(std::min(ov_H, ov_W) / 16) * 2 + 1);
        cv::GaussianBlur(mask_cv, mask_soft, cv::Size(kernel_size, kernel_size), 0);
        auto soft_mask = cv_image_to_torch_tensor<float>(mask_soft);

        // ---- step 3: Laplacian blend ----
        auto t2 = std::chrono::high_resolution_clock::now();
        auto blended = laplacian_blend<float>(overlap1_color, overlap2_color, soft_mask, 6);
        auto t3 = std::chrono::high_resolution_clock::now();
        auto blend_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

        std::println("  Blend time:  {}ms", blend_ms);

        // ---- validation ----
        bool ok = true;

        if (blended.size(0) != ov_H || blended.size(1) != ov_W || blended.size(2) != 3)
        {
            std::println("  FAIL: blended shape [{},{},{}] != expected [{},{},3]", blended.size(0), blended.size(1),
                         blended.size(2), ov_H, ov_W);
            ok = false;
        }
        if (torch::any(torch::isnan(blended)).item<bool>())
        {
            std::println("  FAIL: NaN values in blended result");
            ok = false;
        }
        if (torch::any(torch::isinf(blended)).item<bool>())
        {
            std::println("  FAIL: Inf values in blended result");
            ok = false;
        }

        auto bmin = torch::min(blended).item<float>();
        auto bmax = torch::max(blended).item<float>();
        auto bmean = torch::mean(blended).item<float>();
        auto in1_min = torch::min(overlap1_color).item<float>();
        auto in1_max = torch::max(overlap1_color).item<float>();
        auto in2_min = torch::min(overlap2_color).item<float>();
        auto in2_max = torch::max(overlap2_color).item<float>();
        auto global_min = std::min(in1_min, in2_min);
        auto global_max = std::max(in1_max, in2_max);
        float tol = 0.05f * (global_max - global_min);
        std::println("  Input range: [{:.2f}, {:.2f}]  Blended: min={:.2f}  max={:.2f}  mean={:.2f}", global_min,
                     global_max, bmin, bmax, bmean);

        if (bmin < global_min - tol || bmax > global_max + tol)
        {
            std::println("  FAIL: blended values outside input image range (tol={:.2f})", tol);
            ok = false;
        }

        if (!ok) all_ok = false;
        std::println("  {}", ok ? "PASS" : "FAIL");

        // ---- step 4: save color blended result ----
        {
            namespace fs = std::filesystem;
            auto result_dir = fs::path(image_dir) / "result";
            fs::create_directories(result_dir);

            cv::Mat blend_cv(static_cast<int>(ov_H), static_cast<int>(ov_W), CV_32FC3, blended.data_ptr<float>());
            cv::Mat blend_8u;
            blend_cv.convertTo(blend_8u, CV_8UC3, 255.0);
            std::string out_path = (result_dir / "vertical.jpg").string();
            cv::imwrite(out_path, blend_8u);
            std::println("  Saved: {}", out_path);
        }
    }

    std::println("\n  Real blend pipeline {}", all_ok ? "ALL PASS" : "SOME FAILED");
    return all_ok;
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("laplacian_blend_test");
    program.add_argument("--image-dir").help("Path to stitching tile directory with ImageSet.txt").metavar("DIR");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err)
    {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    std::string image_dir;
    if (program.present("--image-dir")) image_dir = program.get<std::string>("--image-dir");

    std::println("=== Laplacian Blend Tests ===\n");

    int passed = 0;
    int total = 0;

    auto run = [&](bool (*fn)(), const char* /*name*/) {
        ++total;
        try
        {
            if (fn()) ++passed;
        }
        catch (const std::exception& e)
        {
            std::println("  EXCEPTION: {}", e.what());
        }
        std::println("");
    };

    run(test_identical_images, "identical images");
    run(test_white_black_half_mask, "white+black half mask");
    run(test_mask_all_img1, "mask=1.0");
    run(test_mask_all_img2, "mask=0.0");
    run(test_gradient_mask, "gradient mask");
    run(test_levels_parameter, "levels parameter");
    run(test_color_images, "color images");
    run(test_shape_mismatch, "shape mismatch");
    run(test_mask_size_mismatch, "mask size mismatch");
    run(test_double_precision, "double precision");
    run(test_benchmark, "benchmark");
    run(test_color_benchmark, "color benchmark");

    // Real-image pipeline (requires --image-dir)
    {
        ++total;
        try
        {
            if (test_real_blend(image_dir)) ++passed;
        }
        catch (const std::exception& e)
        {
            std::println("  EXCEPTION: {}", e.what());
        }
        std::println("");
    }

    std::println("=== {}/{} tests passed ===", passed, total);
    return (passed == total) ? 0 : 1;
}
