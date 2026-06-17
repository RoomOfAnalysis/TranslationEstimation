#include "seam_find.hpp"
#include "phase_cross_correlation_torch.hpp"
#include "utils_torch.hpp"

#include <argparse/argparse.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <unordered_map>

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
// Test 1 — Identical images → energy map is all zeros
// -----------------------------------------------------------------------
bool test_identical_images()
{
    std::println("--- Test 1: Identical images ---");

    auto img = make_tensor<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, 3, 3);

    auto energy = translation_estimation::compute_energy_map<double>(img, img);
    auto energy_sum = torch::sum(energy).item<double>();

    bool ok = (energy_sum < 1e-9);
    std::println("  Energy sum: {:.6f}  {}", energy_sum, ok ? "PASS" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 2 — Constant-difference images → uniform non-zero energy
// -----------------------------------------------------------------------
bool test_constant_difference()
{
    std::println("--- Test 2: Constant difference ---");

    auto img1 = make_tensor<double>({1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 2, 3);
    auto img2 = make_tensor<double>({3.0, 3.0, 3.0, 3.0, 3.0, 3.0}, 2, 3);

    auto energy = translation_estimation::compute_energy_map<double>(img1, img2);
    auto vals = energy.view({-1});

    // With zero gradient, energy = α·|I1−I2| = 0.5 * 2 = 1.0 everywhere
    bool ok = true;
    for (int64_t i = 0; i < vals.size(0); ++i)
    {
        double v = vals[i].item<double>();
        if (std::abs(v - 1.0) > 1e-6)
        {
            std::println("  FAIL: energy[{}] = {:.6f}, expected 1.0", i, v);
            ok = false;
        }
    }

    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 3 — Crossing gradients → seam follows the minimum‑energy column
// -----------------------------------------------------------------------
bool test_sharp_boundary_horizontal()
{
    std::println("--- Test 3: Crossing gradients (horizontal seam) ---");

    // img1 = ramp 0→1,   img2 = ramp 1→0.
    // |I1−I2| is minimised at the centre where they cross.
    // Gradients are constant and opposite, so |∇I1−∇I2| is constant.
    // The seam should go straight down the middle column.

    constexpr int64_t H = 10;
    constexpr int64_t W = 9; // odd → unambiguous centre column

    std::vector<double> data1(static_cast<size_t>(H * W));
    std::vector<double> data2(static_cast<size_t>(H * W));
    for (int64_t r = 0; r < H; ++r)
    {
        for (int64_t c = 0; c < W; ++c)
        {
            double v1 = static_cast<double>(c) / static_cast<double>(W - 1);
            double v2 = 1.0 - v1;
            data1[static_cast<size_t>(r * W + c)] = v1;
            data2[static_cast<size_t>(r * W + c)] = v2;
        }
    }

    auto img1 = make_tensor(data1, H, W);
    auto img2 = make_tensor(data2, H, W);

    auto result =
        translation_estimation::dijkstra_seam<double>(img1, img2, translation_estimation::SeamDirection::Horizontal);

    // The minimum energy column is W/2 (integer division = 4 for W=9)
    int64_t expected_col = W / 2;
    bool ok = true;
    for (auto [r, c] : result.path)
    {
        if (c < expected_col - 1 || c > expected_col + 1)
        {
            std::println("  FAIL: path[{}] = col {}, expected near {}", r, c, expected_col);
            ok = false;
        }
    }

    std::println("  Path length: {}  {}", result.path.size(), ok ? "PASS" : "FAIL");
    std::println("  Total cost:   {:.4f}", result.total_cost);
    return ok;
}

// -----------------------------------------------------------------------
// Test 4 — Crossing gradients (vertical seam direction)
// -----------------------------------------------------------------------
bool test_sharp_boundary_vertical()
{
    std::println("--- Test 4: Crossing gradients (vertical seam) ---");

    constexpr int64_t H = 9; // odd → unambiguous centre row
    constexpr int64_t W = 10;

    std::vector<double> data1(static_cast<size_t>(H * W));
    std::vector<double> data2(static_cast<size_t>(H * W));
    for (int64_t r = 0; r < H; ++r)
    {
        for (int64_t c = 0; c < W; ++c)
        {
            double v1 = static_cast<double>(r) / static_cast<double>(H - 1);
            double v2 = 1.0 - v1;
            data1[static_cast<size_t>(r * W + c)] = v1;
            data2[static_cast<size_t>(r * W + c)] = v2;
        }
    }

    auto img1 = make_tensor(data1, H, W);
    auto img2 = make_tensor(data2, H, W);

    auto result =
        translation_estimation::dijkstra_seam<double>(img1, img2, translation_estimation::SeamDirection::Vertical);

    int64_t expected_row = H / 2;
    bool ok = true;
    for (auto [r, c] : result.path)
    {
        if (r < expected_row - 1 || r > expected_row + 1)
        {
            std::println("  FAIL: path[{}] = row {}, expected near {}", c, r, expected_row);
            ok = false;
        }
    }

    std::println("  Path length: {}  {}", result.path.size(), ok ? "PASS" : "FAIL");
    std::println("  Total cost:   {:.4f}", result.total_cost);
    return ok;
}

// -----------------------------------------------------------------------
// Test 5 — Narrow overlap (single column)
// -----------------------------------------------------------------------
bool test_narrow_overlap()
{
    std::println("--- Test 5: Narrow overlap (1 column) ---");

    constexpr int64_t H = 20;
    constexpr int64_t W = 1;

    auto img1 = torch::rand({H, W}, torch::kFloat64);
    auto img2 = torch::rand({H, W}, torch::kFloat64);

    auto result =
        translation_estimation::dijkstra_seam<double>(img1, img2, translation_estimation::SeamDirection::Horizontal);

    bool ok = (result.path.size() == static_cast<size_t>(H));

    // Every pixel should be at column 0 (the only column)
    for (auto [r, c] : result.path)
    {
        if (c != 0)
        {
            std::println("  FAIL: path[{}] = col {}, expected 0", r, c);
            ok = false;
        }
    }

    std::println("  Path length: {}  {}", result.path.size(), ok ? "PASS" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 6 — Mask correctness
// -----------------------------------------------------------------------
bool test_mask_correctness()
{
    std::println("--- Test 6: Mask correctness ---");

    constexpr int64_t H = 5;
    constexpr int64_t W = 7;

    // Create a gradient that forces the seam to go down the middle
    std::vector<double> d1(static_cast<size_t>(H * W));
    std::vector<double> d2(static_cast<size_t>(H * W));
    for (int64_t r = 0; r < H; ++r)
    {
        for (int64_t c = 0; c < W; ++c)
        {
            // img1: intensity increases with column
            d1[static_cast<size_t>(r * W + c)] = static_cast<double>(c);
            // img2: intensity decreases with column — cross at column 3
            d2[static_cast<size_t>(r * W + c)] = static_cast<double>(W - 1 - c);
        }
    }

    auto img1 = make_tensor(d1, H, W);
    auto img2 = make_tensor(d2, H, W);

    auto result =
        translation_estimation::dijkstra_seam<double>(img1, img2, translation_estimation::SeamDirection::Horizontal);

    // Check mask coverage: every pixel is either 0 (img1) or 1 (img2)
    auto mask = result.mask.to(torch::kBool);
    auto mask_sum = torch::sum(mask.to(torch::kInt64)).item<int64_t>();
    auto total = H * W;

    // The mask should have a mix of 0 and 1 (not all same)
    bool ok = (mask_sum > 0) && (mask_sum < total);

    std::println("  Mask shape:   [{}, {}]", mask.size(0), mask.size(1));
    std::println("  Mask coverage: {} / {} pixels = img2  {}", mask_sum, total, ok ? "PASS" : "FAIL");

    return ok;
}

// -----------------------------------------------------------------------
// Test 7 — Free mode: L‑shaped seam (top edge → right edge)
// -----------------------------------------------------------------------
bool test_free_l_shaped()
{
    std::println("--- Test 7: Free mode L‑shaped (top→right) ---");

    constexpr int64_t H = 15;
    constexpr int64_t W = 15;

    // Ramp images: img1 brighter toward top‑right, img2 toward bottom‑left.
    // The minimum‑energy seam should curve from top edge to right edge.
    std::vector<double> d1(static_cast<size_t>(H * W));
    std::vector<double> d2(static_cast<size_t>(H * W));
    for (int64_t r = 0; r < H; ++r)
    {
        for (int64_t c = 0; c < W; ++c)
        {
            // img1 grows with (r + c), img2 shrinks — they cross along a diagonal
            double v = static_cast<double>(r + c) / static_cast<double>(H + W - 2);
            d1[static_cast<size_t>(r * W + c)] = v;
            d2[static_cast<size_t>(r * W + c)] = 1.0 - v;
        }
    }

    auto img1 = make_tensor(d1, H, W);
    auto img2 = make_tensor(d2, H, W);

    // Source: top row; Sink: right column, bottom half (rows H/2 .. H-1)
    auto source = torch::zeros({H, W}, torch::kBool);
    auto sink = torch::zeros({H, W}, torch::kBool);
    source.index_put_({0, torch::indexing::Slice()}, true);           // top row
    sink.index_put_({torch::indexing::Slice(H / 2, H), W - 1}, true); // right col, bottom half

    auto result = translation_estimation::dijkstra_seam<double>(img1, img2, translation_estimation::SeamDirection::Free,
                                                                0.5, 0.5, source, sink);

    // The path should start on top edge and end on right edge
    bool ok = true;
    auto [r0, c0] = result.path.front();
    auto [r1, c1] = result.path.back();

    if (r0 != 0)
    {
        std::println("  FAIL: path starts at row {}, expected 0", r0);
        ok = false;
    }
    if (c1 != W - 1)
    {
        std::println("  FAIL: path ends at col {}, expected {}", c1, W - 1);
        ok = false;
    }
    if (result.path.size() < 2)
    {
        std::println("  FAIL: path too short");
        ok = false;
    }

    std::println("  Path: ({},{}) → ({},{})  length={}  {}", r0, c0, r1, c1, result.path.size(), ok ? "PASS" : "FAIL");
    std::println("  Total cost: {:.4f}", result.total_cost);
    return ok;
}

// -----------------------------------------------------------------------
// Test 8 — Free mode: overlapping source/sink → must throw
// -----------------------------------------------------------------------
bool test_free_overlapping_masks()
{
    std::println("--- Test 8: Free mode overlapping source/sink ---");

    constexpr int64_t H = 5;
    constexpr int64_t W = 5;
    auto img1 = torch::rand({H, W}, torch::kFloat64);
    auto img2 = torch::rand({H, W}, torch::kFloat64);

    auto source = torch::zeros({H, W}, torch::kBool);
    auto sink = torch::zeros({H, W}, torch::kBool);
    source.index_put_({0, 0}, true);
    sink.index_put_({0, 0}, true); // overlap!

    bool ok = false;
    try
    {
        translation_estimation::dijkstra_seam<double>(img1, img2, translation_estimation::SeamDirection::Free, 0.5, 0.5,
                                                      source, sink);
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
// Test 9 — Free mode mask correctness
// -----------------------------------------------------------------------
bool test_free_mask_correctness()
{
    std::println("--- Test 9: Free mode mask correctness ---");

    constexpr int64_t H = 10;
    constexpr int64_t W = 10;

    // Same crossing‑gradient setup: seam should split the image diagonally
    std::vector<double> d1(static_cast<size_t>(H * W));
    std::vector<double> d2(static_cast<size_t>(H * W));
    for (int64_t r = 0; r < H; ++r)
    {
        for (int64_t c = 0; c < W; ++c)
        {
            double v = static_cast<double>(r + c) / static_cast<double>(H + W - 2);
            d1[static_cast<size_t>(r * W + c)] = v;
            d2[static_cast<size_t>(r * W + c)] = 1.0 - v;
        }
    }

    auto img1 = make_tensor(d1, H, W);
    auto img2 = make_tensor(d2, H, W);

    auto source = torch::zeros({H, W}, torch::kBool);
    auto sink = torch::zeros({H, W}, torch::kBool);
    source.index_put_({0, torch::indexing::Slice()}, true);           // top row
    sink.index_put_({torch::indexing::Slice(H / 2, H), W - 1}, true); // right col, bottom half

    auto result = translation_estimation::dijkstra_seam<double>(img1, img2, translation_estimation::SeamDirection::Free,
                                                                0.5, 0.5, source, sink);

    auto mask = result.mask.to(torch::kBool);
    auto mask_sum = torch::sum(mask.to(torch::kInt64)).item<int64_t>();
    auto total = H * W;

    // In Free mode, mask == sink_mask (with seam pixels forced to img1).
    // So mask should equal sink_mask everywhere except at seam pixels.
    auto expected = sink.to(torch::kBool).clone();
    // seam pixels forced to false (img1) — account for this
    for (auto [rr, cc] : result.path)
        expected.index_put_({rr, cc}, false);

    bool ok = torch::equal(mask, expected);

    // Additionally: source pixels (except seam) should be img1 (false)
    // and sink pixels (except seam) should be img2 (true)
    if (!ok) std::println("  FAIL: mask != sink_mask (adjusted for seam)");

    std::println("  Mask: {} / {} = img2  {}", mask_sum, total, ok ? "PASS" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 10 — Benchmark
// -----------------------------------------------------------------------
bool test_benchmark()
{
    std::println("--- Test 10: Benchmark ---");

    constexpr int64_t H = 500;
    constexpr int64_t W = 500;

    auto img1 = torch::rand({H, W}, torch::kFloat64);
    auto img2 = torch::rand({H, W}, torch::kFloat64);

    auto t0 = std::chrono::high_resolution_clock::now();

    auto result =
        translation_estimation::dijkstra_seam<double>(img1, img2, translation_estimation::SeamDirection::Horizontal);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    bool ok = (ms < 5000); // generous threshold; typical should be < 500 ms
    std::println("  500×500 overlap: {} ms  Path length: {}  {}", ms, result.path.size(), ok ? "PASS" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Helper: parse ImageSet.txt → map { "rrrr_cccc" → (dx, dy) }
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
// Test 11 — Real stitching pipeline (horizontal + vertical pairs)
// -----------------------------------------------------------------------
bool test_real_pipeline(const std::string& image_dir)
{
    if (image_dir.empty())
    {
        std::println("--- Test 11: Real pipeline — SKIPPED (no --image-dir) ---");
        return true;
    }

    std::println("--- Test 11: Real stitching pipeline ---");
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
    int tested = 0;

    // ---------- horizontal pair: (0,0) ↔ (0,1) ----------
    {
        std::println("\n  -- Horizontal pair: 0000_0000 ↔ 0000_0001 --");

        auto ref_path = image_dir + "/0000_0000.jpg";
        auto mov_path = image_dir + "/0000_0001.jpg";
        auto ref = load_image_to_torch<double>(ref_path);
        auto mov = load_image_to_torch<double>(mov_path);

        std::println("  Ref size:  {}×{}", ref.size(0), ref.size(1));
        std::println("  Mov size:  {}×{}", mov.size(0), mov.size(1));

        // ground truth
        auto it = offsets.find("0000_0001.jpg");
        int64_t gt_dx = it->second.first;
        int64_t gt_dy = it->second.second;
        std::println("  GT shift:  dx={}, dy={}", gt_dx, gt_dy);

        // Use GT shift for overlap extraction (shift estimation may fail with
        // serpentine‑scan image sets due to rotation / exposure differences)
        auto [overlap1, overlap2] = extract_overlap(ref, mov, gt_dx, gt_dy);
        int64_t ov_H = overlap1.size(0), ov_W = overlap1.size(1);
        std::println("  Overlap:     {}×{}", ov_H, ov_W);

        // run seam finder
        auto t0 = std::chrono::high_resolution_clock::now();
        auto seam = dijkstra_seam<double>(overlap1, overlap2, SeamDirection::Horizontal);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        double avg_cost = (seam.path.size() > 0) ? seam.total_cost / static_cast<double>(seam.path.size()) : 0.0;
        std::println("  Seam path:   {} px  total_cost={:.2f}  avg_cost={:.4f}  time={}ms", seam.path.size(),
                     seam.total_cost, avg_cost, ms);

        bool seam_ok = (seam.path.size() == static_cast<size_t>(ov_H));
        if (!seam_ok)
        {
            std::println("  FAIL: path length {} != overlap height {}", seam.path.size(), ov_H);
            all_ok = false;
        }
        ++tested;
    }

    // ---------- vertical pair: (0,0) ↔ (1,0) ----------
    {
        std::println("\n  -- Vertical pair: 0000_0000 ↔ 0001_0000 --");

        auto ref_path = image_dir + "/0000_0000.jpg";
        auto mov_path = image_dir + "/0001_0000.jpg";
        auto ref = load_image_to_torch<double>(ref_path);
        auto mov = load_image_to_torch<double>(mov_path);

        auto it = offsets.find("0001_0000.jpg");
        int64_t gt_dx = it->second.first;
        int64_t gt_dy = it->second.second;
        std::println("  GT shift:  dx={}, dy={}", gt_dx, gt_dy);

        auto [overlap1, overlap2] = extract_overlap(ref, mov, gt_dx, gt_dy);
        int64_t ov_H = overlap1.size(0), ov_W = overlap1.size(1);
        std::println("  Overlap:     {}×{}", ov_H, ov_W);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto seam = dijkstra_seam<double>(overlap1, overlap2, SeamDirection::Vertical);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        double avg_cost = (seam.path.size() > 0) ? seam.total_cost / static_cast<double>(seam.path.size()) : 0.0;
        std::println("  Seam path:   {} px  total_cost={:.2f}  avg_cost={:.4f}  time={}ms", seam.path.size(),
                     seam.total_cost, avg_cost, ms);

        bool seam_ok = (seam.path.size() == static_cast<size_t>(ov_W));
        if (!seam_ok)
        {
            std::println("  FAIL: path length {} != overlap width {}", seam.path.size(), ov_W);
            all_ok = false;
        }
        ++tested;
    }

    bool ok = all_ok && (tested == 2);
    std::println("\n  {} pipeline tests  {}", tested, ok ? "ALL PASS" : "SOME FAILED");
    return ok;
}

// -----------------------------------------------------------------------
// Test 12 — Real images (legacy: two CLI image paths)
// -----------------------------------------------------------------------
bool test_real_images(const std::string& path1, const std::string& path2)
{
    if (path1.empty() || path2.empty())
    {
        std::println("--- Test 8: Real images — SKIPPED (no paths) ---");
        return true;
    }

    std::println("--- Test 8: Real images ---");

    using namespace translation_estimation::utils;
    auto img1 = load_image_to_torch<double>(path1);
    auto img2 = load_image_to_torch<double>(path2);

    // Use the smaller dimensions as the overlap region
    int64_t H = std::min(img1.size(0), img2.size(0));
    int64_t W = std::min(img1.size(1), img2.size(1));

    auto crop1 = img1.slice(0, 0, H).slice(1, 0, W);
    auto crop2 = img2.slice(0, 0, H).slice(1, 0, W);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result =
        translation_estimation::dijkstra_seam<double>(crop1, crop2, translation_estimation::SeamDirection::Horizontal);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::println("  Overlap size:  {} × {}", H, W);
    std::println("  Path length:   {}", result.path.size());
    std::println("  Total cost:    {:.4f}", result.total_cost);
    std::println("  Time:          {} ms", ms);

    // Verify path consistency
    bool ok = (result.path.size() == static_cast<size_t>(H));
    for (auto [r, c] : result.path)
    {
        if (r < 0 || r >= H || c < 0 || c >= W)
        {
            std::println("  FAIL: out-of-bounds path pixel ({}, {})", r, c);
            ok = false;
        }
    }

    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
int main(int argc, char* argv[])
{
    using namespace translation_estimation;

    argparse::ArgumentParser program("seam_find_test");
    program.add_argument("--image1").help("Path to first image").metavar("FILE");
    program.add_argument("--image2").help("Path to second image").metavar("FILE");
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

    std::string img1_path, img2_path, image_dir;
    if (program.present("--image1")) img1_path = program.get<std::string>("--image1");
    if (program.present("--image2")) img2_path = program.get<std::string>("--image2");
    if (program.present("--image-dir")) image_dir = program.get<std::string>("--image-dir");

    int passed = 0;
    int total = 0;

    auto run = [&](bool (*test)(), const char* name) {
        ++total;
        try
        {
            if (test()) ++passed;
        }
        catch (const std::exception& e)
        {
            std::println("  EXCEPTION: {}", e.what());
        }
    };

    run(test_identical_images, "identical");
    run(test_constant_difference, "constant diff");
    run(test_sharp_boundary_horizontal, "sharp H");
    run(test_sharp_boundary_vertical, "sharp V");
    run(test_narrow_overlap, "narrow");
    run(test_mask_correctness, "mask");
    run(test_free_l_shaped, "free L‑shape");
    run(test_free_overlapping_masks, "free overlap check");
    run(test_free_mask_correctness, "free mask");

    // benchmark — run separately to not skew timing
    ++total;
    try
    {
        if (test_benchmark()) ++passed;
    }
    catch (const std::exception& e)
    {
        std::println("  EXCEPTION: {}", e.what());
    }

    // real images — pipeline with ground truth
    ++total;
    try
    {
        if (test_real_pipeline(image_dir)) ++passed;
    }
    catch (const std::exception& e)
    {
        std::println("  EXCEPTION: {}", e.what());
    }

    // real images — legacy CLI pair
    ++total;
    try
    {
        if (test_real_images(img1_path, img2_path)) ++passed;
    }
    catch (const std::exception& e)
    {
        std::println("  EXCEPTION: {}", e.what());
    }

    std::println("\n========================================");
    bool all_passed = (passed == total);
    std::println("{} / {} tests passed  {}", passed, total, all_passed ? "ALL PASSED" : "SOME FAILED");
    std::println("========================================");

    return all_passed ? 0 : 1;
}
