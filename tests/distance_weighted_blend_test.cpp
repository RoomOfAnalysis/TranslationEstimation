#include "distance_weighted_blend.hpp"

#include <print>
#include <cstring>
#include <cmath>
#include <vector>

using namespace translation_estimation;

// -----------------------------------------------------------------------
// Test 1 — clampOutput<uint8_t>
// -----------------------------------------------------------------------
bool test_clamp_uint8()
{
    std::println("--- Test 1: clampOutput<uint8_t> ---");

    bool ok = true;
    ok &= (clampOutput<uint8_t>(0.0f) == 0);
    ok &= (clampOutput<uint8_t>(128.3f) == 128);
    ok &= (clampOutput<uint8_t>(255.0f) == 255);
    ok &= (clampOutput<uint8_t>(-5.0f) == 0);
    ok &= (clampOutput<uint8_t>(300.0f) == 255);
    ok &= (clampOutput<uint8_t>(127.6f) == 128); // round half-up

    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 2 — clampOutput<uint16_t>
// -----------------------------------------------------------------------
bool test_clamp_uint16()
{
    std::println("--- Test 2: clampOutput<uint16_t> ---");

    bool ok = true;
    ok &= (clampOutput<uint16_t>(0.0f) == 0);
    ok &= (clampOutput<uint16_t>(32768.0f) == 32768);
    ok &= (clampOutput<uint16_t>(65535.0f) == 65535);
    ok &= (clampOutput<uint16_t>(-1.0f) == 0);
    ok &= (clampOutput<uint16_t>(70000.0f) == 65535);

    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 3 — normalizePixel (1-channel and 3-channel, cnt=0)
// -----------------------------------------------------------------------
bool test_normalize_pixel()
{
    std::println("--- Test 3: normalizePixel ---");

    bool ok = true;

    // 1-channel: acc=200, cnt=2 → out=100
    {
        uint8_t out;
        float acc = 200.0f;
        normalizePixel<uint8_t>(&acc, 2.0f, &out, 1);
        ok &= (out == 100);
    }

    // 1-channel: cnt=0 → out=0
    {
        uint8_t out;
        float acc = 200.0f;
        normalizePixel<uint8_t>(&acc, 0.0f, &out, 1);
        ok &= (out == 0);
    }

    // 3-channel: acc={100,200,50}, cnt=2 → out={50,100,25}
    {
        float acc[3] = {100.0f, 200.0f, 50.0f};
        uint8_t out[3];
        normalizePixel<uint8_t>(acc, 2.0f, out, 3);
        ok &= (out[0] == 50);
        ok &= (out[1] == 100);
        ok &= (out[2] == 25);
    }

    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 4 — normalizeRegion (2×2, 1-channel)
// -----------------------------------------------------------------------
bool test_normalize_region()
{
    std::println("--- Test 4: normalizeRegion ---");

    // 2×2, 1-channel: acc={10,20,30,40}, cnt={2,4,6,8} → all out=5
    float acc[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float cnt[4] = {2.0f, 4.0f, 6.0f, 8.0f};
    uint8_t out[4];
    normalizeRegion<uint8_t>(acc, cnt, 2, 2, 1, out);

    bool ok = true;
    ok &= (out[0] == 5);
    ok &= (out[1] == 5);
    ok &= (out[2] == 5);
    ok &= (out[3] == 5);

    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 5 — normalizeRegion (2×2, 3-channel)
// -----------------------------------------------------------------------
bool test_normalize_region_multi_channel()
{
    std::println("--- Test 5: normalizeRegion (3-channel) ---");

    // 2×2, 3-channel: each pixel has acc={10,20,30}, cnt=2
    float acc[12] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    float cnt[4] = {2.0f, 4.0f, 6.0f, 8.0f};
    uint8_t out[12];
    normalizeRegion<uint8_t>(acc, cnt, 2, 2, 3, out);

    bool ok = true;
    // pixel (0,0): acc={10,20,30}, cnt=2 → {5,10,15}
    ok &= (out[0] == 5);
    ok &= (out[1] == 10);
    ok &= (out[2] == 15);
    // pixel (0,1): acc={40,50,60}, cnt=4 → {10,12,15}
    ok &= (out[3] == 10);
    ok &= (out[4] == 13); // 50/4=12.5 → 13
    ok &= (out[5] == 15);
    // pixel (1,0): acc={70,80,90}, cnt=6 → {12,13,15}
    // 70/6=11.67→12, 80/6=13.33→13, 90/6=15
    ok &= (out[6] == 12);
    ok &= (out[7] == 13);
    ok &= (out[8] == 15);
    // pixel (1,1): acc={100,110,120}, cnt=8 → {13,14,15}
    // 100/8=12.5→13, 110/8=13.75→14, 120/8=15
    ok &= (out[9] == 13);
    ok &= (out[10] == 14);
    ok &= (out[11] == 15);

    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 6 — normalizeTile (zero-fill for cnt=0)
// -----------------------------------------------------------------------
bool test_normalize_tile()
{
    std::println("--- Test 6: normalizeTile (zero-fill) ---");

    float acc[4] = {10.0f, 0.0f, 0.0f, 40.0f};
    float cnt[4] = {2.0f, 0.0f, 0.0f, 8.0f};
    uint8_t out[4];
    normalizeTile<uint8_t>(acc, cnt, 2, 2, 1, out);

    bool ok = true;
    ok &= (out[0] == 5); // 10/2
    ok &= (out[1] == 0); // cnt=0 → output stays 0 (zero-filled)
    ok &= (out[2] == 0); // cnt=0
    ok &= (out[3] == 5); // 40/8

    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 7 — assembleSparseTiles (two tiles on 4×4 canvas)
// -----------------------------------------------------------------------
bool test_assemble_sparse_tiles()
{
    std::println("--- Test 7: assembleSparseTiles ---");

    // 4×4 canvas, 2×2 tile size, 2 tiles at (0,0) and (1,1)
    float tile0_acc[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float tile0_cnt[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    float tile1_acc[4] = {50.0f, 60.0f, 70.0f, 80.0f};
    float tile1_cnt[4] = {2.0f, 2.0f, 2.0f, 2.0f};

    SparseTile t0{0, 0, tile0_acc, tile0_cnt};
    SparseTile t1{1, 1, tile1_acc, tile1_cnt};

    uint8_t out[16]; // 4×4
    assembleSparseTiles<uint8_t>({t0, t1}, 2, 2, 4, 4, 1, out);

    bool ok = true;
    // Tile 0 at grid (0,0) → canvas pixel (0,0)..(1,1)
    ok &= (out[0 * 4 + 0] == 5);  // 10/2
    ok &= (out[0 * 4 + 1] == 10); // 20/2
    ok &= (out[1 * 4 + 0] == 15); // 30/2
    ok &= (out[1 * 4 + 1] == 20); // 40/2
    // Tile 1 at grid (1,1) → canvas pixel (2,2)..(3,3)
    ok &= (out[2 * 4 + 2] == 25); // 50/2
    ok &= (out[2 * 4 + 3] == 30); // 60/2
    ok &= (out[3 * 4 + 2] == 35); // 70/2
    ok &= (out[3 * 4 + 3] == 40); // 80/2
    // Uncovered pixels stay 0 (zero-filled)
    ok &= (out[0 * 4 + 2] == 0);
    ok &= (out[0 * 4 + 3] == 0);
    ok &= (out[1 * 4 + 2] == 0);
    ok &= (out[1 * 4 + 3] == 0);
    ok &= (out[2 * 4 + 0] == 0);
    ok &= (out[2 * 4 + 1] == 0);

    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 8 — make_blend_weights (torch)
// -----------------------------------------------------------------------
bool test_make_blend_weights()
{
    std::println("--- Test 8: make_blend_weights ---");

    auto W = make_blend_weights<float>(5, 5, 1.0f);

    bool ok = true;
    // Shape
    ok &= (W.size(0) == 5 && W.size(1) == 5);
    // Center pixel (2,2) should have the highest weight
    auto center = W[2][2].item<float>();
    auto corner = W[0][0].item<float>();
    ok &= (center > corner);
    // All weights positive
    ok &= (W.min().item<float>() > 0.0f);
    // Weights ≤ 1.0
    ok &= (W.max().item<float>() <= 1.0f);

    std::println("  center={:.4f} corner={:.4f}  {} PASS", center, corner, ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 9 — make_blend_weights with alpha
// -----------------------------------------------------------------------
bool test_make_blend_weights_alpha()
{
    std::println("--- Test 9: make_blend_weights (alpha=2.0) ---");

    auto W1 = make_blend_weights<float>(7, 7, 1.0f);
    auto W2 = make_blend_weights<float>(7, 7, 2.0f);

    // alpha=2 should make the falloff steeper (corner relatively lower vs center)
    float ratio1 = W1[0][0].item<float>() / W1[3][3].item<float>();
    float ratio2 = W2[0][0].item<float>() / W2[3][3].item<float>();
    bool ok = (ratio2 < ratio1); // steeper falloff

    std::println("  alpha=1 ratio={:.4f}  alpha=2 ratio={:.4f}  {} PASS", ratio1, ratio2, ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 10 — BlendWeightCache
// -----------------------------------------------------------------------
bool test_blend_weight_cache()
{
    std::println("--- Test 10: BlendWeightCache ---");

    BlendWeightCache<float> cache;
    auto& a = cache.get(5, 5, 1.0f);
    auto& b = cache.get(5, 5, 1.0f);
    auto& c = cache.get(3, 3, 1.0f);

    bool ok = true;
    // Same key returns same tensor (by reference)
    ok &= (a.data_ptr<float>() == b.data_ptr<float>());
    // Different key returns different tensor
    ok &= (a.sizes() != c.sizes());

    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
int main()
{
    bool all_ok = true;
    all_ok &= test_clamp_uint8();
    all_ok &= test_clamp_uint16();
    all_ok &= test_normalize_pixel();
    all_ok &= test_normalize_region();
    all_ok &= test_normalize_region_multi_channel();
    all_ok &= test_normalize_tile();
    all_ok &= test_assemble_sparse_tiles();
    all_ok &= test_make_blend_weights();
    all_ok &= test_make_blend_weights_alpha();
    all_ok &= test_blend_weight_cache();

    std::println("\n{}", all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
