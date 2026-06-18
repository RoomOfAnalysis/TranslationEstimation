#include "distance_weighted_blend.hpp"
#include "ometiff_writer.hpp"

#include <print>
#include <filesystem>
#include <cstring>
#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <string>
#include <argparse/argparse.hpp>

using namespace translation_estimation;

// -----------------------------------------------------------------------
// Helper: clean up temp files created by the TIFF writer
// -----------------------------------------------------------------------
static void cleanup(std::string const& base)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::remove(base, ec);
    fs::remove(base + ".fullres.tmp", ec);
    // pyramid temp files (up to 10 levels)
    for (int i = 0; i < 10; ++i)
        fs::remove(base + ".pyrbuf" + std::to_string(i) + ".tmp", ec);
}

// -----------------------------------------------------------------------
// Test 1 — writeFromBuffer<uint8_t> grayscale strip mode
// -----------------------------------------------------------------------
bool test_write_buffer_uint8_strip()
{
    std::println("--- Test 1: writeFromBuffer<uint8_t> strip mode ---");

    uint8_t img[4] = {0, 64, 128, 255}; // 2×2
    OMETiffWriterConfig cfg{};
    cfg.tile_size = 0; // strip mode
    cfg.min_pyramid_dim = 1;
    OMETiffWriter writer(cfg);

    std::string path = "test_uint8_strip.tif";
    writer.writeFromBuffer<uint8_t>(img, 2, 2, 1, path);

    bool ok = std::filesystem::exists(path);
    auto sz = std::filesystem::file_size(path);
    std::println("  file exists={} size={}", ok, sz);

    // Basic sanity: file size should be > 0
    ok &= (sz > 0);

    cleanup(path);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 2 — writeFromBuffer<uint16_t> grayscale strip mode
// -----------------------------------------------------------------------
bool test_write_buffer_uint16_strip()
{
    std::println("--- Test 2: writeFromBuffer<uint16_t> strip mode ---");

    uint16_t img[4] = {0, 16384, 32768, 65535}; // 2×2
    OMETiffWriterConfig cfg{};
    cfg.tile_size = 0;
    cfg.min_pyramid_dim = 1;
    OMETiffWriter writer(cfg);

    std::string path = "test_uint16_strip.tif";
    writer.writeFromBuffer<uint16_t>(img, 2, 2, 1, path);

    bool ok = std::filesystem::exists(path);
    auto sz = std::filesystem::file_size(path);
    std::println("  file exists={} size={}", ok, sz);
    ok &= (sz > 0);

    cleanup(path);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 3 — writeFromBuffer<uint8_t> RGB strip mode
// -----------------------------------------------------------------------
bool test_write_buffer_rgb_strip()
{
    std::println("--- Test 3: writeFromBuffer<uint8_t> RGB strip mode ---");

    // 2×2 RGB: red, green, blue, gray
    uint8_t img[12] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128};
    OMETiffWriterConfig cfg{};
    cfg.tile_size = 0;
    cfg.min_pyramid_dim = 1;
    OMETiffWriter writer(cfg);

    std::string path = "test_rgb_strip.tif";
    writer.writeFromBuffer<uint8_t>(img, 2, 2, 3, path);

    bool ok = std::filesystem::exists(path);
    auto sz = std::filesystem::file_size(path);
    std::println("  file exists={} size={}", ok, sz);
    ok &= (sz > 0);

    cleanup(path);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 4 — writeFromBuffer<uint8_t> tiled mode with pyramid
// -----------------------------------------------------------------------
bool test_write_buffer_tiled()
{
    std::println("--- Test 4: writeFromBuffer<uint8_t> tiled mode ---");

    // 64×64 flat gray — large enough for pyramid with tile_size=16
    std::vector<uint8_t> img(64 * 64, 128);
    OMETiffWriterConfig cfg{};
    cfg.tile_size = 16;
    cfg.min_pyramid_dim = 16; // stop at 16×16, all levels ≥ tile_size
    OMETiffWriter writer(cfg);

    std::string path = "test_tiled.tif";
    writer.writeFromBuffer<uint8_t>(img.data(), 64, 64, 1, path);

    bool ok = std::filesystem::exists(path);
    auto sz = std::filesystem::file_size(path);
    std::println("  file exists={} size={}", ok, sz);
    ok &= (sz > 0);

    cleanup(path);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 5 — writeFromBuffer<uint8_t> tiled RGB with JPEG compression
// -----------------------------------------------------------------------
bool test_write_buffer_tiled_rgb_jpeg()
{
    std::println("--- Test 5: writeFromBuffer<uint8_t> tiled RGB JPEG ---");

    // 64×64 gradient-ish RGB — large enough for pyramid with tile_size=16
    std::vector<uint8_t> img(64 * 64 * 3);
    for (int y = 0; y < 64; ++y)
    {
        for (int x = 0; x < 64; ++x)
        {
            size_t i = (static_cast<size_t>(y) * 64 + x) * 3;
            img[i + 0] = static_cast<uint8_t>(x * 4);       // R ramp
            img[i + 1] = static_cast<uint8_t>(y * 4);       // G ramp
            img[i + 2] = static_cast<uint8_t>((x + y) * 2); // B
        }
    }

    OMETiffWriterConfig cfg{};
    cfg.tile_size = 16;
    cfg.compression = COMPRESSION_JPEG;
    cfg.jpeg_quality = 90;
    cfg.jpeg_ycbcr = true;
    cfg.min_pyramid_dim = 16; // stop at 16×16, all levels ≥ tile_size
    OMETiffWriter writer(cfg);

    std::string path = "test_tiled_rgb_jpeg.tif";
    writer.writeFromBuffer<uint8_t>(img.data(), 64, 64, 3, path);

    bool ok = std::filesystem::exists(path);
    auto sz = std::filesystem::file_size(path);
    std::println("  file exists={} size={}", ok, sz);
    ok &= (sz > 0);

    cleanup(path);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 6 — writeFromBuffer<uint16_t> tiled mode with LZW
// -----------------------------------------------------------------------
bool test_write_buffer_uint16_tiled()
{
    std::println("--- Test 6: writeFromBuffer<uint16_t> tiled LZW ---");

    // 64×64, uint16
    std::vector<uint16_t> img(64 * 64);
    for (int i = 0; i < 4096; ++i)
        img[i] = static_cast<uint16_t>(i * 16);

    OMETiffWriterConfig cfg{};
    cfg.tile_size = 16;
    cfg.compression = COMPRESSION_LZW;
    cfg.min_pyramid_dim = 16; // stop at 16×16
    OMETiffWriter writer(cfg);

    std::string path = "test_uint16_tiled.tif";
    writer.writeFromBuffer<uint16_t>(img.data(), 64, 64, 1, path);

    bool ok = std::filesystem::exists(path);
    auto sz = std::filesystem::file_size(path);
    std::println("  file exists={} size={}", ok, sz);
    ok &= (sz > 0);

    cleanup(path);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 7 — End-to-end: normalizeRegion + writeFromBuffer
// -----------------------------------------------------------------------
bool test_normalize_then_write()
{
    std::println("--- Test 7: normalizeRegion → writeFromBuffer ---");

    // 4×4, 1-channel
    float acc[16] = {};
    float cnt[16] = {};
    for (int i = 0; i < 16; ++i)
    {
        acc[i] = static_cast<float>(i * 20);
        cnt[i] = 2.0f;
    }

    uint8_t img[16];
    normalizeRegion<uint8_t>(acc, cnt, 4, 4, 1, img);

    OMETiffWriterConfig cfg{};
    cfg.tile_size = 0;
    cfg.compression = COMPRESSION_NONE; // lossless for exact pixel comparison
    cfg.min_pyramid_dim = 1;
    OMETiffWriter writer(cfg);

    std::string path = "test_normalize_write.tif";
    writer.writeFromBuffer<uint8_t>(img, 4, 4, 1, path);

    bool ok = std::filesystem::exists(path);
    auto sz = std::filesystem::file_size(path);
    std::println("  file exists={} size={}", ok, sz);
    ok &= (sz > 0);

    // Verify pixel values in output by reading back with libtiff
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (tif)
    {
        uint32_t w = 0, h = 0;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
        std::println("  read-back: {}×{}", w, h);
        ok &= (w == 4 && h == 4);

        uint16_t bps = 0, spp = 0;
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
        ok &= (bps == 8 && spp == 1);

        // Read pixel data
        std::vector<uint8_t> readback(static_cast<size_t>(w) * h);
        if (TIFFReadEncodedStrip(tif, 0, readback.data(), static_cast<tsize_t>(-1)) > 0)
        {
            // Verify first few pixels
            ok &= (readback[0] == 0);    // 0*20/2 = 0
            ok &= (readback[1] == 10);   // 1*20/2 = 10
            ok &= (readback[2] == 20);   // 2*20/2 = 20
            ok &= (readback[15] == 150); // 15*20/2 = 150
            std::println("  pixel[0]={} pixel[1]={} pixel[2]={} pixel[15]={}", readback[0], readback[1], readback[2],
                         readback[15]);
        }
        TIFFClose(tif);
    }

    cleanup(path);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
// Test 8 — Real image assembly: JPEG tiles → SparseTile → assemble → OME-TIFF
// -----------------------------------------------------------------------
bool test_real_image_assembly(std::string const& img_dir)
{
    std::println("--- Test 8: Real image assembly from tiles ---");

    namespace fs = std::filesystem;

    if (!fs::exists(img_dir))
    {
        std::println("  SKIP: image directory not found: {}", img_dir);
        return true; // not a failure
    }

    // ── Discover tiles ─────────────────────────────────────────────────
    struct TileInfo
    {
        uint32_t row, col;
        std::string path;
    };
    std::vector<TileInfo> tiles;

    for (auto const& entry : fs::directory_iterator(img_dir))
    {
        if (!entry.is_regular_file()) continue;
        auto const& name = entry.path().filename().string();
        if (!name.ends_with(".jpg")) continue;

        // Parse RRRR_CCCC.jpg
        auto underscore = name.find('_');
        if (underscore == std::string::npos) continue;
        auto dot = name.find('.', underscore);
        if (dot == std::string::npos) continue;

        uint32_t row = static_cast<uint32_t>(std::stoul(name.substr(0, underscore)));
        uint32_t col = static_cast<uint32_t>(std::stoul(name.substr(underscore + 1, dot - underscore - 1)));

        tiles.push_back({row, col, entry.path().string()});
    }

    if (tiles.empty())
    {
        std::println("  SKIP: no .jpg files found");
        return true;
    }

    std::println("  found {} tile images", tiles.size());

    // ── Load first image to get original tile dimensions ───────────────
    cv::Mat first_orig = cv::imread(tiles[0].path, cv::IMREAD_GRAYSCALE);
    if (first_orig.empty())
    {
        std::println("  FAIL: could not load {}", tiles[0].path);
        return false;
    }
    uint32_t const orig_w = first_orig.cols;
    uint32_t const orig_h = first_orig.rows;

    // Downsample to keep the test fast and memory-friendly
    uint32_t const max_tile_dim = 256;
    double scale = 1.0;
    uint32_t tile_w = orig_w, tile_h = orig_h;
    if (orig_w > max_tile_dim || orig_h > max_tile_dim)
    {
        scale = static_cast<double>(max_tile_dim) / std::max(orig_w, orig_h);
        tile_w = static_cast<uint32_t>(orig_w * scale);
        tile_h = static_cast<uint32_t>(orig_h * scale);
    }
    std::println("  tile size: {}×{} → {}×{} (scale={:.3f})", orig_w, orig_h, tile_w, tile_h, scale);

    // ── Determine canvas dimensions ────────────────────────────────────
    uint32_t max_row = 0, max_col = 0;
    for (auto const& t : tiles)
    {
        if (t.row > max_row) max_row = t.row;
        if (t.col > max_col) max_col = t.col;
    }
    uint32_t const canvas_w = (max_col + 1) * tile_w;
    uint32_t const canvas_h = (max_row + 1) * tile_h;
    std::println("  canvas: {}×{} ({}×{} tiles)", canvas_w, canvas_h, max_col + 1, max_row + 1);

    // ── Load all images, create SparseTile + cnt arrays ────────────────
    size_t const tile_pixels = static_cast<size_t>(tile_w) * tile_h;
    std::vector<cv::Mat> images;
    std::vector<std::vector<float>> cnt_bufs;
    std::vector<SparseTile> sparse_tiles;
    images.reserve(tiles.size());
    cnt_bufs.reserve(tiles.size());
    sparse_tiles.reserve(tiles.size());

    for (auto const& t : tiles)
    {
        cv::Mat img = cv::imread(t.path, cv::IMREAD_GRAYSCALE);
        if (img.empty() || img.cols != orig_w || img.rows != orig_h)
        {
            std::println("  FAIL: inconsistent tile {} ({}×{} vs {}×{})", t.path, img.cols, img.rows, orig_w, orig_h);
            return false;
        }
        if (scale < 1.0) cv::resize(img, img, cv::Size(tile_w, tile_h), 0, 0, cv::INTER_AREA);
        img.convertTo(img, CV_32F);

        cnt_bufs.emplace_back(tile_pixels, 1.0f); // uniform weight

        sparse_tiles.push_back({t.col, t.row, reinterpret_cast<float const*>(img.data), cnt_bufs.back().data()});
        images.push_back(std::move(img));
    }

    // ── Assemble into full-resolution buffer ───────────────────────────
    size_t const canvas_pixels = static_cast<size_t>(canvas_w) * canvas_h;
    size_t const canvas_bytes = canvas_pixels * sizeof(uint8_t);
    MMap canvas_map;
    std::string const output_path = "test_real_assembly.ome.tif";
    std::println("  allocating canvas buffer: {} bytes", canvas_bytes);
    canvas_map.create(output_path + ".fullres.tmp", canvas_bytes, canvas_bytes);
    auto* canvas = reinterpret_cast<uint8_t*>(canvas_map.ptr);
    std::println("  canvas buffer allocated, assembling...");

    try
    {
        assembleSparseTiles<uint8_t>(sparse_tiles, tile_w, tile_h, canvas_w, canvas_h, 1, canvas);
        std::println("  assembly complete, writing OME-TIFF...");
    }
    catch (std::exception const& e)
    {
        std::println("  FAIL during assembly: {}", e.what());
        canvas_map.close();
        return false;
    }

    // ── Write OME-TIFF ────────────────────────────────────────────────
    OMETiffWriterConfig cfg{};
    cfg.tile_size = 0; // strip mode for simplicity
    cfg.compression = COMPRESSION_JPEG;
    cfg.jpeg_quality = 92;
    cfg.jpeg_ycbcr = false; // grayscale
    cfg.min_pyramid_dim = 1;

    try
    {
        OMETiffWriter writer(cfg);
        writer.writeFromBuffer<uint8_t>(canvas, canvas_w, canvas_h, 1, output_path);
        std::println("  write complete");
    }
    catch (std::exception const& e)
    {
        std::println("  FAIL during write: {}", e.what());
        canvas_map.close();
        return false;
    }

    // ── Verify output ─────────────────────────────────────────────────
    bool ok = fs::exists(output_path);
    auto fsz = fs::file_size(output_path);
    std::println("  output: {} ({} bytes)", output_path, fsz);
    ok &= (fsz > 1000); // must be non-trivial

    // Quick read-back check
    TIFF* tif = TIFFOpen(output_path.c_str(), "r");
    if (tif)
    {
        uint32_t rw = 0, rh = 0;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &rw);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &rh);
        std::println("  read-back dimensions: {}×{}", rw, rh);
        ok &= (rw == canvas_w && rh == canvas_h);
        TIFFClose(tif);
    }

    canvas_map.close();
    cleanup(output_path);
    std::println("  {} PASS", ok ? "" : "NOT");
    return ok;
}

// -----------------------------------------------------------------------
int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("ometiff_writer_test");
    program.add_argument("--image-dir").help("Path to stitching tile directory").metavar("DIR");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (std::runtime_error const& err)
    {
        std::println("{}", err.what());
        return 1;
    }

    std::string image_dir;
    if (program.present("--image-dir")) image_dir = program.get<std::string>("--image-dir");

    bool all_ok = true;
    all_ok &= test_write_buffer_uint8_strip();
    all_ok &= test_write_buffer_uint16_strip();
    all_ok &= test_write_buffer_rgb_strip();
    all_ok &= test_write_buffer_tiled();
    all_ok &= test_write_buffer_tiled_rgb_jpeg();
    all_ok &= test_write_buffer_uint16_tiled();
    all_ok &= test_normalize_then_write();
    if (!image_dir.empty()) all_ok &= test_real_image_assembly(image_dir);

    std::println("\n{}", all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
