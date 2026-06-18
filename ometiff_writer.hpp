#pragma once

#include "mmap.hpp"
#include "distance_weighted_blend.hpp"

#include <tiffio.h>

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <print>

namespace translation_estimation
{

    /// @brief Configuration for OME-TIFF output.
    struct OMETiffWriterConfig
    {
        uint32_t tile_size = 512;                // TIFF internal tile size
        uint16_t compression = COMPRESSION_JPEG; // compression for final TIFF
        uint32_t min_pyramid_dim = 256;          // stop adding levels when both dims fall below this
        uint8_t jpeg_quality = 85;               // JPEG quality (1-100), used when compression == JPEG
        bool jpeg_ycbcr = true;                  // Use YCbCr photometric for JPEG RGB (much smaller)
    };

    /// @brief Stateless utility for writing OME-TIFF files from normalized pixel data.
    ///        Supports both sparse-tile and full-res-buffer input paths.
    class OMETiffWriter
    {
    public:
        explicit OMETiffWriter(OMETiffWriterConfig const& cfg = OMETiffWriterConfig{}): cfg_(cfg) {}

        // ── Public API ─────────────────────────────────────────────────────────────

        /// Write a tiled OME-TIFF from a single dense full-resolution buffer.
        /// Handles both tiled and strip output depending on cfg_.tile_size.
        /// @tparam OutT  Output pixel type (uint8_t or uint16_t)
        /// @param full_res  Pointer to full-resolution normalized pixel data (canvas_w × canvas_h × channels).
        /// @param canvas_w, canvas_h  Full canvas dimensions in pixels.
        /// @param channels  Number of channels (1 or 3).
        /// @param output_path  Path to the .tif output file.
        template <typename OutT>
        void writeFromBuffer(OutT const* full_res, uint32_t canvas_w, uint32_t canvas_h, uint16_t channels,
                             std::string const& output_path) const;

        // ── Utility helpers (public for reuse) ────────────────────────────────────

        static std::string buildOMEXML(uint32_t w, uint32_t h, uint16_t ch, bool is_8bit);

        static uint32_t countPyramidLevels(uint32_t w, uint32_t h, uint32_t min_pyramid_dim);

        template <typename OutT>
        static void downsample2x(OutT const* src, uint32_t sw, uint32_t sh, uint16_t ch, OutT* dst, uint32_t dw,
                                 uint32_t dh);

    private:
        OMETiffWriterConfig cfg_;

        // ── TIFF tag helpers ──────────────────────────────────────────────────

        void setTIFFTags(TIFF* t, uint32_t w, uint32_t h, uint16_t ch, bool reduced, bool is_8bit) const;
        void setupSubIFDs(TIFF* tif, uint32_t w, uint32_t h) const;

        // ── Internal TIFF writing ──────────────────────────────────────────────

        template <typename OutT>
        void writeBufferAsTiledIFD(TIFF* tif, OutT const* data, uint32_t w, uint32_t h, uint16_t ch) const;

        template <typename OutT>
        void writeBufferAsStripIFD(TIFF* tif, OutT const* data, uint32_t w, uint32_t h, uint16_t ch) const;

        template <typename OutT>
        void writePyramidLevels(TIFF* tif, OutT const* full_res, uint32_t w, uint32_t h, uint16_t ch,
                                std::string const& output_path, bool is_8bit) const;
    };

    // ═════════════════════════════════════════════════════════════════════════════
    // ── Template implementations ─────────────────────────────────────────────────
    // ═════════════════════════════════════════════════════════════════════════════

    template <typename OutT>
    void OMETiffWriter::downsample2x(OutT const* src, uint32_t sw, uint32_t sh, uint16_t ch, OutT* dst, uint32_t dw,
                                     uint32_t dh)
    {
#pragma omp parallel for schedule(static)
        for (int dy = 0; dy < static_cast<int>(dh); ++dy)
        {
            uint32_t const sy0 = static_cast<uint32_t>(dy) * 2;
            uint32_t const sy1 = std::min(sy0 + 1, sh - 1);
            for (uint32_t dx = 0; dx < dw; ++dx)
            {
                uint32_t const sx0 = dx * 2;
                uint32_t const sx1 = std::min(sx0 + 1, sw - 1);
                size_t const i00 = (static_cast<size_t>(sy0) * sw + sx0) * ch;
                size_t const i01 = (static_cast<size_t>(sy0) * sw + sx1) * ch;
                size_t const i10 = (static_cast<size_t>(sy1) * sw + sx0) * ch;
                size_t const i11 = (static_cast<size_t>(sy1) * sw + sx1) * ch;
                size_t const oi = (static_cast<size_t>(dy) * dw + dx) * ch;

                for (uint16_t c = 0; c < ch; ++c)
                    dst[oi + c] = static_cast<OutT>(
                        (static_cast<uint32_t>(src[i00 + c]) + src[i01 + c] + src[i10 + c] + src[i11 + c] + 2) / 4);
            }
        }
    }

    inline std::string OMETiffWriter::buildOMEXML(uint32_t w, uint32_t h, uint16_t ch, bool is_8bit)
    {
        std::string xml;
        xml.reserve(1024);
        xml += R"(<?xml version="1.0" encoding="UTF-8"?>)";
        xml += R"(<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" )";
        xml += R"(xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" )";
        xml += R"(xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 )";
        xml += R"(http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">)";
        xml += R"(<Image ID="Image:0" Name="result">)";
        xml += R"(<Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type=")";
        xml += is_8bit ? "uint8" : "uint16";
        xml += R"(")";
        xml += R"( SizeX=")" + std::to_string(w) + R"(")";
        xml += R"( SizeY=")" + std::to_string(h) + R"(")";
        xml += R"( SizeC=")" + std::to_string(ch) + R"(")";
        xml += R"( SizeZ="1" SizeT="1")";
        xml += R"( BigEndian="false" Interleaved="true" SignificantBits=")";
        xml += std::to_string(is_8bit ? 8 : 16) + R"(">)";
        xml += R"(<Channel ID="Channel:0:0" SamplesPerPixel=")" + std::to_string(ch) + R"("/>)";
        xml += R"(<TiffData IFD="0" PlaneCount="1"/>)";
        xml += R"(</Pixels></Image></OME>)";
        return xml;
    }

    inline uint32_t OMETiffWriter::countPyramidLevels(uint32_t w, uint32_t h, uint32_t min_pyramid_dim)
    {
        uint32_t levels = 0;
        while (w > min_pyramid_dim || h > min_pyramid_dim)
        {
            w = std::max(1u, w / 2);
            h = std::max(1u, h / 2);
            ++levels;
        }
        return levels;
    }

    // ── Private TIFF helpers ─────────────────────────────────────────────────────

    inline void OMETiffWriter::setupSubIFDs(TIFF* tif, uint32_t w, uint32_t h) const
    {
        uint16_t n_levels = static_cast<uint16_t>(countPyramidLevels(w, h, cfg_.min_pyramid_dim));
        if (n_levels == 0) return;

        std::vector<toff_t> offsets(n_levels, 0);
        TIFFSetField(tif, TIFFTAG_SUBIFD, n_levels, offsets.data());

        std::println("  registered {} SubIFD pyramid levels", n_levels);
    }

    inline void OMETiffWriter::setTIFFTags(TIFF* t, uint32_t w, uint32_t h, uint16_t ch, bool reduced,
                                           bool is_8bit) const
    {
        TIFFSetField(t, TIFFTAG_IMAGEWIDTH, w);
        TIFFSetField(t, TIFFTAG_IMAGELENGTH, h);
        TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, is_8bit ? 8 : 16);
        TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, ch);
        TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
        TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

        uint16_t comp = cfg_.compression;
        if (comp == COMPRESSION_JPEG && !is_8bit) comp = COMPRESSION_LZW;

        TIFFSetField(t, TIFFTAG_COMPRESSION, comp);

        if (comp == COMPRESSION_JPEG)
        {
            TIFFSetField(t, TIFFTAG_JPEGQUALITY, static_cast<int>(cfg_.jpeg_quality));
            if (ch >= 3 && cfg_.jpeg_ycbcr)
            {
                TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_YCBCR);
                TIFFSetField(t, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB);
                TIFFSetField(t, TIFFTAG_YCBCRSUBSAMPLING, 2, 2);
            }
            else
            {
                TIFFSetField(t, TIFFTAG_PHOTOMETRIC, ch == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB);
            }
        }
        else
        {
            TIFFSetField(t, TIFFTAG_PHOTOMETRIC, ch == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB);
            if (comp == COMPRESSION_LZW || comp == COMPRESSION_ADOBE_DEFLATE || comp == COMPRESSION_DEFLATE)
                TIFFSetField(t, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
        }

        TIFFSetField(t, TIFFTAG_SUBFILETYPE,
                     reduced ? static_cast<uint32_t>(FILETYPE_REDUCEDIMAGE) : static_cast<uint32_t>(0));
    }

    // ── Buffer-to-TIFF helpers ───────────────────────────────────────────────────

    template <typename OutT>
    void OMETiffWriter::writeBufferAsTiledIFD(TIFF* tif, OutT const* data, uint32_t w, uint32_t h, uint16_t ch) const
    {
        uint32_t const tw = cfg_.tile_size;
        uint32_t const th_t = cfg_.tile_size;
        TIFFSetField(tif, TIFFTAG_TILEWIDTH, tw);
        TIFFSetField(tif, TIFFTAG_TILELENGTH, th_t);

        uint32_t const nx = (w + tw - 1) / tw;
        uint32_t const ny = (h + th_t - 1) / th_t;
        size_t const te = static_cast<size_t>(tw) * th_t * ch * sizeof(OutT);
        std::vector<OutT> buf(static_cast<size_t>(tw) * th_t * ch, 0);

        for (uint32_t ty = 0; ty < ny; ++ty)
        {
            uint32_t const y0 = ty * th_t;
            for (uint32_t tx = 0; tx < nx; ++tx)
            {
                uint32_t const x0 = tx * tw;
                std::memset(buf.data(), 0, te);

                uint32_t const copy_h = std::min(th_t, h - y0);
                uint32_t const copy_w = std::min(tw, w - x0);

                for (uint32_t ly = 0; ly < copy_h; ++ly)
                {
                    size_t const src_off = (static_cast<size_t>(y0 + ly) * w + x0) * ch;
                    size_t const dst_off = static_cast<size_t>(ly) * tw * ch;
                    std::memcpy(buf.data() + dst_off, data + src_off, static_cast<size_t>(copy_w) * ch * sizeof(OutT));
                }

                toff_t const idx = TIFFComputeTile(tif, x0, y0, 0, 0);
                if (TIFFWriteEncodedTile(tif, idx, buf.data(), static_cast<uint32_t>(te)) < 0)
                    throw std::runtime_error("TIFFWriteEncodedTile failed (pyramid)");
            }
        }
    }

    template <typename OutT>
    void OMETiffWriter::writeBufferAsStripIFD(TIFF* tif, OutT const* data, uint32_t w, uint32_t h, uint16_t ch) const
    {
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, w * ch));
        for (uint32_t y = 0; y < h; ++y)
        {
            size_t const off = static_cast<size_t>(y) * w * ch;
            if (TIFFWriteScanline(tif, const_cast<OutT*>(data + off), y, 0) < 0)
                throw std::runtime_error("TIFFWriteScanline failed (pyramid) at row " + std::to_string(y));
        }
    }

    // ── Pyramid level writer ─────────────────────────────────────────────────────

    template <typename OutT>
    void OMETiffWriter::writePyramidLevels(TIFF* tif, OutT const* full_res, uint32_t w, uint32_t h, uint16_t ch,
                                           std::string const& output_path, bool is_8bit) const
    {
        uint32_t cur_w = w, cur_h = h;
        OutT const* src = full_res;

        MMap pyr_buf[2];
        int dst_idx = 0;

        try
        {
            while (cur_w > cfg_.min_pyramid_dim || cur_h > cfg_.min_pyramid_dim)
            {
                uint32_t const new_w = std::max(1u, cur_w / 2);
                uint32_t const new_h = std::max(1u, cur_h / 2);
                size_t const level_bytes = static_cast<size_t>(new_w) * new_h * ch * sizeof(OutT);

                if (pyr_buf[dst_idx].ptr) pyr_buf[dst_idx].close();
                pyr_buf[dst_idx].create(output_path + ".pyrbuf" + std::to_string(dst_idx) + ".tmp", level_bytes,
                                        level_bytes);
                OutT* dst = reinterpret_cast<OutT*>(pyr_buf[dst_idx].ptr);

                downsample2x(src, cur_w, cur_h, ch, dst, new_w, new_h);

                TIFFWriteDirectory(tif);
                setTIFFTags(tif, new_w, new_h, ch, /*reduced=*/true, is_8bit);

                if (cfg_.tile_size > 0)
                    writeBufferAsTiledIFD(tif, dst, new_w, new_h, ch);
                else
                    writeBufferAsStripIFD(tif, dst, new_w, new_h, ch);

                std::println("  pyramid SubIFD level {}x{} written", new_w, new_h);

                src = dst;
                cur_w = new_w;
                cur_h = new_h;
                dst_idx = 1 - dst_idx;
            }
            TIFFWriteDirectory(tif);
        }
        catch (...)
        {
            if (pyr_buf[0].ptr) pyr_buf[0].close();
            if (pyr_buf[1].ptr) pyr_buf[1].close();
            throw;
        }

        if (pyr_buf[0].ptr) pyr_buf[0].close();
        if (pyr_buf[1].ptr) pyr_buf[1].close();
    }

    // ── Public API implementations ───────────────────────────────────────────────

    template <typename OutT>
    void OMETiffWriter::writeFromBuffer(OutT const* full_res, uint32_t canvas_w, uint32_t canvas_h, uint16_t channels,
                                        std::string const& output_path) const
    {
        bool const is_8bit = std::is_same_v<OutT, uint8_t>;

        TIFF* tif = TIFFOpen(output_path.c_str(), "w8");
        if (!tif) throw std::runtime_error("TIFFOpen failed");

        try
        {
            setTIFFTags(tif, canvas_w, canvas_h, channels, /*reduced=*/false, is_8bit);
            setupSubIFDs(tif, canvas_w, canvas_h);

            // Always write OME-XML metadata (required for OME-TIFF)
            {
                std::string ome_xml = buildOMEXML(canvas_w, canvas_h, channels, is_8bit);
                TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, ome_xml.c_str());
            }

            if (cfg_.tile_size > 0)
                writeBufferAsTiledIFD(tif, full_res, canvas_w, canvas_h, channels);
            else
                writeBufferAsStripIFD(tif, full_res, canvas_w, canvas_h, channels);

            writePyramidLevels(tif, full_res, canvas_w, canvas_h, channels, output_path, is_8bit);
        }
        catch (...)
        {
            TIFFClose(tif);
            throw;
        }
        TIFFClose(tif);
    }

} // namespace translation_estimation
