#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <cstdint>
#include <vector>
#include <filesystem>
#include <limits>
#include <cstring>
#include <algorithm>

#include "mmap.hpp"

namespace translation_estimation
{
    namespace detail
    {
        template <typename F> inline uint64_t blend_weight_key_(int64_t h, int64_t w, F alpha)
        {
            uint32_t alpha_bits;
            float af = static_cast<float>(alpha);
            std::memcpy(&alpha_bits, &af, sizeof(alpha_bits));
            return (static_cast<uint64_t>(static_cast<uint32_t>(h)) << 32) |
                   (static_cast<uint64_t>(static_cast<uint32_t>(w)) |
                    (static_cast<uint64_t>(alpha_bits) << 40)); // low bits of alpha_bits
        }
    } // namespace detail

    /// @brief  Compute spatially-varying blend weights (linear ramp from edges, peaked at center).
    ///         Thread-safe for identical (h,w,alpha) tuples (cached).
    ///
    /// @tparam F  Value type (float/double)
    template <typename F = float> torch::Tensor make_blend_weights(int64_t h, int64_t w, F alpha = F(1))
    {
        // Compute in float32 – float16 arange loses integer precision for dims > 2048
        auto y = torch::arange(h, torch::kFloat32);
        auto x = torch::arange(w, torch::kFloat32);
        auto wy = (F(1) - (y - F(h - 1) / F(2)).abs() * (F(2) / F(h))).clamp_min(F(1e-6));
        auto wx = (F(1) - (x - F(w - 1) / F(2)).abs() * (F(2) / F(w))).clamp_min(F(1e-6));
        auto W = wy.unsqueeze(1) * wx.unsqueeze(0);
        if (alpha != F(1)) W = W.pow(alpha);
        return W.contiguous();
    }

    /// @brief  LRU-cached blend weight lookup. Returns a const reference to a cached tensor.
    ///         When mmap_dir is set, weights exceeding mmap_threshold are stored in
    ///         mmap-backed Buffer storage so physical pages can be reclaimed.
    template <typename F = float> class BlendWeightCache
    {
    public:
        /// @param mmap_dir       Directory for mmap temp files (empty = no mmap).
        /// @param mmap_threshold Minimum tensor bytes before using mmap storage.
        explicit BlendWeightCache(std::filesystem::path mmap_dir = {},
                                  size_t mmap_threshold = std::numeric_limits<size_t>::max())
            : mmap_dir_(std::move(mmap_dir)), mmap_threshold_(mmap_threshold)
        {
        }

        const torch::Tensor& get(int64_t h, int64_t w, F alpha)
        {
            uint64_t key = detail::blend_weight_key_(h, w, alpha);
            auto it = cache_.find(key);
            if (it != cache_.end()) return it->second;

            auto weight = make_blend_weights<F>(h, w, alpha);

            // Optionally move to mmap-backed storage to reduce heap pressure
            if (!mmap_dir_.empty() && static_cast<size_t>(weight.numel()) * weight.element_size() >= mmap_threshold_)
                weight = Buffer<F>::from_tensor(weight, "blend_wgt", mmap_dir_);

            auto [new_it, _] = cache_.emplace(key, std::move(weight));
            return new_it->second;
        }

        void clear() noexcept { cache_.clear(); }

    private:
        std::filesystem::path mmap_dir_;
        size_t mmap_threshold_ = std::numeric_limits<size_t>::max();
        std::unordered_map<uint64_t, torch::Tensor> cache_;
    };

    // ── Accumulator normalization (distance-weighted blending) ──────────────

    /// @brief  Clamp a float value to the output integer type range.
    template <typename OutT> inline OutT clampOutput(float v);

    template <> inline uint8_t clampOutput<uint8_t>(float v)
    {
        return static_cast<uint8_t>(std::clamp(static_cast<int>(v + 0.5f), 0, 255));
    }

    template <> inline uint16_t clampOutput<uint16_t>(float v)
    {
        return static_cast<uint16_t>(std::clamp(static_cast<int>(v + 0.5f), 0, 65535));
    }

    /// @brief  Normalize a single pixel from accumulator values.
    template <typename OutT> inline void normalizePixel(float const* acc, float cnt_val, OutT* out, uint16_t ch)
    {
        float const inv = (cnt_val > 0.f) ? 1.f / cnt_val : 0.f;
        if (ch == 1)
            out[0] = clampOutput<OutT>(acc[0] * inv);
        else
            for (uint16_t ci = 0; ci < ch; ++ci)
                out[ci] = clampOutput<OutT>(acc[ci] * inv);
    }

    /// @brief  Normalize a dense region from acc/cnt arrays into an output buffer.
    ///         Output stride equals region width (contiguous rows).
    template <typename OutT>
    void normalizeRegion(float const* acc, float const* cnt, uint32_t w, uint32_t h, uint16_t ch, OutT* out)
    {
        for (uint32_t y = 0; y < h; ++y)
        {
            size_t const row = static_cast<size_t>(y) * w;
            for (uint32_t x = 0; x < w; ++x)
            {
                size_t const px = row + x;
                normalizePixel<OutT>(acc + px * ch, cnt[px], out + px * ch, ch);
            }
        }
    }

    /// @brief  Normalize a tile: zero-fills output, then normalizes from acc/cnt.
    template <typename OutT>
    void normalizeTile(float const* acc, float const* cnt, uint32_t tw, uint32_t th, uint16_t ch, OutT* out)
    {
        std::memset(out, 0, static_cast<size_t>(tw) * th * ch * sizeof(OutT));
        normalizeRegion<OutT>(acc, cnt, tw, th, ch, out);
    }

    // ── Sparse tile assembly ────────────────────────────────────────────────

    /// @brief  One sparse tile: grid position and raw float accumulator arrays.
    struct SparseTile
    {
        uint32_t tx, ty;  // tile grid coordinates
        float const* acc; // tile_w × tile_h × channels elements
        float const* cnt; // tile_w × tile_h elements
    };

    /// @brief  Assemble sparse tiles into a dense normalized output buffer.
    ///         Zero-fills @p out, normalizes each tile, copies to position.
    template <typename OutT>
    void assembleSparseTiles(std::vector<SparseTile> const& tiles, uint32_t tw, uint32_t th, uint32_t w, uint32_t h,
                             uint16_t ch, OutT* out)
    {
        std::memset(out, 0, static_cast<size_t>(w) * h * ch * sizeof(OutT));
        std::vector<OutT> norm(static_cast<size_t>(tw) * th * ch);
        for (auto const& st : tiles)
        {
            uint32_t const x0 = st.tx * tw;
            uint32_t const y0 = st.ty * th;
            uint32_t const cw = std::min(tw, w - x0);
            uint32_t const cph = std::min(th, h - y0);
            normalizeTile<OutT>(st.acc, st.cnt, tw, th, ch, norm.data());
            for (uint32_t ly = 0; ly < cph; ++ly)
            {
                size_t const src = static_cast<size_t>(ly) * tw * ch;
                size_t const dst = (static_cast<size_t>(y0 + ly) * w + x0) * ch;
                std::memcpy(out + dst, norm.data() + src, static_cast<size_t>(cw) * ch * sizeof(OutT));
            }
        }
    }

} // namespace translation_estimation
