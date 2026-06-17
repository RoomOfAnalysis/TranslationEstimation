#pragma once

#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace translation_estimation
{

    // -----------------------------------------------------------------------
    // Seam direction: controls graph orientation for Dijkstra
    // -----------------------------------------------------------------------
    enum class SeamDirection
    {
        Horizontal, // seam runs top→bottom  (images stitched left↔right)
        Vertical,   // seam runs left→right  (images stitched top↔bottom)
        Free        // arbitrary source→sink  (caller supplies masks)
    };

    // -----------------------------------------------------------------------
    // Result bundle returned by dijkstra_seam
    // -----------------------------------------------------------------------
    template <typename T = double> struct SeamResult
    {
        torch::Tensor mask;                            // (H,W) bool − false=img1, true=img2
        std::vector<std::pair<int64_t, int64_t>> path; // seam pixel coordinates
        T total_cost;                                  // cumulative energy along the seam
    };

    namespace detail
    {

        // -------------------------------------------------------------------
        // Gradient magnitude via central differences (Sobel‑like 3×1 / 1×3)
        // Boundary pixels use forward / backward differences.
        // -------------------------------------------------------------------
        inline torch::Tensor _gradient_magnitude(const torch::Tensor& img)
        {
            auto img_d = img.to(torch::kFloat64).contiguous();
            int64_t H = img_d.size(0);
            int64_t W = img_d.size(1);

            auto gx = torch::zeros_like(img_d);
            auto gy = torch::zeros_like(img_d);

            // ----- x‑gradient (along columns) -----
            if (W >= 3)
            {
                // interior: central difference
                gx.slice(1, 1, W - 1).copy_((img_d.slice(1, 2, W) - img_d.slice(1, 0, W - 2)) * 0.5);
                // left edge:  forward difference
                gx.slice(1, 0, 1).copy_(img_d.slice(1, 1, 2) - img_d.slice(1, 0, 1));
                // right edge: backward difference
                gx.slice(1, W - 1, W).copy_(img_d.slice(1, W - 1, W) - img_d.slice(1, W - 2, W - 1));
            }
            else if (W == 2)
            {
                gx.copy_(img_d.slice(1, 1, 2) - img_d.slice(1, 0, 1));
            }
            // W == 1 → gx stays zero

            // ----- y‑gradient (along rows) -----
            if (H >= 3)
            {
                gy.slice(0, 1, H - 1).copy_((img_d.slice(0, 2, H) - img_d.slice(0, 0, H - 2)) * 0.5);
                gy.slice(0, 0, 1).copy_(img_d.slice(0, 1, 2) - img_d.slice(0, 0, 1));
                gy.slice(0, H - 1, H).copy_(img_d.slice(0, H - 1, H) - img_d.slice(0, H - 2, H - 1));
            }
            else if (H == 2)
            {
                gy.copy_(img_d.slice(0, 1, 2) - img_d.slice(0, 0, 1));
            }
            // H == 1 → gy stays zero

            return torch::sqrt(gx.square() + gy.square());
        }

        // -------------------------------------------------------------------
        // Priority‑queue node for Dijkstra (min‑heap by cost)
        // -------------------------------------------------------------------
        struct _DijkstraNode
        {
            double cost;
            int64_t r, c;
            bool operator>(const _DijkstraNode& other) const { return cost > other.cost; }
        };

        // -------------------------------------------------------------------
        // Core Dijkstra shortest‑path on the energy map.
        //
        // Horizontal: source = top row,  sink = bottom row.
        //             Edges: (r,c) → (r+1, c-1), (r+1, c), (r+1, c+1).
        //
        // Vertical:   source = left col, sink = right col.
        //             Edges: (r,c) → (r-1, c+1), (r, c+1), (r+1, c+1).
        //
        // Free:       source / sink supplied by caller as boolean masks.
        //             Edges: 4‑neighbour, bidirectional (matches flood‑fill).
        // -------------------------------------------------------------------
        template <typename T>
        SeamResult<T> _dijkstra_seam_core(const torch::Tensor& energy, SeamDirection direction,
                                          const torch::Tensor* source_mask = nullptr,
                                          const torch::Tensor* sink_mask = nullptr)
        {
            int64_t H = energy.size(0);
            int64_t W = energy.size(1);

            TORCH_CHECK(H >= 1 && W >= 1, "Energy map must have at least one pixel");

            if (direction == SeamDirection::Free)
            {
                TORCH_CHECK(source_mask != nullptr && sink_mask != nullptr,
                            "Free direction requires source_mask and sink_mask");
                TORCH_CHECK(source_mask->sizes() == energy.sizes(), "source_mask must have same shape as energy map");
                TORCH_CHECK(sink_mask->sizes() == energy.sizes(), "sink_mask must have same shape as energy map");
            }

            auto energy_acc = energy.to(torch::kFloat64).contiguous();
            auto e_ptr = energy_acc.data_ptr<double>();

            const int64_t N = H * W;
            std::vector<double> dist(N, std::numeric_limits<double>::max());
            std::vector<int64_t> prev(N, -1);

            using MinHeap = std::priority_queue<_DijkstraNode, std::vector<_DijkstraNode>, std::greater<_DijkstraNode>>;
            MinHeap pq;

            // ---------- initialise source ----------
            std::vector<int64_t> sink_indices; // for Free mode: collect all sink idxs

            if (direction == SeamDirection::Horizontal)
            {
                for (int64_t c = 0; c < W; ++c)
                {
                    int64_t idx = c; // row 0
                    dist[idx] = e_ptr[idx];
                    pq.push({dist[idx], 0, c});
                }
            }
            else if (direction == SeamDirection::Vertical)
            {
                for (int64_t r = 0; r < H; ++r)
                {
                    int64_t idx = r * W; // column 0
                    dist[idx] = e_ptr[idx];
                    pq.push({dist[idx], r, 0});
                }
            }
            else // Free
            {
                auto src_acc = source_mask->to(torch::kBool).contiguous();
                auto snk_acc = sink_mask->to(torch::kBool).contiguous();
                auto src_ptr = src_acc.data_ptr<bool>();
                auto snk_ptr = snk_acc.data_ptr<bool>();

                for (int64_t idx = 0; idx < N; ++idx)
                {
                    if (src_ptr[idx])
                    {
                        int64_t r = idx / W, c = idx % W;
                        // ensure source & sink are exclusive
                        TORCH_CHECK(!snk_ptr[idx], "source_mask and sink_mask must not overlap at pixel (", r, ",", c,
                                    ")");
                        dist[idx] = e_ptr[idx];
                        pq.push({dist[idx], r, c});
                    }
                    if (snk_ptr[idx]) sink_indices.push_back(idx);
                }
                TORCH_CHECK(!pq.empty(), "source_mask must contain at least one pixel");
                TORCH_CHECK(!sink_indices.empty(), "sink_mask must contain at least one pixel");
            }

            // ---------- Dijkstra loop ----------
            int64_t best_end_idx = -1;
            double best_end_cost = std::numeric_limits<double>::max();

            // Pre‑define static neighbour offsets
            constexpr int64_t dr4[] = {-1, 1, 0, 0};
            constexpr int64_t dc4[] = {0, 0, -1, 1};

            while (!pq.empty())
            {
                auto [cost, r, c] = pq.top();
                pq.pop();

                int64_t cur_idx = r * W + c;
                if (cost > dist[cur_idx]) continue; // stale entry

                // ---------- check destination ----------
                if (direction == SeamDirection::Horizontal)
                {
                    if (r == H - 1)
                    {
                        best_end_idx = cur_idx;
                        best_end_cost = cost;
                        break;
                    }
                }
                else if (direction == SeamDirection::Vertical)
                {
                    if (c == W - 1)
                    {
                        best_end_idx = cur_idx;
                        best_end_cost = cost;
                        break;
                    }
                }
                else // Free
                {
                    // check against pre‑collected sink indices
                    // (linear scan is fine — sink_indices is small compared to |V|)
                    auto snk_acc = sink_mask->to(torch::kBool).contiguous();
                    auto snk_ptr = snk_acc.data_ptr<bool>();
                    if (snk_ptr[cur_idx])
                    {
                        best_end_idx = cur_idx;
                        best_end_cost = cost;
                        break;
                    }
                }

                // ---------- relax neighbours ----------
                if (direction == SeamDirection::Horizontal)
                {
                    int64_t nr = r + 1;
                    for (int64_t dc = -1; dc <= 1; ++dc)
                    {
                        int64_t nc = c + dc;
                        if (nc < 0 || nc >= W) continue;
                        int64_t nidx = nr * W + nc;
                        double new_cost = cost + e_ptr[nidx];
                        if (new_cost < dist[nidx])
                        {
                            dist[nidx] = new_cost;
                            prev[nidx] = cur_idx;
                            pq.push({new_cost, nr, nc});
                        }
                    }
                }
                else if (direction == SeamDirection::Vertical)
                {
                    int64_t nc = c + 1;
                    for (int64_t dr = -1; dr <= 1; ++dr)
                    {
                        int64_t nr = r + dr;
                        if (nr < 0 || nr >= H) continue;
                        int64_t nidx = nr * W + nc;
                        double new_cost = cost + e_ptr[nidx];
                        if (new_cost < dist[nidx])
                        {
                            dist[nidx] = new_cost;
                            prev[nidx] = cur_idx;
                            pq.push({new_cost, nr, nc});
                        }
                    }
                }
                else // Free — 4‑neighbour, bidirectional
                {
                    for (int d = 0; d < 4; ++d)
                    {
                        int64_t nr = r + dr4[d];
                        int64_t nc = c + dc4[d];
                        if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
                        int64_t nidx = nr * W + nc;
                        double new_cost = cost + e_ptr[nidx];
                        if (new_cost < dist[nidx])
                        {
                            dist[nidx] = new_cost;
                            prev[nidx] = cur_idx;
                            pq.push({new_cost, nr, nc});
                        }
                    }
                }
            }

            // ---------- fallback if no path found ----------
            if (best_end_idx < 0) throw std::runtime_error("Dijkstra seam finder: no path found from source to sink");

            // ---------- backtrack ----------
            std::vector<std::pair<int64_t, int64_t>> path;
            for (int64_t idx = best_end_idx; idx != -1; idx = prev[idx])
                path.emplace_back(idx / W, idx % W);
            std::reverse(path.begin(), path.end());

            // ---------- build binary mask ----------
            torch::Tensor mask;
            if (direction == SeamDirection::Horizontal)
            {
                std::vector<int64_t> seam_cols(static_cast<size_t>(H));
                for (auto [rr, cc] : path)
                    seam_cols[static_cast<size_t>(rr)] = cc;

                auto seam_tensor = torch::tensor(seam_cols, torch::kInt64).view({H, 1});
                auto col_range = torch::arange(W, torch::kInt64).view({1, W}).expand({H, W});
                mask = col_range >= seam_tensor; // true = img2 (right side)
            }
            else if (direction == SeamDirection::Vertical)
            {
                std::vector<int64_t> seam_rows(static_cast<size_t>(W));
                for (auto [rr, cc] : path)
                    seam_rows[static_cast<size_t>(cc)] = rr;

                auto seam_tensor = torch::tensor(seam_rows, torch::kInt64).view({1, W});
                auto row_range = torch::arange(H, torch::kInt64).view({H, 1}).expand({H, W});
                mask = row_range >= seam_tensor; // true = img2 (bottom side)
            }
            else // Free — mask is caller‑supplied sink_mask; seam path refines the boundary
            {
                mask = sink_mask->to(torch::kBool).clone();

                // Force seam pixels to img1 (source side) for a clean cut line
                for (auto [rr, cc] : path)
                    mask.index_put_({rr, cc}, false);
            }

            return {mask, std::move(path), static_cast<T>(best_end_cost)};
        }

    } // namespace detail

    // -----------------------------------------------------------------------
    // Compute per‑pixel energy (cost) map from two aligned overlap images.
    //
    // E(i,j) = α · |I₁−I₂| + β · |∇I₁−∇I₂|
    //
    // α, β ∈ [0,1] control the weighting.
    // -----------------------------------------------------------------------
    template <typename T = double>
    torch::Tensor compute_energy_map(const torch::Tensor& overlap_img1, const torch::Tensor& overlap_img2,
                                     T alpha = T{0.5}, T beta = T{0.5})
    {
        TORCH_CHECK(overlap_img1.sizes() == overlap_img2.sizes(), "Overlap images must have the same shape");
        TORCH_CHECK(overlap_img1.dim() == 2, "Images must be 2D");

        auto img1 = overlap_img1.to(torch::kFloat64);
        auto img2 = overlap_img2.to(torch::kFloat64);

        // intensity difference
        auto e_diff = torch::abs(img1 - img2);

        // gradient‑magnitude difference
        auto grad1 = detail::_gradient_magnitude(img1);
        auto grad2 = detail::_gradient_magnitude(img2);
        auto e_grad = torch::abs(grad1 - grad2);

        // combined energy
        auto energy = alpha * e_diff + beta * e_grad;

        return energy;
    }

    // -----------------------------------------------------------------------
    // Dijkstra‑based optimal seam finder.
    //
    // Takes two pre‑aligned overlap regions and returns the lowest‑energy
    // seam path together with a binary mask indicating which image to use
    // on each side of the seam.
    //
    //   overlap_img1  – left/top  image crop in the overlap region
    //   overlap_img2  – right/bottom image crop in the overlap region
    //   direction     – seam orientation (Horizontal / Vertical / Free)
    //   alpha, beta   – energy‑map weights (see compute_energy_map)
    //   source_mask   – (Free only) bool tensor, true = seam start pixels
    //   sink_mask     – (Free only) bool tensor, true = seam end pixels
    //
    // For Horizontal / Vertical the source_mask and sink_mask are ignored
    // and the corresponding full edges are used automatically.
    // -----------------------------------------------------------------------
    template <typename T = double>
    SeamResult<T> dijkstra_seam(const torch::Tensor& overlap_img1, const torch::Tensor& overlap_img2,
                                SeamDirection direction = SeamDirection::Horizontal, T alpha = T{0.5}, T beta = T{0.5},
                                torch::optional<torch::Tensor> source_mask = {},
                                torch::optional<torch::Tensor> sink_mask = {})
    {
        TORCH_CHECK(overlap_img1.sizes() == overlap_img2.sizes(), "Overlap images must have the same shape");
        TORCH_CHECK(overlap_img1.dim() == 2, "Images must be 2D");

        auto energy = compute_energy_map<T>(overlap_img1, overlap_img2, alpha, beta);

        const torch::Tensor* src_ptr = source_mask.has_value() ? &source_mask.value() : nullptr;
        const torch::Tensor* snk_ptr = sink_mask.has_value() ? &sink_mask.value() : nullptr;

        return detail::_dijkstra_seam_core<T>(energy, direction, src_ptr, snk_ptr);
    }

} // namespace translation_estimation
