#pragma once

#ifndef _WIN32
#include <sys/mman.h>
#else
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "mincore.lib")
#pragma comment(lib, "psapi.lib")
#endif

#include <string>
#include <filesystem>
#include <print>
#include <stdexcept>
#include <random>

namespace translation_estimation
{
    struct MMap
    {
        HANDLE hFile = INVALID_HANDLE_VALUE;
        HANDLE hMap = nullptr;
        void* ptr = nullptr;
        size_t size = 0;         // logical bytes (what the caller asked for)
        size_t mapped_bytes = 0; // actual mapped bytes, aligned to alloc granularity
        size_t va_reserved = 0;  // aligned placeholder VA size
        std::filesystem::path file_path;

        MMap() = default;

        MMap(const MMap&) = delete;
        MMap& operator=(const MMap&) = delete;

        MMap(MMap&& o) noexcept { take_(o); }

        MMap& operator=(MMap&& o) noexcept
        {
            if (this != &o)
            {
                close();
                take_(o);
            }
            return *this;
        }

        ~MMap() { close(); }

        // Create the file *and* reserve a giant VA placeholder. `max_bytes`
        // is the upper bound you'll ever resize to; pick generously, it only
        // costs virtual address space.
        void create(std::string const& path, size_t bytes, size_t max_bytes)
        {
            close();
            file_path = path;

            const size_t G = alloc_granularity();
            const size_t aligned_bytes = align_up(bytes, G);
            // Make sure max >= current and is aligned.
            const size_t aligned_max = align_up(std::max(max_bytes, bytes), G);

            hFile = CreateFileW(file_path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
            if (hFile == INVALID_HANDLE_VALUE)
                throw std::runtime_error("CreateFileW: " + win32_error_string(GetLastError()));

            // File length must cover the whole mapping (rounded up).
            set_file_size_(aligned_bytes);

            // Reserve the placeholder VA region.
            void* placeholder = VirtualAlloc2(
                /*Process*/ nullptr,
                /*BaseAddress*/ nullptr,
                /*Size*/ aligned_max,
                /*AllocationType*/ MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                /*PageProtection*/ PAGE_NOACCESS,
                /*ExtendedParams*/ nullptr, 0);
            if (!placeholder)
                throw std::runtime_error("VirtualAlloc2 placeholder: " + win32_error_string(GetLastError()));

            ptr = placeholder;
            va_reserved = aligned_max;

            // Split the placeholder so the leading `aligned_bytes` slot
            // can be replaced with the file mapping.
            if (aligned_max > aligned_bytes)
            {
                if (!VirtualFree(ptr, aligned_bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER))
                    throw std::runtime_error("VirtualFree split: " + win32_error_string(GetLastError()));
            }

            create_section_and_map_(aligned_bytes);
            size = bytes;
            mapped_bytes = aligned_bytes;
        }

        void create_in_dir(std::string const& base_name, size_t bytes,
                           const std::filesystem::path& directory = std::filesystem::temp_directory_path(),
                           size_t max_bytes = 64uz * 1024uz * 1024uz * 1024uz)
        {
            const std::string unique_name = base_name + "_" + random_suffix() + ".tmp";
            create((directory / unique_name).string(), bytes, max_bytes);
        }

        // Resize without ever holding two copies of the data in RAM or VA.
        // The base pointer is preserved.
        void resize_inplace(size_t new_bytes)
        {
            if (new_bytes == size) return;

            const size_t G = alloc_granularity();
            const size_t aligned_new = align_up(new_bytes, G);

            if (aligned_new > va_reserved) throw std::runtime_error("resize_inplace: exceeds reserved VA placeholder");

            // If the aligned mapping size doesn't change, only the logical
            // size + file length need to be updated. Don't touch VA at all.
            if (aligned_new == mapped_bytes)
            {
                // File length stays at mapped_bytes (already covers the view).
                // Just update the logical size.
                size = new_bytes;
                return;
            }

            // 1) Drop the current view back into a placeholder slot.
            if (!UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER))
                throw std::runtime_error("UnmapViewOfFile2: " + win32_error_string(GetLastError()));
            CloseHandle(hMap);
            hMap = nullptr;

            // 2) If there's a trailing placeholder piece, coalesce so we
            //    have one big placeholder again to re-split at the new
            //    boundary.
            if (va_reserved > mapped_bytes)
            {
                std::println("Coalescing placeholder: reserved {} bytes, mapped {} bytes", va_reserved, mapped_bytes);
                if (mapped_bytes > 0)
                {
                    // Merge [unmapped placeholder of size mapped_bytes]
                    // with [trailing placeholder of size va_reserved - mapped_bytes].
                    // Pass the SUM (== va_reserved) and base of the first half.
                    if (!VirtualFree(ptr, va_reserved, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS))
                        throw std::runtime_error("VirtualFree coalesce: " + win32_error_string(GetLastError()));
                }
            }

            // 3) Resize the underlying file to cover the new mapping.
            set_file_size_(aligned_new);

            // 4) Re-split: leading aligned_new bytes will host the mapping.
            if (va_reserved > aligned_new)
            {
                std::println("Re-splitting placeholder: reserved {} bytes, mapped {} bytes", va_reserved, mapped_bytes);
                if (!VirtualFree(ptr, aligned_new, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER))
                    throw std::runtime_error("VirtualFree re-split: " + win32_error_string(GetLastError()));
            }

            // 5) Create a new section at the new size and map it at the
            //    same base address.
            create_section_and_map_(aligned_new);
            size = new_bytes;
            mapped_bytes = aligned_new;
        }

        void close() noexcept
        {
            if (ptr)
            {
                if (hMap) UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);

                // Coalesce any split placeholders before releasing.
                if (va_reserved > mapped_bytes && mapped_bytes != 0)
                    VirtualFree(ptr, va_reserved, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
                VirtualFree(ptr, 0, MEM_RELEASE);
                ptr = nullptr;
            }
            if (hMap)
            {
                CloseHandle(hMap);
                hMap = nullptr;
            }
            if (hFile != INVALID_HANDLE_VALUE)
            {
                CloseHandle(hFile);
                hFile = INVALID_HANDLE_VALUE;
            }
            file_path.clear();
            size = 0;
            mapped_bytes = 0;
            va_reserved = 0;
        }

    private:
        static size_t alloc_granularity()
        {
            static const size_t g = [] {
                SYSTEM_INFO si;
                GetSystemInfo(&si);
                return static_cast<size_t>(si.dwAllocationGranularity);
            }();
            return g;
        }

        static size_t align_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

        void set_file_size_(size_t bytes)
        {
            LARGE_INTEGER li;
            li.QuadPart = static_cast<LONGLONG>(bytes);
            if (!SetFilePointerEx(hFile, li, nullptr, FILE_BEGIN) || !SetEndOfFile(hFile))
                throw std::runtime_error("SetEndOfFile: " + win32_error_string(GetLastError()));
        }

        void create_section_and_map_(size_t bytes)
        {
            // bytes is assumed already aligned to allocation granularity.
            LARGE_INTEGER li;
            li.QuadPart = static_cast<LONGLONG>(bytes);
            const DWORD high = static_cast<DWORD>((li.QuadPart >> 32) & 0xFFFFFFFF);
            const DWORD low = static_cast<DWORD>(li.QuadPart & 0xFFFFFFFF);

            hMap = CreateFileMappingW(hFile, nullptr, PAGE_READWRITE, high, low, nullptr);
            if (!hMap) throw std::runtime_error("CreateFileMappingW: " + win32_error_string(GetLastError()));

            // Map into the leading placeholder slot at fixed address `ptr`.
            // ViewSize must equal the size of the placeholder slot we're replacing.
            void* mapped = MapViewOfFile3(hMap, GetCurrentProcess(),
                                          /*BaseAddress*/ ptr,
                                          /*Offset*/ 0,
                                          /*ViewSize*/ bytes,
                                          /*AllocType*/ MEM_REPLACE_PLACEHOLDER,
                                          /*Protection*/ PAGE_READWRITE,
                                          /*ExtParams*/ nullptr, 0);
            if (!mapped || mapped != ptr)
                throw std::runtime_error("MapViewOfFile3: " + win32_error_string(GetLastError()));
        }

        void take_(MMap& o) noexcept
        {
            hFile = o.hFile;
            hMap = o.hMap;
            ptr = o.ptr;
            size = o.size;
            mapped_bytes = o.mapped_bytes;
            va_reserved = o.va_reserved;
            file_path = std::move(o.file_path);

            o.hFile = INVALID_HANDLE_VALUE;
            o.hMap = nullptr;
            o.ptr = nullptr;
            o.size = 0;
            o.mapped_bytes = 0;
            o.va_reserved = 0;
        }

        static std::string random_suffix()
        {
            thread_local std::mt19937 rng(std::random_device{}());
            std::uniform_int_distribution<int> dist(0, 15);
            const char hex[] = "0123456789abcdef";
            std::string s(8, '0');
            for (auto& c : s)
                c = hex[dist(rng)];
            return s;
        }

        static std::string win32_error_string(DWORD err)
        {
            LPWSTR wbuf = nullptr;
            const DWORD n = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                               FORMAT_MESSAGE_IGNORE_INSERTS,
                                           nullptr, err, 0, reinterpret_cast<LPWSTR>(&wbuf), 0, nullptr);
            std::string msg;
            if (n && wbuf)
            {
                const int sz = WideCharToMultiByte(CP_UTF8, 0, wbuf, static_cast<int>(n), nullptr, 0, nullptr, nullptr);
                msg.resize(static_cast<size_t>(sz));
                WideCharToMultiByte(CP_UTF8, 0, wbuf, static_cast<int>(n), msg.data(), sz, nullptr, nullptr);
                LocalFree(wbuf);
                while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r' || msg.back() == ' '))
                    msg.pop_back();
            }
            return msg + " (Win32 error " + std::to_string(err) + ")";
        }
    };

    /// @brief  Simple buffer abstraction over heap (calloc) or mmap-backed storage.
    ///         Used as the low-level storage layer for dense canvas accumulators.
    template <typename F = float> struct Buffer
    {
        Buffer() = default;
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&&) noexcept = default;
        Buffer& operator=(Buffer&&) noexcept = default;

        ~Buffer() { reset(); }

        void allocate_heap(std::size_t nbytes)
        {
            reset();
            bytes_ = nbytes;
            ptr_heap_ = static_cast<F*>(std::calloc(nbytes ? nbytes : 1, 1));
            if (!ptr_heap_) throw std::bad_alloc();
        }

        void allocate_mmap(std::size_t nbytes, const char* tag, const std::filesystem::path& dir,
                           std::size_t max_bytes = 64uz * 1024uz * 1024uz * 1024uz)
        {
            reset();
            bytes_ = nbytes;
            mm_.create_in_dir(tag, bytes_, dir, max_bytes);
            std::memset(mm_.ptr, 0, bytes_);
        }

        void resize_inplace(std::size_t new_bytes)
        {
            if (mm_.ptr)
            {
                mm_.resize_inplace(new_bytes);
            }
            else if (ptr_heap_)
            {
                F* p = static_cast<F*>(std::realloc(ptr_heap_, new_bytes ? new_bytes : 1));
                if (!p) throw std::bad_alloc();
                ptr_heap_ = p;
            }
            else
            {
                throw std::logic_error("Buffer::resize_inplace on empty buffer");
            }
            bytes_ = new_bytes;
        }

        void reset() noexcept
        {
            if (ptr_heap_)
            {
                std::free(ptr_heap_);
                ptr_heap_ = nullptr;
            }
            mm_.close();
            bytes_ = 0;
        }

        F* ptr() const noexcept { return mm_.ptr ? static_cast<F*>(mm_.ptr) : ptr_heap_; }

        std::size_t bytes() const noexcept { return bytes_; }

        bool empty() const noexcept { return ptr() == nullptr; }

        bool is_mmap() const noexcept { return mm_.ptr != nullptr; }

        /// @brief  Create a memory-mapped float32 tensor from a source tensor.
        ///         Data is copied to a temp-file-backed mmap region, then physical
        ///         pages are released so they are faulted back on demand.
        /// @param src  Source tensor (any dtype — converted to float32).
        /// @param tag  Short name for the mmap temp file (debugging aid).
        /// @param dir  Directory for the temp file (default: system temp).
        /// @return     A float32 tensor backed by mmap storage.
        static torch::Tensor from_tensor(const torch::Tensor& src, const char* tag,
                                         const std::filesystem::path& dir = std::filesystem::temp_directory_path())
        {
            auto f32 = src.to(torch::kFloat32).contiguous();
            const size_t nbytes = static_cast<size_t>(f32.numel()) * sizeof(F);

            Buffer<F> buf;
            buf.allocate_mmap(nbytes, tag, dir, nbytes);
            std::memcpy(buf.ptr(), f32.const_data_ptr<F>(), nbytes);

#ifndef _WIN32
            if (buf.is_mmap()) madvise(buf.ptr(), nbytes, MADV_DONTNEED);
#else
            if (buf.is_mmap())
            {
                // flush to disk
                FlushViewOfFile(buf.ptr(), nbytes);
                // free physical memory, mapping stays
                EmptyWorkingSet(GetCurrentProcess());
            }
#endif

            auto buf_sptr = std::make_shared<Buffer<F>>(std::move(buf));
            auto opts = f32.options();
            return torch::from_blob(buf_sptr->ptr(), f32.sizes().vec(), [buf_sptr](void*) {}, opts);
        }

    private:
        F* ptr_heap_ = nullptr;
        std::size_t bytes_ = 0;
        MMap mm_{};
    };

    /// @brief  Reflow a 2D buffer in-place after resizing — copies old data to
    ///         correct position in the new larger buffer, zeroes padding.
    ///
    /// @param buf           Buffer to reflow (must already be resized to new_total_elems)
    /// @param old_h         Old height (rows)
    /// @param old_row_elems Old row stride in elements
    /// @param new_h         New height
    /// @param new_row_elems New row stride in elements
    /// @param dy            Vertical offset of old data within new buffer (must be >= 0)
    /// @param dx_elems      Horizontal offset of old data within new buffer in elements
    /// @param new_total_elems Total element count in the resized buffer
    /// @param elem_size     Size of each element in bytes
    template <typename F, typename R>
    void reflow_2d_in_place(Buffer<F>& buf, R old_h, std::size_t old_row_elems, R new_h, std::size_t new_row_elems,
                            R dy, std::size_t dx_elems, std::size_t new_total_elems, std::size_t elem_size)
    {
        assert(dy >= 0);
        assert(static_cast<std::size_t>(dy) + static_cast<std::size_t>(old_h) <= static_cast<std::size_t>(new_h));
        assert(dx_elems + old_row_elems <= new_row_elems);

        buf.resize_inplace(new_total_elems * elem_size);
        auto* const p = static_cast<std::byte*>(static_cast<void*>(buf.ptr()));

        if (old_h > 0 && old_row_elems > 0)
        {
            for (R r = old_h; r-- > 0;)
            {
                auto* src = p + static_cast<std::size_t>(r) * old_row_elems * elem_size;
                auto* dst =
                    p + ((static_cast<std::size_t>(r) + static_cast<std::size_t>(dy)) * new_row_elems + dx_elems) *
                            elem_size;
                std::memmove(dst, src, old_row_elems * elem_size);
            }
        }

        if (dy > 0) std::memset(p, 0, static_cast<std::size_t>(dy) * new_row_elems * elem_size);

        const R below_start = dy + old_h;
        if (below_start < new_h)
            std::memset(p + static_cast<std::size_t>(below_start) * new_row_elems * elem_size, 0,
                        static_cast<std::size_t>(new_h - below_start) * new_row_elems * elem_size);

        if (old_h > 0)
        {
            const std::size_t left_bytes = dx_elems * elem_size;
            const std::size_t right_start = (dx_elems + old_row_elems) * elem_size;
            const std::size_t right_bytes = (new_row_elems - dx_elems - old_row_elems) * elem_size;
            if (left_bytes || right_bytes)
            {
                for (R r = dy; r < below_start; ++r)
                {
                    auto* row = p + static_cast<std::size_t>(r) * new_row_elems * elem_size;
                    if (left_bytes) std::memset(row, 0, left_bytes);
                    if (right_bytes) std::memset(row + right_start, 0, right_bytes);
                }
            }
        }
    }
} // namespace translation_estimation