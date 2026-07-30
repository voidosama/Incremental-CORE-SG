#pragma once
// Row-major float storage that can grow (that is the whole point: the CORE-SG
// is incremental, so points keep arriving).
//
// Rows are padded with zeros to a multiple of 8 floats. Because the padding is
// zero in *both* operands, every metric in distance.hpp can run over the full
// padded width and still return the exact value -- which removes the scalar
// remainder loop from the hot path and gives the CUDA kernels a naturally
// coalesced stride.

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "coresg/distance.hpp"
#include "coresg/types.hpp"

namespace coresg {

inline std::uint32_t pad_dim(std::uint32_t d) { return (d + 7u) & ~7u; }

/// Non-owning view over a contiguous block of padded rows.
struct DataView {
    const float* data = nullptr;
    std::uint32_t dim = 0;     // logical dimensionality
    std::uint32_t stride = 0;  // padded row length in floats
    std::size_t n = 0;

    const float* row(std::size_t i) const { return data + i * stride; }
};

class Dataset {
public:
    Dataset() = default;
    Dataset(std::uint32_t dim, MetricKind metric) : dim_(dim), stride_(pad_dim(dim)), metric_(metric) {}

    std::uint32_t dim() const { return dim_; }
    std::uint32_t stride() const { return stride_; }
    std::size_t size() const { return n_; }
    MetricKind metric() const { return metric_; }
    void set_metric(MetricKind m) { metric_ = m; }

    const float* row(std::size_t i) const { return data_.data() + i * stride_; }
    float* mutable_row(std::size_t i) { return data_.data() + i * stride_; }
    const float* data() const { return data_.data(); }

    DataView view() const { return DataView{data_.data(), dim_, stride_, n_}; }

    void reserve(std::size_t rows) { data_.reserve(rows * stride_); }

    /// Append `count` rows of `dim()` floats each. Returns the id of the first
    /// appended row, which is what the incremental pipeline uses to identify
    /// "new" data.
    idx_t append(const float* rows, std::size_t count) {
        if (dim_ == 0) throw std::runtime_error("Dataset: dimensionality not set");
        const idx_t first = static_cast<idx_t>(n_);
        data_.resize((n_ + count) * stride_, 0.0f);
        for (std::size_t i = 0; i < count; ++i) {
            float* dst = data_.data() + (n_ + i) * stride_;
            std::memcpy(dst, rows + i * dim_, dim_ * sizeof(float));
            // Tail already zeroed by resize().
            if (metric_needs_normalisation(metric_)) normalise(dst);
        }
        n_ += count;
        if (n_ > kInvalidIdx) throw std::runtime_error("Dataset: too many points for idx_t");
        return first;
    }

    idx_t append(const std::vector<float>& rows) {
        if (dim_ == 0 || rows.size() % dim_ != 0) {
            throw std::runtime_error("Dataset::append: buffer is not a multiple of dim");
        }
        return append(rows.data(), rows.size() / dim_);
    }

    /// Copy an arbitrary subset of rows into a fresh contiguous block. Used to
    /// stage an NN-Descent instance (a sample of the base data plus the new
    /// points) for a device.
    std::vector<float> gather(const std::vector<idx_t>& ids) const {
        std::vector<float> out(ids.size() * stride_, 0.0f);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            std::memcpy(out.data() + i * stride_, row(ids[i]), stride_ * sizeof(float));
        }
        return out;
    }

private:
    void normalise(float* v) const {
        double s = 0.0;
        for (std::uint32_t i = 0; i < dim_; ++i) s += static_cast<double>(v[i]) * v[i];
        if (s <= 0.0) return;
        const float inv = static_cast<float>(1.0 / std::sqrt(s));
        for (std::uint32_t i = 0; i < dim_; ++i) v[i] *= inv;
    }

    std::vector<float> data_;
    std::uint32_t dim_ = 0;
    std::uint32_t stride_ = 0;
    std::size_t n_ = 0;
    MetricKind metric_ = MetricKind::L2;
};

}  // namespace coresg
