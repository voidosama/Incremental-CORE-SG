#pragma once
// Data loading and result serialisation.

#include <cstdint>
#include <string>
#include <vector>

#include "coresg/types.hpp"

namespace coresg {

struct RawData {
    std::vector<float> values;  ///< n * dim, row major
    std::uint32_t dim = 0;
    std::size_t n = 0;
};

/// `.fvecs` (int32 dim followed by dim floats, repeated).
RawData read_fvecs(const std::string& path, std::size_t max_rows = 0);
/// `.bvecs` (int32 dim followed by dim uint8, repeated); values are widened.
RawData read_bvecs(const std::string& path, std::size_t max_rows = 0);
/// Headerless float32 matrix; `dim` must be supplied.
RawData read_raw_f32(const std::string& path, std::uint32_t dim, std::size_t max_rows = 0);
/// Delimited text, one row per line. Non-numeric leading/trailing columns are
/// skipped when `skip_first_column` / `skip_last_column` are set.
RawData read_csv(const std::string& path, char delimiter = ',', bool skip_header = false,
                 bool skip_first_column = false, bool skip_last_column = false,
                 std::size_t max_rows = 0);

/// Dispatch on the file extension (.fvecs/.bvecs/.csv/.tsv/.bin).
RawData read_any(const std::string& path, std::uint32_t raw_dim = 0, std::size_t max_rows = 0);

void write_edges_tsv(const std::string& path, const std::vector<Edge>& edges);
void write_labels_tsv(const std::string& path, const std::vector<idx_t>& labels);

// ---------------------------------------------------------------------------
// Synthetic data, used by the tests and the benchmark driver.
// ---------------------------------------------------------------------------

/// Gaussian blobs of varying density -- the case CORE-SG exists for, where a
/// single global mpts cannot describe every region.
std::vector<float> make_blobs(std::size_t n, std::uint32_t dim, std::size_t clusters,
                              float separation, std::uint64_t seed,
                              std::vector<int>* labels = nullptr,
                              bool varying_density = true);

std::vector<float> make_uniform(std::size_t n, std::uint32_t dim, std::uint64_t seed);

}  // namespace coresg
