#include "coresg/io.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "coresg/coresg.hpp"

namespace coresg {
namespace {

std::string lowercase_ext(const std::string& path) {
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return "";
    std::string ext = path.substr(dot);
    for (char& c : ext) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    return ext;
}

}  // namespace

RawData read_fvecs(const std::string& path, std::size_t max_rows) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    RawData out;
    std::int32_t dim = 0;
    while (in.read(reinterpret_cast<char*>(&dim), 4)) {
        if (dim <= 0 || dim > (1 << 20)) throw std::runtime_error("bad fvecs dim in " + path);
        if (out.dim == 0) out.dim = static_cast<std::uint32_t>(dim);
        if (static_cast<std::uint32_t>(dim) != out.dim) {
            throw std::runtime_error("ragged fvecs in " + path);
        }
        const std::size_t base = out.values.size();
        out.values.resize(base + out.dim);
        in.read(reinterpret_cast<char*>(out.values.data() + base), out.dim * 4);
        if (!in) throw std::runtime_error("truncated fvecs in " + path);
        ++out.n;
        if (max_rows && out.n >= max_rows) break;
    }
    return out;
}

RawData read_bvecs(const std::string& path, std::size_t max_rows) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    RawData out;
    std::int32_t dim = 0;
    std::vector<unsigned char> buf;
    while (in.read(reinterpret_cast<char*>(&dim), 4)) {
        if (dim <= 0 || dim > (1 << 20)) throw std::runtime_error("bad bvecs dim in " + path);
        if (out.dim == 0) out.dim = static_cast<std::uint32_t>(dim);
        buf.resize(out.dim);
        in.read(reinterpret_cast<char*>(buf.data()), out.dim);
        if (!in) throw std::runtime_error("truncated bvecs in " + path);
        out.values.reserve(out.values.size() + out.dim);
        for (std::uint32_t i = 0; i < out.dim; ++i) out.values.push_back(float(buf[i]));
        ++out.n;
        if (max_rows && out.n >= max_rows) break;
    }
    return out;
}

RawData read_raw_f32(const std::string& path, std::uint32_t dim, std::size_t max_rows) {
    if (dim == 0) throw std::runtime_error("read_raw_f32 needs an explicit dim");
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    const std::streamsize bytes = in.tellg();
    in.seekg(0);
    std::size_t rows = static_cast<std::size_t>(bytes) / (std::size_t(dim) * 4);
    if (max_rows && rows > max_rows) rows = max_rows;
    RawData out;
    out.dim = dim;
    out.n = rows;
    out.values.resize(rows * dim);
    in.read(reinterpret_cast<char*>(out.values.data()),
            static_cast<std::streamsize>(out.values.size() * 4));
    if (!in) throw std::runtime_error("truncated raw file " + path);
    return out;
}

RawData read_csv(const std::string& path, char delimiter, bool skip_header,
                 bool skip_first_column, bool skip_last_column, std::size_t max_rows) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    RawData out;
    std::string line;
    if (skip_header) std::getline(in, line);

    std::vector<float> row;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        row.clear();
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, delimiter)) cells.push_back(cell);
        std::size_t lo = skip_first_column ? 1 : 0;
        std::size_t hi = cells.size() - (skip_last_column ? 1 : 0);
        for (std::size_t i = lo; i < hi; ++i) {
            try {
                row.push_back(std::stof(cells[i]));
            } catch (...) {
                throw std::runtime_error("non-numeric cell in " + path + ": '" + cells[i] + "'");
            }
        }
        if (row.empty()) continue;
        if (out.dim == 0) out.dim = static_cast<std::uint32_t>(row.size());
        if (row.size() != out.dim) throw std::runtime_error("ragged csv in " + path);
        out.values.insert(out.values.end(), row.begin(), row.end());
        ++out.n;
        if (max_rows && out.n >= max_rows) break;
    }
    return out;
}

RawData read_any(const std::string& path, std::uint32_t raw_dim, std::size_t max_rows) {
    const std::string ext = lowercase_ext(path);
    if (ext == ".fvecs") return read_fvecs(path, max_rows);
    if (ext == ".bvecs") return read_bvecs(path, max_rows);
    if (ext == ".bin" || ext == ".raw" || ext == ".f32") return read_raw_f32(path, raw_dim, max_rows);
    if (ext == ".tsv") return read_csv(path, '\t', false, false, false, max_rows);
    if (ext == ".csv") return read_csv(path, ',', false, false, false, max_rows);
    throw std::runtime_error("unknown data format for " + path +
                             " (use .fvecs/.bvecs/.csv/.tsv/.bin)");
}

void write_edges_tsv(const std::string& path, const std::vector<Edge>& edges) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << "u\tv\tweight\n";
    out.setf(std::ios::scientific);
    out.precision(9);
    for (const Edge& e : edges) out << e.u << '\t' << e.v << '\t' << e.w << '\n';
}

void write_labels_tsv(const std::string& path, const std::vector<idx_t>& labels) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << "id\tlabel\n";
    for (std::size_t i = 0; i < labels.size(); ++i) out << i << '\t' << labels[i] << '\n';
}

// ---------------------------------------------------------------------------
// Synthetic data
// ---------------------------------------------------------------------------

std::vector<float> make_blobs(std::size_t n, std::uint32_t dim, std::size_t clusters,
                              float separation, std::uint64_t seed, std::vector<int>* labels,
                              bool varying_density) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    if (clusters == 0) clusters = 1;
    std::vector<std::vector<float>> centres(clusters, std::vector<float>(dim));
    std::vector<float> sigma(clusters, 1.0f);
    for (std::size_t c = 0; c < clusters; ++c) {
        for (std::uint32_t d = 0; d < dim; ++d) centres[c][d] = separation * gauss(rng);
        // Densities spanning an order of magnitude: this is precisely the
        // regime where one mpts value cannot serve the whole data set, i.e.
        // what CORE-SG exists for.
        sigma[c] = varying_density ? (0.15f + 1.35f * unit(rng)) : 1.0f;
    }

    std::vector<float> out(n * dim);
    if (labels) labels->assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t c = rng() % clusters;
        if (labels) (*labels)[i] = static_cast<int>(c);
        for (std::uint32_t d = 0; d < dim; ++d) {
            out[i * dim + d] = centres[c][d] + sigma[c] * gauss(rng);
        }
    }
    return out;
}

std::vector<float> make_uniform(std::size_t n, std::uint32_t dim, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> out(n * dim);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = u(rng);
    return out;
}

// ---------------------------------------------------------------------------
// CoreSG serialisation
// ---------------------------------------------------------------------------

namespace {

constexpr char kMagic[8] = {'C', 'O', 'R', 'E', 'S', 'G', '0', '1'};

template <typename T>
void write_pod(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
void read_pod(std::istream& is, T* v) {
    is.read(reinterpret_cast<char*>(v), sizeof(T));
    if (!is) throw std::runtime_error("truncated coresg file");
}

template <typename T>
void write_vec(std::ostream& os, const std::vector<T>& v) {
    const std::uint64_t count = v.size();
    write_pod(os, count);
    if (count) {
        os.write(reinterpret_cast<const char*>(v.data()),
                 static_cast<std::streamsize>(count * sizeof(T)));
    }
}

template <typename T>
void read_vec(std::istream& is, std::vector<T>* v) {
    std::uint64_t count = 0;
    read_pod(is, &count);
    v->resize(static_cast<std::size_t>(count));
    if (count) {
        is.read(reinterpret_cast<char*>(v->data()),
                static_cast<std::streamsize>(count * sizeof(T)));
        if (!is) throw std::runtime_error("truncated coresg file");
    }
}

}  // namespace

void CoreSG::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.write(kMagic, 8);

    write_pod(out, std::uint32_t(params_.k_max));
    write_pod(out, std::uint32_t(data_.dim()));
    write_pod(out, std::uint32_t(data_.stride()));
    write_pod(out, std::uint32_t(static_cast<std::uint8_t>(params_.metric)));
    write_pod(out, std::uint32_t(static_cast<std::uint8_t>(params_.missing_mode)));
    write_pod(out, std::uint64_t(data_.size()));

    // Raw padded rows: keeps load() a straight memcpy and preserves the
    // normalisation already applied for cosine.
    const std::uint64_t floats = std::uint64_t(data_.size()) * data_.stride();
    write_pod(out, floats);
    if (floats) {
        out.write(reinterpret_cast<const char*>(data_.data()),
                  static_cast<std::streamsize>(floats * sizeof(float)));
    }

    write_vec(out, knn_.ids_storage());
    write_vec(out, knn_.dist_storage());
    write_vec(out, knn_.flag_storage());
    write_vec(out, knn_.count_storage());

    write_vec(out, missing_);
    write_pod(out, std::uint64_t(repair_count_));
    if (!out) throw std::runtime_error("write failed for " + path);
}

void CoreSG::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    char magic[8] = {0};
    in.read(magic, 8);
    if (!in || std::memcmp(magic, kMagic, 8) != 0) {
        throw std::runtime_error(path + " is not a coresg file");
    }

    std::uint32_t k_max = 0, dim = 0, stride = 0, metric = 0, mode = 0;
    std::uint64_t n = 0, floats = 0;
    read_pod(in, &k_max);
    read_pod(in, &dim);
    read_pod(in, &stride);
    read_pod(in, &metric);
    read_pod(in, &mode);
    read_pod(in, &n);

    params_.k_max = k_max;
    params_.metric = static_cast<MetricKind>(metric);
    params_.missing_mode = static_cast<MissingEdgeMode>(mode);

    read_pod(in, &floats);
    std::vector<float> padded(static_cast<std::size_t>(floats));
    if (floats) {
        in.read(reinterpret_cast<char*>(padded.data()),
                static_cast<std::streamsize>(floats * sizeof(float)));
        if (!in) throw std::runtime_error("truncated coresg payload in " + path);
    }

    // Rebuild the Dataset by stripping the padding back out; append() re-pads
    // and (for cosine) re-normalises, which is idempotent here.
    data_ = Dataset(dim, params_.metric);
    data_.reserve(static_cast<std::size_t>(n));
    std::vector<float> row(dim);
    for (std::uint64_t i = 0; i < n; ++i) {
        std::memcpy(row.data(), padded.data() + i * stride, dim * sizeof(float));
        data_.append(row.data(), 1);
    }
    std::vector<float>().swap(padded);

    knn_.reset(static_cast<idx_t>(n), k_max);
    read_vec(in, &knn_.ids_storage());
    read_vec(in, &knn_.dist_storage());
    read_vec(in, &knn_.flag_storage());
    read_vec(in, &knn_.count_storage());
    knn_.ensure_locks();

    read_vec(in, &missing_);
    std::uint64_t repair = 0;
    read_pod(in, &repair);
    repair_count_ = static_cast<std::size_t>(repair);

    // Hubness is derived state: recount rather than serialise, so a file
    // stays valid if the scoring rule is ever refined.
    conn_.reset(static_cast<idx_t>(n));
    conn_.rebuild(knn_, pool_.get());

    hub_.detach(&knn_);
    hub_.set_params(params_.hub);
    hub_.build(knn_, pool_.get());
    hub_.set_components(&conn_.labels(), conn_.components());
    const std::size_t edges = knn_.undirected_edge_count(pool_.get());
    hub_.rescore_all(knn_, data_.dim(), edges, pool_.get());
    hub_.set_track_removals(params_.incremental_connectivity);
    hub_.attach(&knn_);
    built_ = true;
}

}  // namespace coresg
