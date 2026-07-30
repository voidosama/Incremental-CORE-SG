#include "coresg/incremental_knn.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>

#include "coresg/resource.hpp"
#include "coresg/thread_pool.hpp"

namespace coresg {
namespace {

using Clock = std::chrono::steady_clock;
inline double since(const Clock::time_point& t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

/// Partial Fisher-Yates: the first `count` entries of `pool` become a uniform
/// sample without replacement. O(count) per call, no allocation.
void partial_shuffle(std::vector<idx_t>& pool, std::size_t count, std::mt19937_64& rng) {
    const std::size_t n = pool.size();
    const std::size_t take = std::min(count, n);
    for (std::size_t i = 0; i < take; ++i) {
        std::uniform_int_distribution<std::size_t> d(i, n - 1);
        const std::size_t j = d(rng);
        std::swap(pool[i], pool[j]);
    }
}

}  // namespace

std::vector<IncrementalKnn::Instance> IncrementalKnn::make_instances(idx_t n_base, idx_t first_new,
                                                                    idx_t n_total,
                                                                    std::uint64_t seed) const {
    std::vector<Instance> out;
    const std::size_t n_new = n_total - first_new;

    std::uint32_t s = params_.instances ? params_.instances : 1u;
    if (n_base < params_.min_base_for_split) s = 1;

    std::mt19937_64 rng(seed);
    std::vector<idx_t> base(n_base);
    std::iota(base.begin(), base.end(), idx_t(0));
    std::shuffle(base.begin(), base.end(), rng);

    if (params_.sampling == InstanceSampling::Partition) {
        const std::size_t block = (n_base + s - 1) / (s ? s : 1);
        for (std::uint32_t i = 0; i < s; ++i) {
            const std::size_t lo = std::size_t(i) * block;
            if (lo >= n_base && i > 0) break;
            const std::size_t hi = std::min<std::size_t>(lo + block, n_base);
            Instance inst;
            inst.nodes.assign(base.begin() + lo, base.begin() + hi);

            // A sprinkle of points from the other blocks so that instances
            // are not blind to each other's regions. Draws that land back in
            // this block are removed by the unique() below.
            const std::size_t extra =
                static_cast<std::size_t>(params_.overlap_fraction * double(hi - lo));
            for (std::size_t t = 0; t < extra && n_base > 0; ++t) {
                inst.nodes.push_back(base[rng() % n_base]);
            }
            std::sort(inst.nodes.begin(), inst.nodes.end());
            inst.nodes.erase(std::unique(inst.nodes.begin(), inst.nodes.end()), inst.nodes.end());

            inst.owned_begin = inst.nodes.size();
            for (std::size_t j = 0; j < n_new; ++j) {
                inst.nodes.push_back(static_cast<idx_t>(first_new + j));
            }
            out.push_back(std::move(inst));
        }
    } else {
        double frac = params_.sample_fraction;
        if (!(frac > 0.0)) frac = 1.0 / double(s);
        const std::size_t take = static_cast<std::size_t>(frac * double(n_base));
        for (std::uint32_t i = 0; i < s; ++i) {
            partial_shuffle(base, take, rng);
            Instance inst;
            inst.nodes.assign(base.begin(), base.begin() + std::min(take, base.size()));
            std::sort(inst.nodes.begin(), inst.nodes.end());
            inst.owned_begin = inst.nodes.size();
            for (std::size_t j = 0; j < n_new; ++j) {
                inst.nodes.push_back(static_cast<idx_t>(first_new + j));
            }
            out.push_back(std::move(inst));
        }
    }
    return out;
}

std::vector<IncrementalKnn::Instance> IncrementalKnn::make_build_instances(
    idx_t n_total, std::uint64_t seed) const {
    std::vector<Instance> out;
    std::uint32_t s = params_.instances ? params_.instances : 1u;
    if (n_total < params_.min_base_for_split) s = 1;

    if (s == 1) {
        Instance inst;
        inst.nodes.resize(n_total);
        std::iota(inst.nodes.begin(), inst.nodes.end(), idx_t(0));
        inst.owned_begin = 0;
        out.push_back(std::move(inst));
        return out;
    }

    std::mt19937_64 rng(seed);
    std::vector<idx_t> perm(n_total);
    std::iota(perm.begin(), perm.end(), idx_t(0));
    std::shuffle(perm.begin(), perm.end(), rng);

    const std::size_t block = (n_total + s - 1) / s;
    for (std::uint32_t i = 0; i < s; ++i) {
        const std::size_t lo = std::size_t(i) * block;
        if (lo >= n_total) break;
        const std::size_t hi = std::min<std::size_t>(lo + block, n_total);
        Instance inst;
        inst.nodes.assign(perm.begin() + lo, perm.begin() + hi);
        const std::size_t extra =
            static_cast<std::size_t>(params_.overlap_fraction * double(hi - lo));
        for (std::size_t t = 0; t < extra; ++t) inst.nodes.push_back(perm[rng() % n_total]);
        std::sort(inst.nodes.begin(), inst.nodes.end());
        inst.nodes.erase(std::unique(inst.nodes.begin(), inst.nodes.end()), inst.nodes.end());
        inst.owned_begin = 0;  // every node in a build instance is owned
        out.push_back(std::move(inst));
    }
    return out;
}

void IncrementalKnn::stage_job(const Dataset& data, const KnnGraph& graph, const Instance& inst,
                               const NNDescentParams& nnd, std::vector<idx_t>* g2l,
                               KnnJob* job) const {
    const std::size_t m = inst.nodes.size();
    job->vectors = data.gather(inst.nodes);
    job->dim = data.dim();
    job->stride = data.stride();
    job->local_n = m;
    job->metric = data.metric();
    job->params = nnd;
    job->params.k = graph.k();
    job->graph.reset(static_cast<idx_t>(m), graph.k());

    for (std::size_t i = 0; i < m; ++i) (*g2l)[inst.nodes[i]] = static_cast<idx_t>(i);

    if (params_.warm_start) {
        // Inherit the edges whose other endpoint also made it into this
        // instance. Flagged "new" so the very first local join uses them.
        for (std::size_t i = 0; i < m; ++i) {
            const idx_t gid = inst.nodes[i];
            const idx_t* I = graph.ids(gid);
            const dist_t* D = graph.dists(gid);
            const std::uint32_t c = graph.degree(gid);
            for (std::uint32_t j = 0; j < c; ++j) {
                const idx_t lj = (*g2l)[I[j]];
                if (lj != kInvalidIdx) {
                    job->graph.try_insert(static_cast<idx_t>(i), lj, D[j], true);
                }
            }
        }
    }

    for (std::size_t i = 0; i < m; ++i) (*g2l)[inst.nodes[i]] = kInvalidIdx;
}

std::size_t IncrementalKnn::merge_job(const KnnJob& job, const Instance& inst, idx_t first_new,
                                      KnnGraph* graph, std::size_t* candidates) const {
    const std::size_t m = inst.nodes.size();
    const unsigned nt = (pool_ && pool_->size() > 1) ? pool_->size() : 1u;
    std::vector<std::size_t> applied(nt, 0);
    std::vector<std::size_t> offered(nt, 0);

    auto body = [&](std::size_t lo, std::size_t hi, int slot) {
        std::size_t a = 0, o = 0;
        for (std::size_t i = lo; i < hi; ++i) {
            const idx_t gid = inst.nodes[i];
            const idx_t* I = job.graph.ids(static_cast<idx_t>(i));
            const dist_t* D = job.graph.dists(static_cast<idx_t>(i));
            const std::uint32_t c = job.graph.degree(static_cast<idx_t>(i));
            const bool gid_is_new = gid >= first_new;
            for (std::uint32_t j = 0; j < c; ++j) {
                const idx_t gj = inst.nodes[I[j]];
                const bool touches_new = gid_is_new || gj >= first_new;
                if (!touches_new && !params_.apply_base_base) continue;
                ++o;
                // Symmetric: an instance edge improves both endpoints' lists.
                if (graph->try_insert_mt(gid, gj, D[j], true).inserted) ++a;
                if (graph->try_insert_mt(gj, gid, D[j], true).inserted) ++a;
            }
        }
        applied[slot] += a;
        offered[slot] += o;
    };

    if (nt > 1) {
        pool_->parallel_ranges(m, body);
    } else {
        body(0, m, 0);
    }

    std::size_t a_total = 0, o_total = 0;
    for (unsigned t = 0; t < nt; ++t) {
        a_total += applied[t];
        o_total += offered[t];
    }
    if (candidates) *candidates += o_total;
    return a_total;
}

std::vector<idx_t> IncrementalKnn::refinement_set(const KnnGraph& graph, idx_t first_new,
                                                  idx_t n_total,
                                                  IncrementalKnnStats* stats) const {
    // Ceiling on the work a batch may do. Rings are appended in order, so
    // truncating the tail drops the least relevant nodes first and the new
    // points always survive.
    std::size_t cap = n_total;
    if (params_.refine_max_fraction > 0.0 && params_.refine_max_fraction < 1.0) {
        cap = static_cast<std::size_t>(params_.refine_max_fraction * double(n_total));
    }
    if (params_.refine_max_nodes && params_.refine_max_nodes < cap) {
        cap = params_.refine_max_nodes;
    }
    const std::size_t n_new = n_total - first_new;
    if (cap < n_new) cap = n_new;  // never drop a new point

    std::vector<std::uint8_t> mark(n_total, 0);
    std::vector<idx_t> active;
    active.reserve(std::min<std::size_t>(cap, std::size_t(n_new) * (graph.k() + 1)) + 1);

    for (idx_t v = first_new; v < n_total; ++v) {
        if (!mark[v]) {
            mark[v] = 1;
            active.push_back(v);
        }
    }
    const std::size_t ring0 = active.size();

    auto expand = [&](std::size_t lo, std::size_t hi) {
        for (std::size_t t = lo; t < hi && active.size() < cap; ++t) {
            const idx_t v = active[t];
            const idx_t* I = graph.ids(v);
            const std::uint32_t c = graph.degree(v);
            for (std::uint32_t j = 0; j < c && active.size() < cap; ++j) {
                const idx_t q = I[j];
                if (q < n_total && !mark[q]) {
                    mark[q] = 1;
                    active.push_back(q);
                }
            }
        }
    };

    expand(0, ring0);
    const std::size_t ring1 = active.size();
    if (params_.refine_second_ring) expand(ring0, ring1);

    if (stats) {
        stats->refine_ring0 = ring0;
        stats->refine_ring1 = ring1 - ring0;
        stats->refine_capped = active.size() >= cap && cap < std::size_t(n_total);
    }
    return active;
}

IncrementalKnnStats IncrementalKnn::build(const Dataset& data, KnnGraph* graph) {
    IncrementalKnnStats st;
    const idx_t n = static_cast<idx_t>(data.size());
    st.new_points = n;
    if (n == 0) return st;

    graph->grow(n);
    graph->ensure_locks();

    if (params_.brute_force_limit && n <= params_.brute_force_limit) {
        const auto t0 = Clock::now();
        brute_force_knn(data.view(), data.metric(), graph, pool_);
        st.instances = 1;
        st.instance_nodes = n;
        st.instance_seconds = since(t0);
        for (idx_t v = 0; v < n; ++v) st.applied_edges += graph->degree(v);
        st.candidate_edges = st.applied_edges;
        return st;
    }

    std::vector<Instance> instances = make_build_instances(n, params_.seed);
    st.instances = instances.size();

    if (instances.size() == 1) {
        // Nothing to partition: one descent over the whole set. Still goes
        // through the backend so the GPU path is used when available.
        std::vector<KnnJob> jobs(1);
        KnnJob& job = jobs[0];
        job.vectors.assign(data.data(), data.data() + std::size_t(n) * data.stride());
        job.dim = data.dim();
        job.stride = data.stride();
        job.local_n = n;
        job.metric = data.metric();
        job.params = params_.instance_nnd;
        job.params.k = graph->k();
        job.graph.reset(n, graph->k());
        job.job_id = 0;

        const auto t0 = Clock::now();
        JobScheduler sched(backend_);
        sched.run_all(&jobs);
        st.instance_seconds = since(t0);
        st.per_instance.push_back(job.stats);
        // The single-instance fast path bypasses the multi-instance loop, so
        // it has to account for itself or every un-partitioned run reports
        // "0 jobs on device" and looks like a silent host demotion.
        if (job.ran_on == DeviceKind::Cuda) ++st.jobs_on_device; else ++st.jobs_on_host;
        if (job.out_of_memory) ++st.jobs_out_of_memory;
        st.largest_instance = n;
        st.largest_instance_bytes = instance_device_bytes(
            n, data.stride(), graph->k(), params_.instance_nnd.sample);
        st.instance_nodes = n;

        // Adopt the result wholesale: same node ids, same k.
        graph->ids_storage() = job.graph.ids_storage();
        graph->dist_storage() = job.graph.dist_storage();
        graph->flag_storage() = job.graph.flag_storage();
        graph->count_storage() = job.graph.count_storage();
        for (idx_t v = 0; v < n; ++v) st.applied_edges += graph->degree(v);
        st.candidate_edges = st.applied_edges;
        return st;
    }

    // Partitioned build: independent blocks, merge, then a global refinement.
    std::vector<KnnJob> jobs(instances.size());
    std::vector<idx_t> g2l(n, kInvalidIdx);
    for (std::size_t i = 0; i < instances.size(); ++i) {
        stage_job(data, *graph, instances[i], params_.instance_nnd, &g2l, &jobs[i]);
        jobs[i].job_id = static_cast<int>(i);
        st.instance_nodes += instances[i].nodes.size();
        // The biggest instance is what has to fit on a device; record it so a
        // capacity claim rests on a measured figure rather than an estimate.
        if (instances[i].nodes.size() > st.largest_instance) {
            st.largest_instance = instances[i].nodes.size();
            st.largest_instance_bytes = instance_device_bytes(
                instances[i].nodes.size(), data.stride(), graph->k(),
                params_.instance_nnd.sample);
        }
    }

    const auto t_inst = Clock::now();
    {
        JobScheduler sched(backend_);
        sched.run_all(&jobs);
    }
    st.instance_seconds = since(t_inst);
    for (const KnnJob& j : jobs) {
        st.per_instance.push_back(j.stats);
        if (j.ran_on == DeviceKind::Cuda) ++st.jobs_on_device; else ++st.jobs_on_host;
        if (j.out_of_memory) ++st.jobs_out_of_memory;
    }

    const auto t_merge = Clock::now();
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        st.applied_edges += merge_job(jobs[i], instances[i], /*first_new=*/0, graph,
                                      &st.candidate_edges);
        std::vector<float>().swap(jobs[i].vectors);
    }
    st.merge_seconds = since(t_merge);

    const auto t_ref = Clock::now();
    NNDescentParams rp = params_.refine_nnd;
    rp.k = graph->k();
    st.refine_stats = nndescent(data.view(), data.metric(), graph, rp, pool_);
    st.refine_seconds = since(t_ref);
    st.refine_nodes = n;
    return st;
}

IncrementalKnnStats IncrementalKnn::insert(const Dataset& data, KnnGraph* graph, idx_t first_new) {
    IncrementalKnnStats st;
    const idx_t n_total = static_cast<idx_t>(data.size());
    if (n_total <= first_new) return st;

    st.base_points = first_new;
    st.new_points = n_total - first_new;

    graph->grow(n_total);
    graph->ensure_locks();

    if (params_.brute_force_limit && n_total <= params_.brute_force_limit) {
        // Exact reference path: recompute every neighbourhood. The existing
        // lists are kept and simply improved, so the observer still sees the
        // individual edge changes and the hubness index stays incremental.
        const auto t0 = Clock::now();
        brute_force_knn(data.view(), data.metric(), graph, pool_);
        st.instances = 1;
        st.instance_nodes = n_total;
        st.instance_seconds = since(t0);
        st.refine_nodes = n_total;
        return st;
    }

    std::vector<Instance> instances = make_instances(first_new, first_new, n_total, params_.seed);
    st.instances = instances.size();

    std::vector<KnnJob> jobs(instances.size());
    std::vector<idx_t> g2l(n_total, kInvalidIdx);
    for (std::size_t i = 0; i < instances.size(); ++i) {
        stage_job(data, *graph, instances[i], params_.instance_nnd, &g2l, &jobs[i]);
        jobs[i].job_id = static_cast<int>(i);
        st.instance_nodes += instances[i].nodes.size();
        // The biggest instance is what has to fit on a device; record it so a
        // capacity claim rests on a measured figure rather than an estimate.
        if (instances[i].nodes.size() > st.largest_instance) {
            st.largest_instance = instances[i].nodes.size();
            st.largest_instance_bytes = instance_device_bytes(
                instances[i].nodes.size(), data.stride(), graph->k(),
                params_.instance_nnd.sample);
        }
    }

    const auto t_inst = Clock::now();
    {
        JobScheduler sched(backend_);
        sched.run_all(&jobs);
    }
    st.instance_seconds = since(t_inst);
    for (const KnnJob& j : jobs) {
        st.per_instance.push_back(j.stats);
        if (j.ran_on == DeviceKind::Cuda) ++st.jobs_on_device; else ++st.jobs_on_host;
        if (j.out_of_memory) ++st.jobs_out_of_memory;
    }

    // Merge: each instance offers the neighbourhoods it found. Because the
    // global lists are sorted top-k, "collect the neighbourhoods of each new
    // point and keep the best k" is exactly repeated insertion.
    const auto t_merge = Clock::now();
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        st.applied_edges += merge_job(jobs[i], instances[i], first_new, graph,
                                      &st.candidate_edges);
        std::vector<float>().swap(jobs[i].vectors);
    }
    st.merge_seconds = since(t_merge);

    // Restricted refinement over the new points and their neighbourhoods.
    const auto t_ref = Clock::now();
    std::vector<idx_t> active = refinement_set(*graph, first_new, n_total, &st);
    st.refine_nodes = active.size();
    NNDescentParams rp = params_.refine_nnd;
    rp.k = graph->k();
    st.refine_stats = nndescent(data.view(), data.metric(), graph, rp, pool_, &active);
    st.refine_seconds = since(t_ref);

    if (params_.verbose) {
        std::printf(
            "[incremental] +%zu points over %zu instances: instances %.3fs, merge %.3fs, "
            "refine %.3fs (%zu active nodes)\n",
            st.new_points, st.instances, st.instance_seconds, st.merge_seconds,
            st.refine_seconds, st.refine_nodes);
    }
    return st;
}

}  // namespace coresg
