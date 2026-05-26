// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Tool: eval_recall_brute_force
// Loads a serialised hgraph index and evaluates recall@k against ground truth
// computed on-the-fly via brute-force L2 distances over the original
// vector_data.bin. Uses label.bin to map brute-force row indices to the
// external labels returned by KnnSearch.

#include <vsag/vsag.h>

#include <argparse/argparse.hpp>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <algorithm>

using namespace vsag;

namespace {

uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

double ns_to_s(uint64_t ns) {
    return static_cast<double>(ns) / 1e9;
}

void log_stage(const std::string& msg) {
    std::cerr << "[eval_recall_bf] " << msg << std::endl;
}

struct MmapVectorData {
    int fd = -1;
    size_t bytes = 0;
    const float* data = nullptr;
    size_t row_count = 0;
    ~MmapVectorData() {
        if (data != nullptr && bytes > 0) {
            ::munmap(const_cast<float*>(data), bytes);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }
};

std::unique_ptr<MmapVectorData>
mmap_vector_data(const std::string& path, int64_t dim) {
    auto holder = std::make_unique<MmapVectorData>();
    holder->fd = ::open(path.c_str(), O_RDONLY);
    if (holder->fd < 0) {
        throw std::runtime_error("failed to open vector file: " + path);
    }
    struct stat st {};
    if (::fstat(holder->fd, &st) != 0) {
        throw std::runtime_error("failed to stat vector file: " + path);
    }
    if (st.st_size <= 0 || (st.st_size % static_cast<off_t>(sizeof(float))) != 0) {
        throw std::runtime_error("vector_data.bin not aligned to float32: " + path);
    }
    const auto float_count = static_cast<size_t>(st.st_size / sizeof(float));
    if (float_count % static_cast<size_t>(dim) != 0) {
        throw std::runtime_error("vector_data.bin float count not aligned to dim: " + path);
    }
    holder->bytes = static_cast<size_t>(st.st_size);
    holder->row_count = float_count / static_cast<size_t>(dim);
    void* mapped = ::mmap(nullptr, holder->bytes, PROT_READ, MAP_SHARED, holder->fd, 0);
    if (mapped == MAP_FAILED) {
        throw std::runtime_error("mmap failed for " + path + ": " + std::strerror(errno));
    }
    holder->data = static_cast<const float*>(mapped);
    ::madvise(const_cast<float*>(holder->data), holder->bytes, MADV_SEQUENTIAL);
    return holder;
}

inline float l2_sqr(const float* a, const float* b, int64_t dim) {
    float sum = 0.0F;
    for (int64_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

struct TopK {
    int64_t k;
    std::priority_queue<std::pair<float, int64_t>> heap;
    explicit TopK(int64_t kk) : k(kk) {}
    void push(float dist, int64_t id) {
        if (static_cast<int64_t>(heap.size()) < k) {
            heap.emplace(dist, id);
        } else if (dist < heap.top().first) {
            heap.pop();
            heap.emplace(dist, id);
        }
    }
    std::vector<std::pair<float, int64_t>> sorted_ascending() {
        std::vector<std::pair<float, int64_t>> out;
        out.reserve(heap.size());
        while (not heap.empty()) {
            out.push_back(heap.top());
            heap.pop();
        }
        std::reverse(out.begin(), out.end());
        return out;
    }
};

std::vector<int64_t>
parse_query_indices(int64_t total_rows, int64_t query_count, int64_t query_stride) {
    if (query_count <= 0) throw std::runtime_error("query_count must be positive");
    if (query_count > total_rows) throw std::runtime_error("query_count > total_rows");
    std::vector<int64_t> indices;
    indices.reserve(query_count);
    if (query_stride <= 0) {
        const double step = static_cast<double>(total_rows) / static_cast<double>(query_count);
        for (int64_t i = 0; i < query_count; ++i) {
            indices.push_back(static_cast<int64_t>(static_cast<double>(i) * step));
        }
    } else {
        for (int64_t i = 0; i < query_count; ++i) {
            int64_t idx = i * query_stride;
            if (idx >= total_rows) break;
            indices.push_back(idx);
        }
    }
    return indices;
}

std::vector<std::vector<int64_t>>
brute_force_top_k(const float* base, int64_t total_rows, int64_t dim,
                  const float* queries, int64_t query_count, int64_t k, int threads) {
    if (threads <= 0) {
        threads = static_cast<int>(std::thread::hardware_concurrency());
        if (threads <= 0) threads = 8;
    }
    log_stage("brute_force start: rows=" + std::to_string(total_rows) +
              ", queries=" + std::to_string(query_count) + ", k=" + std::to_string(k) +
              ", threads=" + std::to_string(threads));

    std::vector<std::vector<TopK>> per_thread;
    per_thread.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        per_thread.emplace_back();
        per_thread[t].reserve(query_count);
        for (int64_t q = 0; q < query_count; ++q) {
            per_thread[t].emplace_back(k);
        }
    }

    std::atomic<int64_t> processed_blocks{0};
    constexpr int64_t kBlock = 8192;
    const int64_t block_count = (total_rows + kBlock - 1) / kBlock;

    auto worker = [&](int thread_id) {
        for (int64_t blk = thread_id; blk < block_count; blk += threads) {
            const int64_t row_begin = blk * kBlock;
            const int64_t row_end = std::min(total_rows, row_begin + kBlock);
            for (int64_t q = 0; q < query_count; ++q) {
                const float* qvec = queries + q * dim;
                auto& heap = per_thread[thread_id][q];
                for (int64_t r = row_begin; r < row_end; ++r) {
                    const float* base_vec = base + r * dim;
                    const float dist = l2_sqr(qvec, base_vec, dim);
                    heap.push(dist, r);
                }
            }
            const auto done = processed_blocks.fetch_add(1) + 1;
            if ((done % 256) == 0 || done == block_count) {
                log_stage("brute_force progress " + std::to_string(done) + "/" +
                          std::to_string(block_count));
            }
        }
    };

    std::vector<std::thread> ths;
    ths.reserve(threads);
    const auto bf_begin = now_ns();
    for (int t = 0; t < threads; ++t) ths.emplace_back(worker, t);
    for (auto& th : ths) th.join();
    log_stage("brute_force compute finished in " +
              std::to_string(ns_to_s(now_ns() - bf_begin)) + "s");

    std::vector<std::vector<int64_t>> result(query_count);
    for (int64_t q = 0; q < query_count; ++q) {
        TopK merged(k);
        for (int t = 0; t < threads; ++t) {
            for (auto& entry : per_thread[t][q].sorted_ascending()) {
                merged.push(entry.first, entry.second);
            }
        }
        auto sorted = merged.sorted_ascending();
        result[q].reserve(sorted.size());
        for (auto& entry : sorted) result[q].push_back(entry.second);
    }
    return result;
}

std::vector<int> parse_int_list(const std::string& csv) {
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        out.push_back(std::stoi(item));
    }
    return out;
}

std::string read_file_text(const std::string& path) {
    std::ifstream in(path);
    if (not in.is_open()) throw std::runtime_error("failed to open: " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int64_t extract_dim_from_json(const std::string& json) {
    auto pos = json.find("\"dim\"");
    if (pos == std::string::npos) return -1;
    auto colon = json.find(':', pos);
    if (colon == std::string::npos) return -1;
    int64_t value = 0;
    int sign = 1;
    size_t i = colon + 1;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) ++i;
    if (i < json.size() && json[i] == '-') { sign = -1; ++i; }
    bool has_digit = false;
    while (i < json.size() && std::isdigit(static_cast<unsigned char>(json[i]))) {
        value = value * 10 + (json[i] - '0');
        ++i;
        has_digit = true;
    }
    if (not has_digit) return -1;
    return sign * value;
}

std::vector<int64_t> read_labels_int64(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (not input.is_open()) throw std::runtime_error("failed to open label file: " + path);
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size < 0 || (size % static_cast<std::streamoff>(sizeof(int64_t))) != 0) {
        throw std::runtime_error("label.bin not aligned to int64: " + path);
    }
    std::vector<int64_t> labels(static_cast<size_t>(size / sizeof(int64_t)));
    if (not labels.empty()) {
        input.read(reinterpret_cast<char*>(labels.data()), size);
    }
    if (not input.good() && not input.eof()) {
        throw std::runtime_error("failed to read label file: " + path);
    }
    return labels;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        argparse::ArgumentParser parser("eval_recall_brute_force");
        parser.add_argument<std::string>("--index_path", "-i").required();
        parser.add_argument<std::string>("--vector_path", "-v").required();
        parser.add_argument<std::string>("--label_path", "-l").required();
        parser.add_argument<std::string>("--build_parameter", "-bp").required();
        parser.add_argument("--query_count").default_value(int64_t{1000}).scan<'i', int64_t>();
        parser.add_argument("--query_stride").default_value(int64_t{0}).scan<'i', int64_t>();
        parser.add_argument("--topk").default_value(int64_t{10}).scan<'i', int64_t>();
        parser.add_argument<std::string>("--ef_search").default_value(std::string("50,100,200,400,1000"));
        parser.add_argument("--brute_force_threads").default_value(100).scan<'i', int>();
        parser.add_argument("--search_threads").default_value(16).scan<'i', int>();
        parser.add_argument("--dim").default_value(int64_t{-1}).scan<'i', int64_t>();
        parser.parse_args(argc, argv);

        const auto index_path = parser.get<std::string>("--index_path");
        const auto vector_path = parser.get<std::string>("--vector_path");
        const auto label_path = parser.get<std::string>("--label_path");
        const auto build_parameter_arg = parser.get<std::string>("--build_parameter");
        const auto query_count = parser.get<int64_t>("--query_count");
        const auto query_stride = parser.get<int64_t>("--query_stride");
        const auto topk = parser.get<int64_t>("--topk");
        const auto ef_list = parse_int_list(parser.get<std::string>("--ef_search"));
        const auto bf_threads = parser.get<int>("--brute_force_threads");
        const auto search_threads = parser.get<int>("--search_threads");
        int64_t dim = parser.get<int64_t>("--dim");

        std::string build_param_json;
        {
            std::ifstream test(build_parameter_arg);
            if (test.is_open()) build_param_json = read_file_text(build_parameter_arg);
            else build_param_json = build_parameter_arg;
        }
        if (dim <= 0) {
            dim = extract_dim_from_json(build_param_json);
            if (dim <= 0) throw std::runtime_error("could not infer dim, pass --dim");
        }
        log_stage("dim=" + std::to_string(dim));

        log_stage("mmap vector file: " + vector_path);
        auto base_holder = mmap_vector_data(vector_path, dim);
        const float* base = base_holder->data;
        const int64_t total_rows = static_cast<int64_t>(base_holder->row_count);
        log_stage("base rows=" + std::to_string(total_rows));

        log_stage("loading labels: " + label_path);
        auto labels = read_labels_int64(label_path);
        if (static_cast<int64_t>(labels.size()) < total_rows) {
            throw std::runtime_error("label count < vector row count");
        }
        if (static_cast<int64_t>(labels.size()) > total_rows) {
            log_stage("label count > vector row count, truncating");
            labels.resize(static_cast<size_t>(total_rows));
        }

        auto query_indices = parse_query_indices(total_rows, query_count, query_stride);
        const int64_t actual_q = static_cast<int64_t>(query_indices.size());
        std::vector<float> queries(static_cast<size_t>(actual_q * dim));
        for (int64_t q = 0; q < actual_q; ++q) {
            std::memcpy(queries.data() + q * dim, base + query_indices[q] * dim,
                        static_cast<size_t>(dim) * sizeof(float));
        }
        log_stage("collected " + std::to_string(actual_q) + " queries");

        log_stage("create index from build_parameter");
        auto create_result = Factory::CreateIndex("hgraph", build_param_json);
        if (not create_result.has_value()) {
            throw std::runtime_error("CreateIndex failed: " + create_result.error().message);
        }
        auto index = create_result.value();

        log_stage("deserialize index from " + index_path);
        const auto deser_begin = now_ns();
        std::ifstream istr(index_path, std::ios::binary);
        if (not istr.is_open()) throw std::runtime_error("open index_path failed: " + index_path);
        auto deser_result = index->Deserialize(istr);
        if (not deser_result.has_value()) {
            throw std::runtime_error("Deserialize failed: " + deser_result.error().message);
        }
        log_stage("deserialize finished in " +
                  std::to_string(ns_to_s(now_ns() - deser_begin)) + "s");

        auto gt = brute_force_top_k(base, total_rows, dim, queries.data(),
                                    actual_q, topk, bf_threads);

        std::cout << "ef_search\trecall@" << topk << "\tavg_latency_us\twall_qps\n";

        std::mutex io_mu;
        for (int ef : ef_list) {
            std::ostringstream sp;
            sp << "{\"hgraph\":{\"ef_search\":" << ef << "}}";
            const std::string search_param = sp.str();

            std::atomic<int64_t> total_correct{0};
            std::atomic<uint64_t> total_search_ns{0};

            auto worker = [&](int64_t qbegin, int64_t qend) {
                for (int64_t qi = qbegin; qi < qend; ++qi) {
                    auto query_ds = Dataset::Make();
                    query_ds->NumElements(1)
                        ->Dim(dim)
                        ->Float32Vectors(queries.data() + qi * dim)
                        ->Owner(false);
                    const auto t0 = now_ns();
                    auto result = index->KnnSearch(query_ds, topk, search_param);
                    const auto t1 = now_ns();
                    if (not result.has_value()) {
                        std::lock_guard<std::mutex> lk(io_mu);
                        std::cerr << "knn search failed at qi=" << qi
                                  << ": " << result.error().message << std::endl;
                        continue;
                    }
                    total_search_ns.fetch_add(t1 - t0);
                    auto ds = result.value();
                    const auto* ids = ds->GetIds();
                    const auto count = ds->GetDim();
                    int64_t correct = 0;
                    for (int64_t i = 0; i < count; ++i) {
                        const int64_t got = ids[i];
                        for (int64_t g_row : gt[qi]) {
                            if (labels[g_row] == got) {
                                ++correct;
                                break;
                            }
                        }
                    }
                    total_correct.fetch_add(correct);
                }
            };

            const auto run_begin = now_ns();
            std::vector<std::thread> ths;
            ths.reserve(search_threads);
            const int64_t chunk = (actual_q + search_threads - 1) / search_threads;
            for (int t = 0; t < search_threads; ++t) {
                const int64_t b = std::min<int64_t>(actual_q, t * chunk);
                const int64_t e = std::min<int64_t>(actual_q, b + chunk);
                if (b >= e) break;
                ths.emplace_back(worker, b, e);
            }
            for (auto& th : ths) th.join();
            const auto wall_ns = now_ns() - run_begin;

            const double recall = static_cast<double>(total_correct.load()) /
                                  static_cast<double>(actual_q * topk);
            const double avg_latency_us =
                static_cast<double>(total_search_ns.load()) / actual_q / 1000.0;
            const double wall_qps =
                static_cast<double>(actual_q) / (static_cast<double>(wall_ns) / 1e9);

            std::cout << ef << '\t' << recall << '\t' << avg_latency_us << '\t'
                      << wall_qps << std::endl;
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return 1;
    }
}
