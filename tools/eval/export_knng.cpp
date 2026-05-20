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

#include <vsag/vsag.h>

#include <argparse/argparse.hpp>
#include <fmt/format.h>
#include <omp.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "eval_dataset.h"
#include "typing.h"

namespace {

void
check(bool condition, const std::string& message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

void
parse_args(argparse::ArgumentParser& parser, int argc, char** argv) {
    parser.add_argument("--datapath", "-d")
        .required()
        .help("The HDF5 dataset path. The tool reads the /train split as the query source.");
    parser.add_argument("--index_path", "-i")
        .required()
        .help("The serialized HGraph index path.");
    parser.add_argument("--create_params", "-c")
        .required()
        .help("The JSON create_params string used to create the HGraph index.");
    parser.add_argument("--search_params", "-s")
        .required()
        .help("The JSON search parameter string used for each base->base HGraph search.");
    parser.add_argument("--output_path", "-o")
        .required()
        .help("The output KNNG path. The file stores fixed-width uint32 rows without a header.");
    parser.add_argument("--topk", "-k")
        .default_value(70)
        .scan<'i', int>()
        .help("The number of neighbors to export per base vector.");
    parser.add_argument("--num_threads", "-t")
        .default_value(16)
        .scan<'i', int>()
        .help("The number of concurrent search threads.");
    parser.add_argument("--limit")
        .default_value(static_cast<uint64_t>(0))
        .scan<'u', uint64_t>()
        .help("Optional row limit for smoke tests. 0 exports all rows.");
    parser.add_argument("--progress_interval")
        .default_value(static_cast<uint64_t>(10000))
        .scan<'u', uint64_t>()
        .help("Print progress every N exported rows.");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        throw;
    }
}

vsag::IndexPtr
load_hgraph_index(const std::string& index_path, const std::string& create_params) {
    auto create_result = vsag::Factory::CreateIndex("hgraph", create_params);
    check(create_result.has_value(),
          fmt::format("failed to create hgraph index: {}", create_result.error().message));

    std::ifstream input(index_path, std::ios::binary);
    check(input.good(), fmt::format("failed to open index file: {}", index_path));

    auto index = create_result.value();
    auto deserialize_result = index->Deserialize(input);
    check(deserialize_result.has_value(),
          fmt::format("failed to deserialize index: {}", deserialize_result.error().message));
    return index;
}

int
open_output(const std::string& output_path, uint64_t row_count, uint64_t topk) {
    auto output_dir = std::filesystem::path(output_path).parent_path();
    if (not output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
    }

    int fd = ::open(output_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    check(fd >= 0,
          fmt::format("failed to open output file {}: {}", output_path, std::strerror(errno)));

    const auto total_bytes = static_cast<off_t>(row_count * topk * sizeof(vsag::InnerIdType));
    const auto resize_status = ::ftruncate(fd, total_bytes);
    if (resize_status != 0) {
        const auto error_message =
            fmt::format("failed to resize output file {}: {}", output_path, std::strerror(errno));
        ::close(fd);
        throw std::runtime_error(error_message);
    }
    return fd;
}

void
write_row(int fd,
          uint64_t row_id,
          const std::vector<vsag::InnerIdType>& row,
          uint64_t topk,
          const std::string& output_path) {
    const auto* bytes = reinterpret_cast<const char*>(row.data());
    const auto total_bytes = static_cast<size_t>(topk * sizeof(vsag::InnerIdType));
    size_t written = 0;
    while (written < total_bytes) {
        const auto offset = static_cast<off_t>(row_id * total_bytes + written);
        const auto result = ::pwrite(fd, bytes + written, total_bytes - written, offset);
        check(result >= 0,
              fmt::format("failed to write row {} into {}: {}",
                          row_id,
                          output_path,
                          std::strerror(errno)));
        written += static_cast<size_t>(result);
    }
}

}  // namespace

int
main(int argc, char** argv) {
    argparse::ArgumentParser parser("export_knng");
    parse_args(parser, argc, argv);

    const auto datapath = parser.get<std::string>("--datapath");
    const auto index_path = parser.get<std::string>("--index_path");
    const auto create_params = parser.get<std::string>("--create_params");
    const auto search_params = parser.get<std::string>("--search_params");
    const auto output_path = parser.get<std::string>("--output_path");
    const auto topk = static_cast<uint64_t>(parser.get<int>("--topk"));
    const auto num_threads = parser.get<int>("--num_threads");
    const auto limit = parser.get<uint64_t>("--limit");
    const auto progress_interval = parser.get<uint64_t>("--progress_interval");

    check(topk > 0, "--topk must be positive");
    check(num_threads > 0, "--num_threads must be positive");

    auto dataset = vsag::eval::EvalDataset::Load(datapath);
    check(dataset->GetVectorType() == vsag::eval::DENSE_VECTORS,
          "export_knng currently only supports dense datasets");
    check(dataset->GetTrainDataType() == vsag::DATATYPE_FLOAT32,
          "export_knng currently only supports float32 train vectors");

    const auto total_base = static_cast<uint64_t>(dataset->GetNumberOfBase());
    check(total_base > topk, "dataset base count must be larger than --topk");
    const auto row_count = limit == 0 ? total_base : std::min(limit, total_base);
    const auto query_k = static_cast<int64_t>(std::min<uint64_t>(total_base, topk + 1));
    const auto* train_vectors = static_cast<const float*>(dataset->GetTrain());
    const auto dim = dataset->GetDim();

    auto index = load_hgraph_index(index_path, create_params);
    check(static_cast<uint64_t>(index->GetNumElements()) == total_base,
          fmt::format("index size {} does not match dataset base count {}",
                      index->GetNumElements(),
                      total_base));

    int output_fd = open_output(output_path, row_count, topk);
    std::atomic<uint64_t> finished{0};
    std::atomic<uint64_t> padded_rows{0};
    std::atomic<bool> failed{false};
    std::mutex error_mutex;
    std::string error_message;

    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t inner_id = 0; inner_id < static_cast<int64_t>(row_count); ++inner_id) {
        if (failed.load(std::memory_order_relaxed)) {
            continue;
        }

        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(train_vectors + inner_id * static_cast<uint64_t>(dim))
            ->Owner(false);

        auto search_result = index->KnnSearch(query, query_k, search_params);
        if (not search_result.has_value()) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (error_message.empty()) {
                error_message = fmt::format(
                    "search failed at row {}: {}", inner_id, search_result.error().message);
            }
            continue;
        }

        std::vector<vsag::InnerIdType> row(topk, 0);
        uint64_t neighbor_count = 0;
        auto result_dataset = search_result.value();
        const auto* ids = result_dataset->GetIds();
        const auto result_size = static_cast<uint64_t>(result_dataset->GetDim());
        for (uint64_t rank = 0; rank < result_size and neighbor_count < topk; ++rank) {
            const auto label_id = ids[rank];
            check(label_id >= 0,
                  fmt::format("search returned negative label id {} at row {}",
                              label_id,
                              inner_id));
            const auto neighbor = static_cast<uint64_t>(label_id);
            if (neighbor >= total_base || neighbor == static_cast<uint64_t>(inner_id)) {
                continue;
            }
            bool duplicate = false;
            for (uint64_t i = 0; i < neighbor_count; ++i) {
                if (row[i] == neighbor) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }
            row[neighbor_count++] = static_cast<vsag::InnerIdType>(neighbor);
        }

        if (neighbor_count == 0) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (error_message.empty()) {
                error_message = fmt::format("row {} produced no usable neighbors", inner_id);
            }
            continue;
        }

        if (neighbor_count < topk) {
            padded_rows.fetch_add(1, std::memory_order_relaxed);
            const auto pad_value = row[neighbor_count - 1];
            while (neighbor_count < topk) {
                row[neighbor_count++] = pad_value;
            }
        }

        try {
            write_row(output_fd, static_cast<uint64_t>(inner_id), row, topk, output_path);
        } catch (const std::exception& err) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (error_message.empty()) {
                error_message = err.what();
            }
            continue;
        }

        const auto done = finished.fetch_add(1, std::memory_order_relaxed) + 1;
        if (progress_interval > 0 && done % progress_interval == 0) {
#pragma omp critical(export_knng_progress)
            {
                std::cout << "[export_knng] exported " << done << "/" << row_count
                          << " rows" << std::endl;
            }
        }
    }

    ::close(output_fd);

    if (failed.load(std::memory_order_relaxed)) {
        throw std::runtime_error(error_message.empty() ? "export_knng failed" : error_message);
    }

    std::cout << "[export_knng] finished rows=" << finished.load(std::memory_order_relaxed)
              << " padded_rows=" << padded_rows.load(std::memory_order_relaxed)
              << " output=" << output_path << std::endl;
    return 0;
}