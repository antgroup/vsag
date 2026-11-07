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

// End-to-end profiling, determinism, bounded sampling, HTML output and the CLI.

#include <omp.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <regex>
#include <sstream>

#include "dataset_profiler.h"
#include "html_report.h"
#include "profile_cli.h"
#include "profile_dataset_test_utils.h"
#include "sha256.h"

using namespace vsag::dataset_profile;
using namespace vsag::dataset_profile::testing;
using Catch::Matchers::ContainsSubstring;

namespace {

ProfileOptions
small_options(const std::string& path) {
    ProfileOptions options;
    options.data_path = path;
    options.sample_size = 1000;
    options.knn_queries = 200;
    options.pair_count = 5000;
    options.hubness_sample = 500;
    options.quantization_base = 600;
    options.quantization_queries = 50;
    options.projection_points = 300;
    return options;
}

const JsonType&
at(const JsonType& report, const std::string& path) {
    const JsonType* node = find_path(report, path);
    if (node == nullptr) {
        FAIL("missing report key " << path);
    }
    return *node;
}

JsonType
without_run(JsonType report) {
    report.erase("run");
    return report;
}

std::string
read_text(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int
cli(const std::vector<std::string>& arguments,
    std::string* out = nullptr,
    std::string* err = nullptr) {
    std::vector<std::string> storage = {"profile_dataset"};
    storage.insert(storage.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& argument : storage) {
        argv.push_back(argument.data());
    }
    std::ostringstream out_stream;
    std::ostringstream err_stream;
    const int code = run_cli(static_cast<int>(argv.size()), argv.data(), out_stream, err_stream);
    if (out != nullptr) {
        *out = out_stream.str();
    }
    if (err != nullptr) {
        *err = err_stream.str();
    }
    return code;
}

uint64_t
count_occurrences(const std::string& text, const std::string& needle) {
    uint64_t count = 0;
    for (auto position = text.find(needle); position != std::string::npos;
         position = text.find(needle, position + 1)) {
        ++count;
    }
    return count;
}

}  // namespace

TEST_CASE("Dense profile contains every documented metric with exact defect counts",
          "[profile][report]") {
    TempDir dir;
    const auto path = dir.File("dense.hdf5");
    const DenseFixture fixture = make_dense_fixture(path);
    const JsonType report = profile_dataset(small_options(path), nullptr);
    CHECK(report["schema"] == REPORT_SCHEMA);
    CHECK(report["schema_version"] == REPORT_SCHEMA_VERSION);
    CHECK(report["config"]["metric"] == "l2");
    for (const char* key : {"splits.train.basic.rows",
                            "splits.train.basic.dim",
                            "splits.train.invalid_values.rows_with_invalid",
                            "splits.train.norms.summary.p50",
                            "splits.train.norms.histogram.counts",
                            "splits.train.norms.unit_norm_ratio",
                            "splits.train.norms.outliers.high",
                            "splits.train.dimensions.per_dimension.mean",
                            "splits.train.dimensions.per_dimension.std",
                            "splits.train.dimensions.per_dimension.zero_ratio",
                            "splits.train.dimensions.per_dimension.excess_kurtosis",
                            "splits.train.dimensions.per_dimension.outlier_ratio_6_sigma",
                            "splits.train.dimensions.summary.constant_dimensions",
                            "splits.train.values.histogram.counts",
                            "splits.train.values.outliers_6_sigma.ratio",
                            "splits.train.sparsity.zero_ratio",
                            "splits.train.sparsity.nonzeros_per_row.summary.p50",
                            "splits.train.duplicates.duplicate_rows",
                            "splits.train.pairwise_distances.metric_distance.summary.p50",
                            "splits.train.pairwise_distances.concentration_ratio",
                            "splits.train.nearest_neighbors.first_neighbor_distance.summary.p50",
                            "splits.train.nearest_neighbors.near_duplicate_ratio",
                            "splits.train.intrinsic_dimensionality.lid_mle.summary.p50",
                            "splits.train.intrinsic_dimensionality.two_nn",
                            "splits.train.relative_contrast.summary.p50",
                            "splits.train.hubness.skewness",
                            "splits.train.anisotropy.average_cosine_similarity.value",
                            "splits.train.anisotropy.mean_vector_norm_ratio.value",
                            "splits.train.pca.components_for_variance.90",
                            "splits.train.pca.explained_variance_ratio",
                            "splits.train.clustering.sizes",
                            "splits.train.clustering.silhouette",
                            "splits.train.projection.points",
                            "splits.train.quantization.schemes.sq8.recall_at_k",
                            "splits.test.norms.summary.p50",
                            "cross_split.norm_ks_statistic.value",
                            "cross_split.out_of_distribution_ratio.value",
                            "ground_truth.distance_recall_at_k",
                            "labels.query_selectivity.summary.p50",
                            "determinism.metrics_sha256"}) {
        INFO(key);
        CHECK(find_path(report, key) != nullptr);
    }
    const JsonType& train = report["splits"]["train"];
    CHECK(train["invalid_values"]["nan"] == 1);
    CHECK(train["invalid_values"]["positive_infinity"] == 1);
    CHECK(train["invalid_values"]["negative_infinity"] == 1);
    CHECK(train["invalid_values"]["rows_with_invalid"] == 3);
    CHECK(train["duplicates"]["duplicate_rows"] == 10);
    CHECK(train["duplicates"]["groups"] == 10);
    CHECK(train["duplicates"]["scope"] == "full");
    CHECK(train["norms"]["zero_norm_rows"] == 1);
    CHECK(train["norms"]["outliers"]["high"].get<uint64_t>() >= 3);
    CHECK(train["norms"]["summary"]["count"] == fixture.rows - 3);
    CHECK(at(report, "ground_truth.status") == "checked");
    CHECK(at(report, "ground_truth.distance_recall_at_k") == 1.0);
    CHECK(at(report, "ground_truth.best_matching_distance_convention") ==
          "euclidean (sqrt of l2sqr)");
    CHECK(at(report, "labels.query_selectivity.queries_without_matching_base_rows") == 16);
    std::set<std::string> signals;
    for (const auto& signal : report["signals"]) {
        signals.insert(signal["id"].get<std::string>());
        for (const auto& key : signal["evidence"]) {
            CHECK(find_path(report, key.get<std::string>()) != nullptr);
        }
    }
    CHECK(signals.count("train.invalid_values") == 1);
    CHECK(signals.count("train.zero_vectors") == 1);
    CHECK(signals.count("train.exact_duplicates") == 1);
    CHECK(report["signals"][0]["severity"] == "critical");
}

TEST_CASE("Full-split and sampled metrics are labeled and bounded by the configuration",
          "[profile][report]") {
    TempDir dir;
    const auto path = dir.File("dense.hdf5");
    make_dense_fixture(path);
    ProfileOptions options = small_options(path);
    options.sample_size = 300;
    options.knn_queries = 40;
    options.pair_count = 777;
    options.hubness_sample = 120;
    options.projection_points = 90;
    const JsonType report = profile_dataset(options, nullptr);
    const JsonType& train = report["splits"]["train"];
    for (const char* section :
         {"invalid_values", "norms", "dimensions", "sparsity", "values", "duplicates"}) {
        INFO(section);
        CHECK(train[section]["scope"] == "full");
    }
    for (const char* section : {"pairwise_distances",
                                "nearest_neighbors",
                                "intrinsic_dimensionality",
                                "relative_contrast",
                                "hubness",
                                "pca",
                                "clustering",
                                "projection",
                                "quantization"}) {
        INFO(section);
        CHECK(train[section]["scope"] == "sample");
    }
    CHECK(train["pairwise_distances"]["pairs"] == 777);
    CHECK(train["pairwise_distances"]["sample"]["population"] == 3000);
    CHECK(train["pairwise_distances"]["sample"]["sampled_rows"] == 300);
    CHECK(train["nearest_neighbors"]["sample"]["used_rows"] == 40);
    CHECK(train["nearest_neighbors"]["first_neighbor_distance"]["summary"]["count"] == 40);
    CHECK(train["hubness"]["sample"]["used_rows"] == 120);
    CHECK(train["projection"]["points"].size() == 90);
    CHECK(train["pca"]["sample"]["used_rows"].get<uint64_t>() <= 300);
    CHECK(train["anisotropy"]["scope"] == "mixed");
    CHECK(train["anisotropy"]["mean_vector_norm_ratio"]["scope"] == "full");
}

TEST_CASE("Identical inputs give identical metrics for any thread count, chunk size and run",
          "[profile][determinism]") {
    TempDir dir;
    const auto path = dir.File("dense.hdf5");
    make_dense_fixture(path, 20000);
    ProfileOptions options = small_options(path);
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    options.chunk_mb = 1;
    const JsonType single = profile_dataset(options, nullptr);
    omp_set_num_threads(std::max(threads, 4));
    options.chunk_mb = 64;
    const JsonType parallel = profile_dataset(options, nullptr);
    const JsonType again = profile_dataset(options, nullptr);
    omp_set_num_threads(threads);
    CHECK(single["determinism"]["metrics_sha256"] == parallel["determinism"]["metrics_sha256"]);
    CHECK(parallel["determinism"]["metrics_sha256"] == again["determinism"]["metrics_sha256"]);
    CHECK(without_run(single) == without_run(parallel));
    CHECK(metrics_digest(parallel) == parallel["determinism"]["metrics_sha256"].get<std::string>());

    options.seed = 7;
    const JsonType reseeded = profile_dataset(options, nullptr);
    CHECK(reseeded["determinism"]["metrics_sha256"] != parallel["determinism"]["metrics_sha256"]);
    // Full-split metrics do not depend on the seed; sampled ones do.
    CHECK(reseeded["splits"]["train"]["norms"] == parallel["splits"]["train"]["norms"]);
    CHECK(reseeded["splits"]["train"]["dimensions"] == parallel["splits"]["train"]["dimensions"]);
    CHECK(reseeded["splits"]["train"]["duplicates"] == parallel["splits"]["train"]["duplicates"]);
    CHECK(reseeded["splits"]["train"]["pairwise_distances"] !=
          parallel["splits"]["train"]["pairwise_distances"]);
}

TEST_CASE("Sample-based work stays bounded on a larger split", "[profile][determinism]") {
    TempDir dir;
    const auto path = dir.File("large.hdf5");
    const uint64_t rows = 100000;
    const uint64_t dim = 4;
    {
        DeterministicRng rng(1);
        std::vector<float> train(rows * dim);
        for (auto& value : train) {
            value = static_cast<float>(rng.UniformDouble());
        }
        H5::H5File file(path, H5F_ACC_TRUNC);
        write_string_attribute(file, "distance", "euclidean");
        write_matrix(file, "train", train, rows, dim, H5::PredType::NATIVE_FLOAT);
    }
    ProfileOptions options;
    options.data_path = path;
    options.sample_size = 500;
    options.knn_queries = 50;
    options.pair_count = 2000;
    const JsonType report = profile_dataset(options, nullptr);
    const JsonType& train = report["splits"]["train"];
    CHECK(train["norms"]["summary"]["count"] == rows);
    CHECK(train["nearest_neighbors"]["searched_rows"] == rows);
    CHECK(train["nearest_neighbors"]["sample"]["used_rows"] == 50);
    CHECK(train["pairwise_distances"]["sample"]["sampled_rows"] == 500);
    // An all-pairs matrix for 100k rows would need 80 GB; the profile stays far below 2 GB.
    CHECK(report["run"]["peak_rss_bytes"].get<uint64_t>() < 2ULL * 1024 * 1024 * 1024);
    CHECK(report["input"]["eval_compatibility"]["compatible"] == false);
}

TEST_CASE("Sparse profile reports term statistics, pruning and ground truth", "[profile][report]") {
    TempDir dir;
    for (const bool with_offsets : {true, false}) {
        const auto path = dir.File(with_offsets ? "sparse.hdf5" : "sparse_no_offsets.hdf5");
        const SparseFixture fixture = make_sparse_fixture(path, with_offsets);
        const JsonType report = profile_dataset(small_options(path), nullptr);
        CHECK(report["input"]["vector_type"] == "sparse");
        CHECK(report["config"]["metric"] == "ip");
        const JsonType& train = report["splits"]["train"];
        CHECK(train["rows"] == fixture.train.size());
        CHECK(train["sparsity"]["empty_rows"] == 1);
        CHECK(train["duplicates"]["duplicate_rows"] == 1);
        CHECK(train["basic"]["offsets_present"] == with_offsets);
        uint32_t max_id = 0;
        for (const auto& record : fixture.train) {
            for (const uint32_t id : record.ids) {
                max_id = std::max(max_id, id);
            }
        }
        CHECK(train["terms"]["max_term_id"] == max_id);
        CHECK(train["terms"]["min_term_id_limit"] == max_id + 1);
        const double kept_10 =
            train["terms"]["doc_prune_retained_term_fraction"]["by_ratio"]["10"].get<double>();
        const double kept_50 =
            train["terms"]["doc_prune_retained_term_fraction"]["by_ratio"]["50"].get<double>();
        CHECK(kept_10 > kept_50);
        CHECK(kept_10 <= 1.0);
        CHECK(find_path(report,
                        "splits.train.nearest_neighbors.first_neighbor_distance.summary.p50") !=
              nullptr);
        CHECK(find_path(report, "cross_split.query_term_coverage.value") != nullptr);
        CHECK(find_path(report, "cross_split.postings_per_query.summary.p50") != nullptr);
        CHECK(at(report, "ground_truth.distance_recall_at_k") == 1.0);
        CHECK(at(report, "ground_truth.best_matching_distance_convention") == "1 - similarity");
    }
    ProfileOptions options = small_options(dir.File("sparse.hdf5"));
    options.metric = "l2";
    REQUIRE_THROWS_WITH(profile_dataset(options, nullptr), ContainsSubstring("ip and cosine"));
}

TEST_CASE("Sparse record counts must match the declared offsets", "[profile][report]") {
    TempDir dir;
    const std::vector<SparseRecord> records = {
        SparseRecord{{1, 4}, {0.5F, 0.25F}}, SparseRecord{{2}, {1.0F}}, SparseRecord{{3}, {2.0F}}};
    std::vector<uint64_t> offsets;
    const auto bytes = encode_sparse(records, &offsets);
    auto write = [&](const std::string& path, const std::vector<uint64_t>& declared) {
        H5::H5File file(path, H5F_ACC_TRUNC);
        write_string_attribute(file, "type", "sparse");
        write_string_attribute(file, "distance", "ip");
        write_vector(file, "train", bytes, H5::PredType::NATIVE_INT8);
        write_vector(file, "train_offsets", declared, H5::PredType::NATIVE_UINT64);
    };
    // The offsets declare one record while the stream holds three: per-row arrays are sized
    // from the offsets, so the extra records must be rejected before they are written.
    const auto extra = dir.File("extra_records.hdf5");
    write(extra, {offsets[0], offsets[1]});
    REQUIRE_THROWS_WITH(profile_dataset(small_options(extra), nullptr),
                        ContainsSubstring("more records than its offsets declare (1)"));
    std::vector<uint64_t> too_many = offsets;
    too_many.push_back(offsets.back());
    too_many.push_back(offsets.back());
    const auto missing = dir.File("missing_records.hdf5");
    write(missing, too_many);
    REQUIRE_THROWS_WITH(profile_dataset(small_options(missing), nullptr),
                        ContainsSubstring("has 3 records but its offsets declare 5"));
}

TEST_CASE("Term ids above 2^26 keep their document frequency across splits", "[profile][report]") {
    TempDir dir;
    const uint32_t large = 1U << 26U;
    const std::vector<SparseRecord> train(40, SparseRecord{{5, large}, {1.0F, 0.5F}});
    const std::vector<SparseRecord> test(8, SparseRecord{{large}, {1.0F}});
    const auto path = dir.File("large_terms.hdf5");
    {
        H5::H5File file(path, H5F_ACC_TRUNC);
        write_string_attribute(file, "type", "sparse");
        write_string_attribute(file, "distance", "ip");
        write_vector(file, "train", encode_sparse(train, nullptr), H5::PredType::NATIVE_INT8);
        write_vector(file, "test", encode_sparse(test, nullptr), H5::PredType::NATIVE_INT8);
    }
    const JsonType report = profile_dataset(small_options(path), nullptr);
    CHECK(report["splits"]["train"]["terms"]["max_term_id"] == large);
    CHECK(at(report, "cross_split.query_term_coverage.value") == 1.0);
    CHECK(at(report, "cross_split.postings_per_query.summary.p50") == 40.0);
}

TEST_CASE("Finite rows whose norm exceeds the float range keep their norm", "[profile][report]") {
    TempDir dir;
    const float big = std::numeric_limits<float>::max();
    const uint64_t rows = 64;
    const uint64_t dim = 4;
    std::vector<float> data(rows * dim);
    DeterministicRng rng(3);
    for (auto& value : data) {
        value = static_cast<float>(rng.UniformDouble());
    }
    data[5 * dim] = big;
    data[5 * dim + 1] = big;
    const auto dense = dir.File("huge_norm_dense.hdf5");
    {
        H5::H5File file(dense, H5F_ACC_TRUNC);
        write_string_attribute(file, "distance", "euclidean");
        write_matrix(file, "train", data, rows, dim, H5::PredType::NATIVE_FLOAT);
    }
    std::vector<SparseRecord> records(30, SparseRecord{{1}, {0.5F}});
    records.push_back(SparseRecord{{2, 3}, {big, big}});
    const auto sparse = dir.File("huge_norm_sparse.hdf5");
    {
        H5::H5File file(sparse, H5F_ACC_TRUNC);
        write_string_attribute(file, "type", "sparse");
        write_string_attribute(file, "distance", "ip");
        write_vector(file, "train", encode_sparse(records, nullptr), H5::PredType::NATIVE_INT8);
    }
    for (const auto& [path, count] :
         std::vector<std::pair<std::string, uint64_t>>{{dense, rows}, {sparse, records.size()}}) {
        INFO(path);
        const JsonType report = profile_dataset(small_options(path), nullptr);
        const JsonType& summary = report["splits"]["train"]["norms"]["summary"];
        CHECK(summary["count"] == count);
        CHECK(summary["max"].get<double>() > static_cast<double>(big));
    }
}

TEST_CASE("Ground truth whose ids do not match its distances is reported", "[profile][report]") {
    TempDir dir;
    const auto path = dir.File("wrong_ids.hdf5");
    make_dense_fixture(path);
    // Move every stored id to a row of another cluster but keep the stored distances, which
    // are still the true neighbor distances.
    {
        H5::H5File file(path, H5F_ACC_RDWR);
        H5::DataSet dataset = file.openDataSet("neighbors");
        std::vector<int64_t> ids(100ULL * 10ULL);
        dataset.read(ids.data(), H5::PredType::NATIVE_INT64);
        for (auto& id : ids) {
            id = (id + 1501) % 3000;
        }
        dataset.write(ids.data(), H5::PredType::NATIVE_INT64);
    }
    const JsonType report = profile_dataset(small_options(path), nullptr);
    CHECK(at(report, "ground_truth.id_recall_at_k").get<double>() < 0.05);
    CHECK(at(report, "ground_truth.distance_recall_at_k").get<double>() < 0.05);
    CHECK(at(report, "ground_truth.best_convention_mean_relative_error").get<double>() > 0.1);
    bool mismatch = false;
    for (const auto& signal : report["signals"]) {
        mismatch = mismatch || signal["id"] == "ground_truth.mismatch";
    }
    CHECK(mismatch);
}

TEST_CASE("Tiny and degenerate splits produce valid reports", "[profile][report]") {
    TempDir dir;
    for (const uint64_t rows : {1ULL, 2ULL, 5ULL}) {
        INFO("rows = " << rows);
        const auto path = dir.File("tiny_" + std::to_string(rows) + ".hdf5");
        {
            std::vector<float> train(rows * 3);
            for (uint64_t i = 0; i < train.size(); ++i) {
                train[i] = static_cast<float>(i % 7) - 3.0F;
            }
            H5::H5File file(path, H5F_ACC_TRUNC);
            write_string_attribute(file, "distance", "angular");
            write_matrix(file, "train", train, rows, 3, H5::PredType::NATIVE_FLOAT);
        }
        const JsonType report = profile_dataset(small_options(path), nullptr);
        CHECK(report["splits"]["train"]["rows"] == rows);
        CHECK(report["splits"]["train"]["norms"]["summary"]["count"] == rows);
        CHECK_NOTHROW(render_html_report(report));
        // Every number in the report is finite: NaN and infinity are stored as null.
        uint64_t non_finite = 0;
        std::function<void(const JsonType&)> visit = [&](const JsonType& node) {
            if (node.is_number_float() && !std::isfinite(node.get<double>())) {
                ++non_finite;
            }
            if (node.is_structured()) {
                for (const auto& child : node) {
                    visit(child);
                }
            }
        };
        visit(report);
        CHECK(non_finite == 0);
    }
    const auto constant = dir.File("constant.hdf5");
    {
        H5::H5File file(constant, H5F_ACC_TRUNC);
        write_string_attribute(file, "distance", "euclidean");
        write_matrix(file,
                     "train",
                     std::vector<float>(400ULL * 4ULL, 2.5F),
                     400,
                     4,
                     H5::PredType::NATIVE_FLOAT);
    }
    const JsonType report = profile_dataset(small_options(constant), nullptr);
    CHECK(report["splits"]["train"]["duplicates"]["duplicate_rows"] == 399);
    CHECK(report["splits"]["train"]["dimensions"]["summary"]["constant_dimensions"] == 4);
    CHECK(report["splits"]["train"]["nearest_neighbors"]["exact_duplicate_neighbor_ratio"] == 1.0);
    CHECK_NOTHROW(render_html_report(report));
}

TEST_CASE("Cosine profiles exclude zero vectors and agree with normalized geometry",
          "[profile][report]") {
    TempDir dir;
    const auto path = dir.File("angular.hdf5");
    make_dense_fixture(path, 3000, "angular");
    const JsonType report = profile_dataset(small_options(path), nullptr);
    CHECK(report["config"]["metric"] == "cosine");
    CHECK(report["splits"]["train"]["pca"]["input"] == "l2-normalized rows");
    // The planted zero row and the three invalid rows are not sampled for distances.
    CHECK(
        report["splits"]["train"]["pairwise_distances"]["sample"]["valid_sampled_rows"]
            .get<uint64_t>() <=
        report["splits"]["train"]["pairwise_distances"]["sample"]["sampled_rows"].get<uint64_t>());
    bool critical_zero = false;
    for (const auto& signal : report["signals"]) {
        critical_zero = critical_zero ||
                        (signal["id"] == "train.zero_vectors" && signal["severity"] == "critical");
    }
    CHECK(critical_zero);
}

TEST_CASE("HTML report is self-contained, labeled and escaped", "[profile][html]") {
    TempDir dir;
    const auto path = dir.File("dense.hdf5");
    make_dense_fixture(path);
    JsonType report = profile_dataset(small_options(path), nullptr);
    report["input"]["file_name"] = "<script>alert('x')</script>";
    const std::string html = render_html_report(report);
    CHECK_THAT(html, ContainsSubstring("Content-Security-Policy"));
    CHECK_THAT(html, ContainsSubstring("default-src 'none'"));
    CHECK_FALSE(std::regex_search(html, std::regex(R"((src|href)\s*=\s*["']?\s*(https?:)?//)")));
    CHECK(html.find("<link") == std::string::npos);
    CHECK(html.find("@import") == std::string::npos);
    CHECK(html.find("url(") == std::string::npos);
    CHECK(count_occurrences(html, "<script") == 2);
    CHECK(count_occurrences(html, "</script>") == 2);
    CHECK_THAT(html, ContainsSubstring("&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;"));
    // Only the report's own script may run: the policy pins its hash and allows no inline
    // handlers.
    const std::string open_tag = "<script>";
    const auto script_begin = html.rfind(open_tag) + open_tag.size();
    const std::string script =
        html.substr(script_begin, html.find("</script>", script_begin) - script_begin);
    CHECK_THAT(html, ContainsSubstring("script-src 'sha256-" + sha256_base64(script) + "'"));
    const std::string policy = html_content_security_policy();
    CHECK_THAT(policy, ContainsSubstring("style-src 'unsafe-inline'"));
    CHECK(count_occurrences(policy, "unsafe-inline") == 1);
    // Split names of a saved profile are escaped wherever they appear.
    JsonType hostile = report;
    hostile["splits"]["train\"><img src=x onerror=alert(1)>"] = report["splits"]["train"];
    const std::string escaped = render_html_report(hostile);
    CHECK(escaped.find("<img") == std::string::npos);
    CHECK_THAT(escaped, ContainsSubstring("&quot;&gt;&lt;img src=x onerror=alert(1)&gt;"));
    // Every chart card states its metric key, definition and scope (full split or sample).
    const std::string marker = "<figure class=\"card\"";
    uint64_t cards = 0;
    for (auto position = html.find(marker); position != std::string::npos;
         position = html.find(marker, position + 1)) {
        const auto end = html.find("</figure>", position);
        const std::string card = html.substr(position, end - position);
        CHECK_THAT(card, ContainsSubstring("class=\"scope "));
        CHECK_THAT(card, ContainsSubstring("class=\"definition\""));
        CHECK_THAT(card, ContainsSubstring("class=\"key\""));
        const bool full = card.find("scope full") != std::string::npos;
        const bool sample = card.find("scope sample") != std::string::npos;
        const bool mixed = card.find("scope mixed") != std::string::npos;
        CHECK((full || sample || mixed));
        if (sample) {
            CHECK_THAT(card, ContainsSubstring("n = "));
        }
        if (full) {
            CHECK_THAT(card, ContainsSubstring("N = "));
        }
        ++cards;
    }
    CHECK(cards >= 20);
}

TEST_CASE("CLI writes reports, re-renders saved profiles and rejects bad input", "[profile][cli]") {
    TempDir dir;
    const auto data = dir.File("dense.hdf5");
    make_dense_fixture(data);
    const auto json = dir.File("out.json");
    const auto html = dir.File("out.html");
    std::string out;
    std::string err;
    REQUIRE(
        cli({"-d", data, "-o", json, "--html", html, "--sample-size", "400", "--knn-queries", "50"},
            &out,
            &err) == EXIT_OK);
    CHECK_THAT(out, ContainsSubstring("metrics SHA-256"));
    CHECK_THAT(out, ContainsSubstring("AI analysis: disabled"));
    CHECK_THAT(err, ContainsSubstring("pass 1/2"));
    const JsonType report = JsonType::parse(read_text(json));
    CHECK(report["config"]["sample_size"] == 400);
    CHECK_FALSE(report.contains("ai_analysis"));
    CHECK_THAT(read_text(html), ContainsSubstring("<!doctype html>"));

    const auto html_again = dir.File("again.html");
    REQUIRE(cli({"--from-profile", json, "--html", html_again, "-q"}) == EXIT_OK);
    CHECK_THAT(read_text(html_again),
               ContainsSubstring(report["determinism"]["metrics_sha256"].get<std::string>()));
    // Saved profiles may only name the train and test splits.
    JsonType hostile = report;
    hostile["splits"]["x\"><img src=x onerror=alert(1)>"] = report["splits"]["train"];
    const auto hostile_json = dir.File("hostile.json");
    {
        std::ofstream file(hostile_json);
        file << hostile.dump();
    }
    CHECK(cli({"--from-profile", hostile_json, "--html", dir.File("hostile.html"), "-q"},
              &out,
              &err) == EXIT_FAILURE_INPUT);
    CHECK_THAT(err, ContainsSubstring("unsupported split name"));

    const auto no_html_json = dir.File("no_html.json");
    REQUIRE(cli({"-d",
                 data,
                 "-o",
                 no_html_json,
                 "--no-html",
                 "-q",
                 "--sample-size",
                 "300",
                 "--disable",
                 "pca,clustering,quantization,hubness"}) == EXIT_OK);
    const JsonType reduced = JsonType::parse(read_text(no_html_json));
    CHECK_FALSE(reduced["splits"]["train"].contains("pca"));
    CHECK_FALSE(reduced["splits"]["train"].contains("quantization"));

    CHECK(cli({}, &out, &err) == EXIT_FAILURE_INPUT);
    CHECK_THAT(err, ContainsSubstring("--datapath or --from-profile"));
    CHECK(cli({"-d", dir.File("missing.hdf5"), "-q"}, &out, &err) == EXIT_FAILURE_INPUT);
    CHECK_THAT(err, ContainsSubstring("does not exist"));
    CHECK(cli({"-d", data, "--ai-provider", "magic"}, &out, &err) == EXIT_FAILURE_INPUT);
    CHECK_THAT(err, ContainsSubstring("unsupported AI provider"));
    CHECK(cli({"-d", data, "--seed", "abc"}, &out, &err) == EXIT_FAILURE_INPUT);
    CHECK(cli({"--from-profile", data}, &out, &err) == EXIT_FAILURE_INPUT);
    CHECK_THAT(err, ContainsSubstring("not valid JSON"));
}
