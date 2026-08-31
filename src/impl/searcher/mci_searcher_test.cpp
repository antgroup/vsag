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

#include "mci_searcher.h"

#include "searcher_test.h"

using namespace vsag;

TEST_CASE("MCISearcher threshold results backfill past non-finite traversal seeds",
          "[ut][MCISearcher][threshold][nonfinite]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = 1;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_IP;

    constexpr const char* param_template = R"({{"type": "{}"}})";
    auto quantizer_param = QuantizerParameter::GetQuantizerParameterByJson(
        JsonType::Parse(fmt::format(param_template, "fp32")));
    auto io_param = IOParameter::GetIOParameterByJson(
        JsonType::Parse(fmt::format(param_template, "memory_io")));
    auto flatten = std::make_shared<
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_IP>, FixedLayout<MemoryIO>>>(
        quantizer_param, io_param, common);
    flatten->SetQuantizer(
        std::make_shared<FP32Quantizer<MetricType::METRIC_TYPE_IP>>(1, allocator.get()));
    flatten->SetIO(std::make_unique<MemoryIO>(allocator.get()));

    std::vector<float> vectors = {std::numeric_limits<float>::max(), 0.0F};
    std::vector<InnerIdType> ids = {0, 1};
    flatten->Train(vectors.data(), ids.size());
    flatten->BatchInsertVector(vectors.data(), ids.size(), ids.data());

    auto cliques = std::make_shared<CliqueDataCell>(allocator.get());
    Vector<InnerIdType> p_maxc(allocator.get());
    Vector<InnerIdType> maxcs(allocator.get());
    Vector<InnerIdType> p_node_to_cid(allocator.get());
    Vector<InnerIdType> node_to_cids(allocator.get());
    p_maxc.insert(p_maxc.end(), {0, 2});
    maxcs.insert(maxcs.end(), {0, 1});
    p_node_to_cid.insert(p_node_to_cid.end(), {0, 1, 2});
    node_to_cids.insert(node_to_cids.end(), {0, 0});
    cliques->Assign(std::move(p_maxc),
                    std::move(maxcs),
                    std::move(p_node_to_cid),
                    std::move(node_to_cids),
                    ids.size());

    InnerSearchParam search_param;
    search_param.ef = 1;
    search_param.topk = 1;
    search_param.distance_threshold = 1.0F;
    Vector<InnerIdType> seeds(allocator.get());
    seeds.push_back(0);

    const bool use_precise_csr = GENERATE(false, true);
    bool used_precise_csr = false;
    MCISearcherParam mci_param;
    mci_param.seed_count = 1;
    mci_param.seed_inner_ids = &seeds;
    mci_param.precise_vectors = use_precise_csr ? vectors.data() : nullptr;
    mci_param.dim = 1;
    mci_param.precise_vector_stride = 1;
    mci_param.metric = MetricType::METRIC_TYPE_IP;
    mci_param.used_precise_float_csr = &used_precise_csr;

    const float query = std::numeric_limits<float>::max();
    auto result =
        MCISearcher(common).Search(cliques, flatten, &query, search_param, mci_param, nullptr);

    REQUIRE(used_precise_csr == use_precise_csr);
    REQUIRE(result->Size() == 1);
    REQUIRE(result->Top().second == 1);
    REQUIRE(result->Top().first == 1.0F);
}
