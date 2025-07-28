
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

#include <cstring>
#include <iostream>
#include <random>
#include <thread>

#include "message_queue.h"
#include "receiver.h"
#include "sender.h"
#include "vsag/factory.h"

vsag::IndexPtr
CreateIndex() {
    auto build_string =
        R"({"dim":960,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":200,"base_io_type":"memory_io"}})";
    auto index = vsag::Factory::CreateIndex("hgraph", build_string);
    return index.value();
}

void
AddIndex(vsag::IndexPtr& index) {
    int64_t num_vectors = 100;
    int64_t dim = 960;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> datas(num_vectors * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        datas[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(datas.data())
        ->Owner(false);
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index HGraph contains: " << index->GetNumElements()
                  << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }
}

void
Search(vsag::IndexPtr& index) {
    int64_t dim = 960;
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    std::vector<float> query_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);

    /******************* KnnSearch For HGraph Index *****************/
    auto hgraph_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;
    auto result = index->KnnSearch(query, topk, hgraph_search_parameters).value();

    /******************* Print Search Result *****************/
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }
}

int
main() {
    MessageQueue send_queue;
    MessageQueue recv_queue;

    auto index1 = CreateIndex();
    AddIndex(index1);
    Search(index1);
    auto index2 = CreateIndex();

    Sender sender(index1.get(), &send_queue, &recv_queue);
    Receiver receiver(index2.get(), &recv_queue, &send_queue);

    std::thread producer([&sender]() {
        sender.Init();

        while (not sender.IsFinished()) {
            sender.Run();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            sender.Pause();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    std::thread consumer([&receiver]() {
        while (not receiver.IsFinished()) {
            receiver.Run();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            receiver.Pause();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    producer.join();
    consumer.join();

    Search(index2);

    return 0;
}