// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <sstream>

#include "vsag/lite/index.h"

int
main() {
    auto created = vsag::lite::Index::Create(3);
    if (not created) {
        return 1;
    }
    auto& index = **created;
    float first[]{0, 1, 2};
    float second[]{2, 1, 0};
    if (not index.Add(42, first, 3) or not index.Add(7, second, 3) or
        not index.BuildGraph(vsag::lite::VectorStorage::RABITQ8, 2, 8) or
        not index.Update(42, second, 3) or not index.Remove(7) or not index.Add(9, first, 3)) {
        return 1;
    }
    std::stringstream snapshot;
    if (not index.Save(snapshot)) {
        return 1;
    }
    auto loaded = vsag::lite::Index::Load(snapshot);
    if (not loaded) {
        return 1;
    }
    auto result =
        (*loaded)->SearchWithOptions(first, 3, 2, {8}, [](int64_t id) { return id == 9; });
    if (not result or result->size() != 1 or result->front().id != 9) {
        return 1;
    }
    std::cout << "RaBitQ loaded id=" << result->front().id
              << " estimated_l2=" << result->front().distance << '\n';
    return 0;
}
