// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <vsag/lite/index.h>

#include <fstream>
#include <iostream>

int
main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: lite_example NEW_SNAPSHOT_PATH\n";
        return 1;
    }
    // The example intentionally refuses to replace an existing readable file.
    if (std::ifstream(argv[1]).good()) {
        std::cerr << "snapshot already exists\n";
        return 1;
    }
    auto index = vsag::lite::Index::Create(3);
    const float vector[]{1, 2, 3};
    if (not index or not(*index)->Add(42, vector, 3)) {
        return 1;
    }
    std::ofstream output(argv[1], std::ios::binary);
    if (not(*index)->Save(output)) {
        return 1;
    }
    output.close();
    if (not output) {
        return 1;
    }
    std::ifstream input(argv[1], std::ios::binary);
    auto loaded = vsag::lite::Index::Load(input);
    if (not loaded) {
        return 1;
    }
    auto result = (*loaded)->Search(vector, 3, 1);
    if (not result or result->size() != 1 or result->front().id != 42) {
        return 1;
    }
    std::cout << "id=" << result->front().id << " squared_l2=" << result->front().distance << '\n';
}
