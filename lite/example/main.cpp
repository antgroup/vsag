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
    if (not index) {
        std::cerr << "Create failed: " << index.error().message << '\n';
        return 1;
    }
    const float first[]{1, 2, 3};
    const float updated[]{2, 3, 4};
    const float remaining[]{5, 6, 7};

    auto add_first = (*index)->Add(42, first, 3);
    if (not add_first) {
        std::cerr << "Add(42) failed: " << add_first.error().message << '\n';
        return 1;
    }
    auto add_remaining = (*index)->Add(7, remaining, 3);
    if (not add_remaining) {
        std::cerr << "Add(7) failed: " << add_remaining.error().message << '\n';
        return 1;
    }
    auto before = (*index)->Search(first, 3, 1);
    if (not before) {
        std::cerr << "Search before Update failed: " << before.error().message << '\n';
        return 1;
    }
    if (before->size() != 1 or before->front().id != 42 or before->front().distance != 0) {
        std::cerr << "Search before Update returned an unexpected result\n";
        return 1;
    }
    std::cout << "added id=" << before->front().id << " squared_l2=" << before->front().distance
              << '\n';

    auto update = (*index)->Update(42, updated, 3);
    if (not update) {
        std::cerr << "Update failed: " << update.error().message << '\n';
        return 1;
    }
    auto after = (*index)->Search(updated, 3, 1);
    if (not after) {
        std::cerr << "Search after Update failed: " << after.error().message << '\n';
        return 1;
    }
    if (after->size() != 1 or after->front().id != 42 or after->front().distance != 0) {
        std::cerr << "Search after Update returned an unexpected result\n";
        return 1;
    }
    std::cout << "updated id=" << after->front().id << " squared_l2=" << after->front().distance
              << '\n';

    if (not(*index)->Remove(42) or (*index)->Size() != 1) {
        std::cerr << "Remove failed\n";
        return 1;
    }
    auto after_remove = (*index)->Search(updated, 3, 2);
    if (not after_remove) {
        std::cerr << "Search after Remove failed: " << after_remove.error().message << '\n';
        return 1;
    }
    if (after_remove->size() != 1 or after_remove->front().id != 7) {
        std::cerr << "Search after Remove still returned the removed ID\n";
        return 1;
    }
    std::cout << "removed id=42 remaining=" << (*index)->Size() << '\n';

    std::ofstream output(argv[1], std::ios::binary);
    if (not output) {
        std::cerr << "Could not open snapshot for writing\n";
        return 1;
    }
    auto save = (*index)->Save(output);
    if (not save) {
        std::cerr << "Save failed: " << save.error().message << '\n';
        return 1;
    }
    output.close();
    if (not output) {
        std::cerr << "Could not close snapshot after writing\n";
        return 1;
    }
    std::ifstream input(argv[1], std::ios::binary);
    if (not input) {
        std::cerr << "Could not open snapshot for reading\n";
        return 1;
    }
    auto loaded = vsag::lite::Index::Load(input);
    if (not loaded) {
        std::cerr << "Load failed: " << loaded.error().message << '\n';
        return 1;
    }
    auto restored = (*loaded)->Search(remaining, 3, 1);
    if (not restored) {
        std::cerr << "Search after Load failed: " << restored.error().message << '\n';
        return 1;
    }
    if ((*loaded)->Size() != 1 or restored->size() != 1 or restored->front().id != 7 or
        restored->front().distance != 0) {
        std::cerr << "Search after Load returned an unexpected result\n";
        return 1;
    }
    std::cout << "loaded id=" << restored->front().id
              << " squared_l2=" << restored->front().distance << '\n';
}
