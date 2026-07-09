
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

#include "utils/atomic_visibility_bitmap.h"

#include <atomic>
#include <thread>
#include <vector>

#include "unittest.h"

using namespace vsag;

TEST_CASE("AtomicVisibilityBitmap Basic", "[ut][AtomicVisibilityBitmap]") {
    SECTION("mark then ready") {
        AtomicVisibilityBitmap bitmap;
        bitmap.Resize(128);
        for (uint64_t i = 0; i < 128; ++i) {
            REQUIRE_FALSE(bitmap.IsReady(i));
        }
        bitmap.Mark(0);
        bitmap.Mark(63);
        bitmap.Mark(64);
        bitmap.Mark(127);
        REQUIRE(bitmap.IsReady(0));
        REQUIRE(bitmap.IsReady(63));
        REQUIRE(bitmap.IsReady(64));
        REQUIRE(bitmap.IsReady(127));
        // Marking one bit must not set its neighbours.
        REQUIRE_FALSE(bitmap.IsReady(1));
        REQUIRE_FALSE(bitmap.IsReady(62));
        REQUIRE_FALSE(bitmap.IsReady(65));
        REQUIRE_FALSE(bitmap.IsReady(126));
    }

    SECTION("mark is idempotent") {
        AtomicVisibilityBitmap bitmap;
        bitmap.Resize(64);
        bitmap.Mark(10);
        bitmap.Mark(10);
        REQUIRE(bitmap.IsReady(10));
    }

    SECTION("mark range") {
        AtomicVisibilityBitmap bitmap;
        bitmap.Resize(200);
        bitmap.MarkRange(130);
        for (uint64_t i = 0; i < 130; ++i) {
            REQUIRE(bitmap.IsReady(i));
        }
        for (uint64_t i = 130; i < 200; ++i) {
            REQUIRE_FALSE(bitmap.IsReady(i));
        }
    }

    SECTION("mark range of zero is a no-op") {
        AtomicVisibilityBitmap bitmap;
        bitmap.Resize(64);
        bitmap.MarkRange(0);
        for (uint64_t i = 0; i < 64; ++i) {
            REQUIRE_FALSE(bitmap.IsReady(i));
        }
    }
}

TEST_CASE("AtomicVisibilityBitmap Resize", "[ut][AtomicVisibilityBitmap]") {
    SECTION("grow preserves existing bits and clears new ones") {
        AtomicVisibilityBitmap bitmap;
        bitmap.Resize(64);
        bitmap.Mark(3);
        bitmap.Mark(50);

        // Grow across a word boundary; previously set bits must survive.
        bitmap.Resize(256);
        REQUIRE(bitmap.IsReady(3));
        REQUIRE(bitmap.IsReady(50));
        for (uint64_t i = 64; i < 256; ++i) {
            REQUIRE_FALSE(bitmap.IsReady(i));
        }

        // The grown region is usable.
        bitmap.Mark(200);
        REQUIRE(bitmap.IsReady(200));
    }

    SECTION("shrink is a no-op") {
        AtomicVisibilityBitmap bitmap;
        bitmap.Resize(256);
        bitmap.Mark(200);
        // Requesting a smaller capacity must not drop storage or bits.
        bitmap.Resize(64);
        REQUIRE(bitmap.IsReady(200));
    }

    SECTION("non-multiple-of-word capacity is addressable") {
        AtomicVisibilityBitmap bitmap;
        bitmap.Resize(65);
        bitmap.Mark(64);
        REQUIRE(bitmap.IsReady(64));
    }
}

TEST_CASE("AtomicVisibilityBitmap Concurrent Mark And Read",
          "[ut][AtomicVisibilityBitmap][concurrent]") {
    // Mirrors the HGraph usage: many producers publish distinct slots while a
    // reader scans. The release/acquire pair must never report a slot ready
    // before its Mark(), and every marked slot must eventually be observed.
    constexpr uint64_t slot_count = 4096;
    AtomicVisibilityBitmap bitmap;
    bitmap.Resize(slot_count);

    std::atomic<bool> start{false};
    std::atomic<uint64_t> missing_ready_count{0};
    std::atomic<bool> reader_timed_out{false};

    constexpr uint32_t marker_threads = 4;
    std::vector<std::thread> markers;
    markers.reserve(marker_threads);
    for (uint32_t t = 0; t < marker_threads; ++t) {
        markers.emplace_back([&, t]() {
            while (not start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (uint64_t i = t; i < slot_count; i += marker_threads) {
                bitmap.Mark(i);
            }
        });
    }

    // Reader keeps scanning until every slot is ready; a slot reported ready is
    // checked to stay ready (bits are never cleared).
    std::thread reader([&]() {
        while (not start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        uint64_t ready = 0;
        constexpr uint64_t max_scan_attempts = 100000;
        uint64_t scan_attempts = 0;
        while (ready < slot_count && scan_attempts < max_scan_attempts) {
            ready = 0;
            for (uint64_t i = 0; i < slot_count; ++i) {
                if (bitmap.IsReady(i)) {
                    ++ready;
                }
            }
            ++scan_attempts;
            std::this_thread::yield();
        }
        reader_timed_out.store(ready < slot_count, std::memory_order_relaxed);
    });

    start.store(true, std::memory_order_release);
    for (auto& th : markers) {
        th.join();
    }
    reader.join();
    REQUIRE_FALSE(reader_timed_out.load(std::memory_order_relaxed));

    // After all markers joined, every slot must be ready.
    for (uint64_t i = 0; i < slot_count; ++i) {
        if (not bitmap.IsReady(i)) {
            missing_ready_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
    REQUIRE(missing_ready_count.load() == 0);
}
