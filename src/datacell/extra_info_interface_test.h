
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

#pragma once

#include <algorithm>
#include <random>

#include "extra_info_interface.h"

namespace vsag {
class ExtraInfoInterfaceTest {
public:
    ExtraInfoInterfaceTest(ExtraInfoInterfacePtr extra_info) : extra_info_(extra_info) {
    }

    void
    BasicTest(uint64_t base_count);

    void
    TestSerializeAndDeserialize(ExtraInfoInterfacePtr other);

    void
    TestForceInMemory(uint64_t force_count);

public:
    ExtraInfoInterfacePtr extra_info_{nullptr};
};
}  // namespace vsag
