
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

#include <string>

namespace hnswlib {

using DISTFUNC = float (*)(const void*, const void*, const void*);

class SpaceInterface {
public:
    // virtual void search(void *);
    virtual size_t
    get_data_size() = 0;

    virtual DISTFUNC
    get_dist_func() = 0;

    virtual void*
    get_dist_func_param() = 0;

    virtual ~SpaceInterface() {
    }
};
}  // namespace hnswlib
