// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "datacell/flatten_optimized_build_interface.h"

namespace vsag {

class Pyramid;

// RAII guard that keeps the temporary RaBitQ scalar build codes alive for the
// duration of a Pyramid build. The plain build path and the imported-cache
// build path share the same optimized construction session so both benefit
// from scalar-code distance evaluation.
class PyramidOptimizedBuildSession {
public:
    explicit PyramidOptimizedBuildSession(Pyramid& pyramid);

    PyramidOptimizedBuildSession(const PyramidOptimizedBuildSession&) = delete;
    PyramidOptimizedBuildSession&
    operator=(const PyramidOptimizedBuildSession&) = delete;

    ~PyramidOptimizedBuildSession() noexcept;

    [[nodiscard]] bool
    Active() const;

    void
    Commit();

private:
    Pyramid* pyramid_{nullptr};
    FlattenOptimizedBuildInterfacePtr optimized_build_codes_{nullptr};
};

}  // namespace vsag
