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

#include "pyramid_fast_build.h"

#include <utility>

#include "pyramid.h"

namespace vsag {

PyramidOptimizedBuildSession::PyramidOptimizedBuildSession(Pyramid& pyramid) : pyramid_(&pyramid) {
    if (pyramid.optimized_build_codes_ != nullptr) {
        return;
    }
    auto optimized_build_codes =
        std::dynamic_pointer_cast<FlattenOptimizedBuildInterface>(pyramid.construction_codes());
    if (optimized_build_codes == nullptr) {
        return;
    }
    FlattenOptimizedBuildContext context{pyramid.thread_pool_, pyramid.build_thread_count_};
    if (not optimized_build_codes->BeginOptimizedBuild(context)) {
        return;
    }
    optimized_build_codes_ = std::move(optimized_build_codes);
    pyramid_->optimized_build_codes_ = optimized_build_codes_;
}

PyramidOptimizedBuildSession::~PyramidOptimizedBuildSession() noexcept {
    if (optimized_build_codes_ != nullptr) {
        optimized_build_codes_->AbortOptimizedBuild();
        pyramid_->optimized_build_codes_.reset();
    }
}

void
PyramidOptimizedBuildSession::Commit() {
    if (optimized_build_codes_ == nullptr) {
        return;
    }
    optimized_build_codes_->FinalizeOptimizedBuild();
    pyramid_->optimized_build_codes_.reset();
    optimized_build_codes_.reset();
}

bool
PyramidOptimizedBuildSession::Active() const {
    return optimized_build_codes_ != nullptr;
}

}  // namespace vsag
