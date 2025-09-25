
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

#include <functional>

#include "datacell/extra_info_interface.h"
#include "typing.h"
#include "vsag/bitset.h"
#include "vsag/filter.h"

namespace vsag {

class ExtraInfoWrapperFilter : public Filter {
public:
    ExtraInfoWrapperFilter(const FilterPtr filter_impl, const ExtraInfoInterfacePtr& extra_infos)
        : filter_impl_(filter_impl), extra_infos_(extra_infos){};

    [[nodiscard]] bool
    CheckValid(int64_t inner_id) const override;

    [[nodiscard]] float
    ValidRatio() const override {
        return filter_impl_->ValidRatio();
    }

    [[nodiscard]] Distribution
    FilterDistribution() const override {
        return filter_impl_->FilterDistribution();
    }

private:
    const FilterPtr filter_impl_;
    const ExtraInfoInterfacePtr& extra_infos_;
};

}  // namespace vsag
