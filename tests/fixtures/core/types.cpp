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

#include "types.h"

#include <cmath>

namespace fixtures {

comparable_float_t::comparable_float_t(float val) {
    this->value = val;
}

bool
comparable_float_t::operator==(const comparable_float_t& d) const {
    double a = this->value;
    double b = d.value;
    if (std::abs(a - b) < epsilon) {
        return true;
    }
    double max_value = std::max(std::abs(a), std::abs(b));
    int power = std::max(0, int(log10(max_value) + 1));
    return std::abs(a - b) <= epsilon * pow(10.0, power);
}

std::ostream&
operator<<(std::ostream& os, const comparable_float_t& obj) {
    os << obj.value;
    return os;
}

IOItem::~IOItem() {
    delete[] data_;
}

}  // namespace fixtures
