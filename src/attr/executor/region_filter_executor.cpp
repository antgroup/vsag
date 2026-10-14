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

#include "region_filter_executor.h"

#include <limits>
#include <memory>
#include <type_traits>

#include "vsag_exception.h"

namespace vsag {
namespace {

template <typename T>
std::vector<const MultiBitsetManager*>
get_managers(const AttrInvertedInterfacePtr& attr_index,
             const std::string& field_name,
             const std::vector<int64_t>& values) {
    AttributeValue<T> attribute;
    attribute.name_ = field_name;
    auto& converted = attribute.GetValue();
    converted.reserve(values.size());
    for (const auto value : values) {
        if constexpr (std::is_unsigned_v<T>) {
            if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<T>::max()) {
                throw VsagException(ErrorType::INVALID_ARGUMENT,
                                    "region_filter value does not fit field '" + field_name + "'");
            }
        } else if (value < static_cast<int64_t>(std::numeric_limits<T>::min()) ||
                   value > static_cast<int64_t>(std::numeric_limits<T>::max())) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "region_filter value does not fit field '" + field_name + "'");
        }
        converted.emplace_back(static_cast<T>(value));
    }
    return attr_index->GetBitsetsByAttr(attribute);
}

const std::vector<int64_t>&
get_values(const ExprPtr& expression, const std::string& argument_name) {
    auto values = std::dynamic_pointer_cast<const IntListConstant<int64_t>>(expression);
    if (values == nullptr) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            "region_filter " + argument_name + " must contain signed 64-bit integer values");
    }
    return values->values;
}

const std::string&
get_field_name(const ExprPtr& expression, const std::string& argument_name) {
    auto field = std::dynamic_pointer_cast<const FieldExpression>(expression);
    if (field == nullptr || field->fieldName.empty()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "region_filter " + argument_name + " must be a field name");
    }
    return field->fieldName;
}

}  // namespace

RegionFilterExecutor::RegionFilterExecutor(Allocator* allocator,
                                           const ExprPtr& expression,
                                           const AttrInvertedInterfacePtr& attr_index)
    : Executor(allocator, expression, attr_index) {
    if (attr_index == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "region_filter requires an attribute index");
    }
    if (this->bitset_type_ != ComputableBitsetType::FastBitset) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "region_filter currently supports FastBitset only");
    }

    auto region = std::dynamic_pointer_cast<const RegionFilterExpression>(expression);
    if (region == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "region_filter executor requires a region filter expression");
    }

    const auto& type_field = get_field_name(region->region_type, "type field");
    const auto& region_field = get_field_name(region->region_list, "region field");
    const auto& residence_field = get_field_name(region->residence_list, "residence field");
    const std::vector<int64_t> type_values{0, 1, 2, 3, 4, 5, -1};
    const auto type_field_type = attr_index->GetTypeOfField(type_field);
    if (type_field_type == AttrValueType::UINT8 || type_field_type == AttrValueType::UINT16 ||
        type_field_type == AttrValueType::UINT32 || type_field_type == AttrValueType::UINT64) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "region_filter type field must represent -1");
    }

    this->region_type_managers_ = GetIntegerManagers(attr_index, type_field, type_values);
    this->region_managers_ =
        GetIntegerManagers(attr_index, region_field, get_values(region->regions, "region values"));
    this->residence_managers_ = GetIntegerManagers(
        attr_index, residence_field, get_values(region->residences, "residence values"));
    this->trigger_managers_ = GetIntegerManagers(
        attr_index, residence_field, get_values(region->triggers, "trigger values"));
}

std::vector<const MultiBitsetManager*>
RegionFilterExecutor::GetIntegerManagers(const AttrInvertedInterfacePtr& attr_index,
                                         const std::string& field_name,
                                         const std::vector<int64_t>& values) {
    AttrValueType type;
    try {
        type = attr_index->GetTypeOfField(field_name);
    } catch (const std::exception& error) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "invalid region_filter field '" + field_name + "': " + error.what());
    }

    switch (type) {
        case AttrValueType::INT8:
            return get_managers<int8_t>(attr_index, field_name, values);
        case AttrValueType::INT16:
            return get_managers<int16_t>(attr_index, field_name, values);
        case AttrValueType::INT32:
            return get_managers<int32_t>(attr_index, field_name, values);
        case AttrValueType::INT64:
            return get_managers<int64_t>(attr_index, field_name, values);
        case AttrValueType::UINT8:
            return get_managers<uint8_t>(attr_index, field_name, values);
        case AttrValueType::UINT16:
            return get_managers<uint16_t>(attr_index, field_name, values);
        case AttrValueType::UINT32:
            return get_managers<uint32_t>(attr_index, field_name, values);
        case AttrValueType::UINT64:
            return get_managers<uint64_t>(attr_index, field_name, values);
        default:
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "region_filter field '" + field_name + "' must be integer typed");
    }
}

ComputableBitset*
RegionFilterExecutor::NewBitset() {
    return ComputableBitset::MakeRawInstance(this->bitset_type_, this->allocator_);
}

void
RegionFilterExecutor::Accumulate(ComputableBitset* target,
                                 const std::vector<const MultiBitsetManager*>& managers,
                                 BucketIdType bucket_id) {
    for (const auto* manager : managers) {
        if (manager != nullptr) {
            const auto* bitset = manager->GetOneBitset(bucket_id);
            if (bitset != nullptr) {
                target->Or(bitset);
            }
        }
    }
}

void
RegionFilterExecutor::Clear() {
    Executor::Clear();
    const std::array<ComputableBitset*, 4> scratch{
        this->region_bitset_.get(),
        this->residence_bitset_.get(),
        this->trigger_bitset_.get(),
        this->intersection_bitset_.get(),
    };
    for (auto* bitset : scratch) {
        if (bitset != nullptr) {
            bitset->Clear();
        }
    }
    for (const auto& bitset : this->type_bitsets_) {
        if (bitset != nullptr) {
            bitset->Clear();
        }
    }
}

void
RegionFilterExecutor::Init() {
    if (this->bitset_ != nullptr) {
        return;
    }

    std::unique_ptr<ComputableBitset> result(NewBitset());
    std::unique_ptr<ComputableBitset> region(NewBitset());
    std::unique_ptr<ComputableBitset> residence(NewBitset());
    std::unique_ptr<ComputableBitset> trigger(NewBitset());
    std::unique_ptr<ComputableBitset> intersection(NewBitset());
    std::array<std::unique_ptr<ComputableBitset>, TYPE_COUNT> type_bitsets;
    for (auto& bitset : type_bitsets) {
        bitset.reset(NewBitset());
    }

    this->region_bitset_ = std::move(region);
    this->residence_bitset_ = std::move(residence);
    this->trigger_bitset_ = std::move(trigger);
    this->intersection_bitset_ = std::move(intersection);
    this->type_bitsets_ = std::move(type_bitsets);
    this->bitset_ = result.release();
    this->own_bitset_ = true;
}

Filter*
RegionFilterExecutor::Run(BucketIdType bucket_id) {
    if (this->bitset_ == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "region_filter executor must be initialized before Run");
    }
    Clear();
    Accumulate(this->region_bitset_.get(), this->region_managers_, bucket_id);
    Accumulate(this->residence_bitset_.get(), this->residence_managers_, bucket_id);
    Accumulate(this->trigger_bitset_.get(), this->trigger_managers_, bucket_id);

    // ComputableBitset has no copy operation; OR from the cleared bitset copies the region bits.
    this->intersection_bitset_->Or(this->region_bitset_.get());
    this->intersection_bitset_->And(this->residence_bitset_.get());

    auto apply_type = [this, bucket_id](uint64_t manager_index, ComputableBitset* matching_values) {
        if (manager_index >= this->region_type_managers_.size()) {
            return;
        }
        const auto* manager = this->region_type_managers_[manager_index];
        if (manager == nullptr) {
            return;
        }
        const auto* type_values = manager->GetOneBitset(bucket_id);
        if (type_values == nullptr) {
            return;
        }
        this->type_bitsets_[manager_index]->Or(matching_values);
        this->type_bitsets_[manager_index]->And(type_values);
        this->bitset_->Or(this->type_bitsets_[manager_index].get());
    };

    if (this->region_type_managers_.size() > MATCH_ALL &&
        this->region_type_managers_[MATCH_ALL] != nullptr) {
        const auto* all_type = this->region_type_managers_[MATCH_ALL]->GetOneBitset(bucket_id);
        if (all_type != nullptr) {
            this->bitset_->Or(all_type);
        }
    }
    apply_type(RESIDENCE, this->residence_bitset_.get());
    apply_type(REGION, this->region_bitset_.get());

    this->type_bitsets_[REGION_OR_RESIDENCE]->Or(this->region_bitset_.get());
    this->type_bitsets_[REGION_OR_RESIDENCE]->Or(this->residence_bitset_.get());
    apply_type(REGION_OR_RESIDENCE, this->type_bitsets_[REGION_OR_RESIDENCE].get());

    apply_type(REGION_AND_RESIDENCE, this->intersection_bitset_.get());

    this->type_bitsets_[REGION_WITHOUT_RESIDENCE]->Or(this->residence_bitset_.get());
    this->type_bitsets_[REGION_WITHOUT_RESIDENCE]->Not();
    this->type_bitsets_[REGION_WITHOUT_RESIDENCE]->And(this->region_bitset_.get());
    apply_type(REGION_WITHOUT_RESIDENCE, this->type_bitsets_[REGION_WITHOUT_RESIDENCE].get());

    this->type_bitsets_[TRIGGER_OR_INTERSECTION]->Or(this->trigger_bitset_.get());
    this->type_bitsets_[TRIGGER_OR_INTERSECTION]->Or(this->intersection_bitset_.get());
    apply_type(TRIGGER_OR_INTERSECTION, this->type_bitsets_[TRIGGER_OR_INTERSECTION].get());

    this->only_bitset_ = true;
    WhiteListFilter::TryToUpdate(this->filter_, this->bitset_);
    return this->filter_;
}

}  // namespace vsag
