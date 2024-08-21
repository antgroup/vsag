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

#include <cstring>
#include <deque>
#include <functional>
#include <mutex>

#include "../../default_allocator.h"

namespace vsag {

extern void*
allocate(size_t size);

extern void
deallocate(void* p);

extern void*
reallocate(void* p, size_t size);

}  // namespace vsag

namespace hnswlib {
typedef unsigned short int vl_type;

class VisitedList {
public:
    vl_type curV;
    vl_type* mass;
    unsigned int numelements;

    VisitedList(int numelements1, vsag::Allocator* allocator) : allocator_(allocator) {
        curV = -1;
        numelements = numelements1;
        mass = (vl_type*)allocator_->Allocate(numelements * sizeof(vl_type));
    }

    void
    reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    ~VisitedList() {
        allocator_->Deallocate(mass);
    }

    vsag::Allocator* allocator_;
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    std::deque<VisitedList*> pool;
    std::mutex poolguard;
    int numelements;

public:
    VisitedListPool(int initmaxpools, int numelements1, vsag::Allocator* allocator)
        : allocator_(allocator) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements, allocator_));
    }

    VisitedList*
    getFreeVisitedList() {
        VisitedList* rez;
        {
            std::unique_lock<std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements, allocator_);
            }
        }
        rez->reset();
        return rez;
    }

    void
    releaseVisitedList(VisitedList* vl) {
        std::unique_lock<std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList* rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }

private:
    vsag::Allocator* allocator_;
};

class BlockManager {
public:
    BlockManager(size_t max_elements,
                 size_t size_data_per_element,
                 size_t block_size_limit,
                 vsag::Allocator* allocator)
        : max_elements_(max_elements),
          size_data_per_element_(size_data_per_element),
          allocator_(allocator) {
        data_num_per_block_ = block_size_limit / size_data_per_element_;
        block_size_ = size_data_per_element * data_num_per_block_;
        size_t full_blocks = (max_elements * size_data_per_element) / block_size_;
        size_t remaining_size = (max_elements * size_data_per_element) % block_size_;
        for (size_t i = 0; i < full_blocks; ++i) {
            blocks_.push_back(static_cast<char*>(allocator_->Allocate(block_size_)));
            block_lens_.push_back(block_size_);
        }
        if (remaining_size > 0) {
            blocks_.push_back(static_cast<char*>(allocator_->Allocate(remaining_size)));
            block_lens_.push_back(remaining_size);
        }
    }

    ~BlockManager() {
        for (char* block : blocks_) {
            allocator_->Deallocate(block);
        }
    }

    char*
    getElementPtr(size_t index, size_t offset) {
        if (index >= max_elements_) {
            throw std::out_of_range("Index is out of range:" + std::to_string(index));
        }

        size_t block_index = (index * size_data_per_element_) / block_size_;
        size_t offset_in_block = (index * size_data_per_element_) % block_size_;
        return blocks_[block_index] + offset_in_block + offset;
    }

    bool
    resize(size_t new_max_elements) {
        if (new_max_elements < max_elements_) {
            throw std::runtime_error("new_max_elements is less than max_elements_");
        }

        size_t new_full_blocks = (new_max_elements * size_data_per_element_) / block_size_;
        size_t new_remaining_size = (new_max_elements * size_data_per_element_) % block_size_;

        try {
            bool append_more_block = blocks_.size() <= new_full_blocks;
            // Adjust the size of the last block. There are two scenarios here: when more blocks
            // need to be padded, the last block should be converted from a remaining_block to a
            // full_block; otherwise, the size of the remaining_block should be increased to make
            // it a larger remaining_block.
            if (!blocks_.empty() && blocks_.back() != nullptr &&
                block_lens_.back() != block_size_) {
                char* last_block = blocks_.back();

                size_t new_last_block_size = append_more_block ? block_size_ : new_remaining_size;
                auto new_last_block = allocator_->Reallocate(last_block, new_last_block_size);
                if (new_last_block == nullptr) {
                    return false;
                }
                blocks_.back() = static_cast<char*>(new_last_block);
                block_lens_.back() = new_last_block_size;
            }

            // If the current number of blocks is less than the number of complete blocks needed, proceed with padding.
            while (blocks_.size() < new_full_blocks) {
                blocks_.push_back(static_cast<char*>(allocator_->Allocate(block_size_)));
                block_lens_.push_back(block_size_);
            }

            // Padding the last block is necessary only when there are not enough blocks.
            if (new_remaining_size > 0 && append_more_block) {
                blocks_.push_back(static_cast<char*>(allocator_->Allocate(new_remaining_size)));
                block_lens_.push_back(new_remaining_size);
            }
            max_elements_ = new_max_elements;
            return true;
        } catch (const std::bad_alloc&) {
            return false;
        }
    }

    bool
    serialize(char*& buffer, size_t cur_element_count) {
        size_t store_size = cur_element_count * size_data_per_element_;
        size_t offset = 0;
        for (int i = 0; i < blocks_.size(); ++i) {
            size_t new_offset = offset + block_lens_[i];
            size_t current_block_size = std::min(new_offset, store_size) - offset;
            std::memcpy(buffer + offset, blocks_[i], current_block_size);
            offset = new_offset;
            if (new_offset >= store_size) {
                break;
            }
        }
        buffer += store_size;
        return true;
    }

    bool
    serialize(std::ostream& ofs, size_t cur_element_count) {
        size_t store_size = cur_element_count * size_data_per_element_;
        try {
            size_t offset = 0;
            for (int i = 0; i < blocks_.size(); ++i) {
                size_t new_offset = offset + block_lens_[i];
                size_t current_block_size = std::min(new_offset, store_size) - offset;
                ofs.write(blocks_[i], current_block_size);
                offset = new_offset;
                if (new_offset >= store_size) {
                    break;
                }
            }
        } catch (const std::ios_base::failure&) {
            return false;
        }
        return true;
    }

    void
    deserialize(std::function<void(uint64_t, uint64_t, void*)> read_func,
                uint64_t cursor,
                size_t cur_element_count) {
        size_t offset = 0;
        size_t need_read_size = cur_element_count * size_data_per_element_;
        for (size_t i = 0; i < blocks_.size(); ++i) {
            size_t current_read_size = std::min(need_read_size, offset + block_lens_[i]) - offset;
            read_func(cursor + offset, current_read_size, blocks_[i]);
            offset += block_lens_[i];
            if (offset >= need_read_size) {
                break;
            }
        }
    }

    bool
    deserialize(std::istream& ifs, size_t cur_element_count) {
        try {
            size_t offset = 0;
            size_t need_read_size = cur_element_count * size_data_per_element_;
            for (size_t i = 0; i < blocks_.size(); ++i) {
                size_t current_read_size =
                    std::min(need_read_size, offset + block_lens_[i]) - offset;
                ifs.read(blocks_[i], current_read_size);
                offset += block_lens_[i];
                if (offset >= need_read_size) {
                    break;
                }
            }
        } catch (const std::ios_base::failure&) {
            return false;
        }
        return true;
    }

    size_t
    getSize() {
        return max_elements_ * size_data_per_element_;
    }

private:
    std::vector<char*> blocks_;
    size_t data_num_per_block_ = 0;
    size_t block_size_ = 0;
    size_t size_data_per_element_;
    size_t max_elements_;
    std::vector<size_t> block_lens_;
    vsag::Allocator* allocator_;
};

}  // namespace hnswlib
