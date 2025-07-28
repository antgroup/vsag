
#include "meta_table.h"

#include <algorithm>
#include <cstring>
void
MetaTable::InsertMetaItem(const MetaItem& meta_item) {
    auto iter = std::lower_bound(this->meta_items_.begin(), this->meta_items_.end(), meta_item);
    this->meta_items_.insert(iter, meta_item);
}

const char*
MetaTable::GetDataPtr(uint64_t offset, uint64_t size) {
    auto iter = std::upper_bound(
        this->meta_items_.begin(), this->meta_items_.end(), MetaItem(nullptr, 0, offset));
    if (iter == this->meta_items_.begin()) {
        return nullptr;
    }
    iter--;
    if (iter->offset + iter->size < offset + size) {
        char* data = new char[size];
        int64_t cur = 0;
        int64_t end = offset + size;
        while (iter != this->meta_items_.end()) {
            if (iter->offset + iter->size < end) {
                auto len = iter->size + iter->offset - offset;
                memcpy(data + cur, iter->addr + (offset - iter->offset), len);
                cur += len;
                offset += len;
                size -= len;
                iter++;
            } else {
                memcpy(data + cur, iter->addr, size);
                break;
            }
        }
        return data;
    } else {
        return iter->addr + (offset - iter->offset);
    }
}