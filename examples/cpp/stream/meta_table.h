
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

struct MetaItem {
public:
    MetaItem(char* addr, uint64_t size, uint64_t offset) : addr(addr), size(size), offset(offset) {
    }

    MetaItem() = default;

    bool
    operator<(const MetaItem& other) const {
        return offset < other.offset;
    }

public:
    char* addr{nullptr};
    uint64_t size{0};
    uint64_t offset{0};
};

class MetaTable {
public:
    MetaTable() = default;
    ~MetaTable() = default;

    void
    InsertMetaItem(const MetaItem& meta_item);

    const char*
    GetDataPtr(uint64_t offset, uint64_t size);

    uint64_t
    Size() {
        return meta_items_.size();
    }

private:
    std::vector<MetaItem> meta_items_;
};
