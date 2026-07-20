#pragma once

#include <cstdint>

#include "simd/mci_hash_simd.h"
// #include "../tools/fastIO.hpp"

#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>
namespace vsag::mci {

using pair = std::pair<uint32_t, uint32_t>;

// namespace CCRMCE {

class cuckoo_hash {
    using int32 = int32_t;
    const int unfilled = -1;

private:
    /* data */
    int32 capacity;
    int32 mask;
    int32 size;
    int32 buff_size = sizeof(int32);
    int32* hashtable = nullptr;

    void
    rehash(int32** _table) {
        int32 oldcapacity = capacity;
        mask = mask == 0 ? 1 : ((mask << 1) | 1);
        capacity = (mask + 1) * buff_size;
        int32* newhash = new int32[capacity];
        memset((newhash), unfilled, sizeof(int32) * capacity);
        for (int32 i = 0; i < oldcapacity; ++i) {
            if ((*_table)[i] != unfilled)
                insert((*_table)[i], &newhash);
        }
        std::swap((*_table), newhash);
        delete[] newhash;
    }
    void
    insert(const int32& _u, int32** _table) {
        int32 hs = hash1(_u);
        for (int32 i = 0; i < buff_size; ++i) {
            if ((*_table)[hs * buff_size + i] == unfilled) {
                (*_table)[hs * buff_size + i] = _u;
                return;
            }
        }
        hs = hash2(_u);
        for (int32 i = 0; i < buff_size; ++i) {
            if ((*_table)[hs * buff_size + i] == unfilled) {
                (*_table)[hs * buff_size + i] = _u;
                return;
            }
        }

        bool use_hash1 = true;
        int32 u = _u;
        for (int32 i = 0; i < mask; ++i) {
            int32 replaced;
            if (use_hash1)
                hs = hash1(u);
            else
                hs = hash2(u);
            int32 j = 0;
            for (; j < buff_size; ++j) {
                if ((*_table)[hs * buff_size + j] == unfilled)
                    break;
            }
            if (buff_size == j) {
                replaced = std::move((*_table)[hs * buff_size]);
                j = 1;
                for (; j < buff_size; j++) {
                    (*_table)[hs * buff_size + j - 1] = std::move((*_table)[hs * buff_size + j]);
                }
                (*_table)[hs * buff_size + j - 1] = u;
            } else {
                replaced = std::move((*_table)[hs * buff_size + j]);
                (*_table)[hs * buff_size + j] = u;
            }
            use_hash1 = hs == hash2(replaced);
            u = std::move(replaced);
            if (u == unfilled)
                return;
        }
        rehash(_table);
        insert(u, _table);
    }

    int32
    hash1(const int32 x) {
        return x & mask;
    }
    int32
    hash2(const int32 x) {
        return ~x & mask;
    }

public:
    cuckoo_hash(/* args */) {
        capacity = 0;
        hashtable = nullptr;
        mask = 0;
        size = 0;
    }
    ~cuckoo_hash() {
        if (hashtable != nullptr) {
            delete[] hashtable;
            hashtable = nullptr;
        }
    }

    void
    reserve(int32 _size) {
        if (capacity >= _size)
            return;
        mask = mask == 0 ? 1 : ((mask << 1) | 1);
        while (_size >= mask * buff_size) mask = (mask << 1) | 1;
        capacity = (mask + 1) * buff_size;
        if (hashtable != nullptr) {
            delete[] hashtable;
            hashtable = nullptr;
        }
        hashtable = new int32[capacity];
        memset(hashtable, unfilled, sizeof(int32) * capacity);
    }

    void
    insert(const int32& _u) {
        if (find(_u))
            return;
        insert(_u, &hashtable);
        size++;
    }

    bool
    find(const int32& _u) {
        int32 hs1 = hash1(_u);

        int32 hs2 = hash2(_u);

        return simd::mci_hash_contains_two_int32x4(
            &hashtable[buff_size * hs1], &hashtable[buff_size * hs2], _u);
    }
    int32
    get_capacity() {
        return capacity;
    }
    int32
    get_mask() {
        return mask;
    }
    int32*
    get_hashtable() {
        return hashtable;
    }
};

class graph_mce {
public:
    uint32_t n, m;
    uint32_t max_d = 0;
    std::vector<uint32_t> p_idx, p_idx2, p_edge;
    std::vector<std::vector<uint32_t>> p_deep_idx;
    std::vector<std::vector<uint32_t>> deg;
    std::vector<pair> edges;
    uint32_t core_number;

    //mp[i] = v; mp2[v] = i;
    std::vector<uint32_t> mp, mp2;

    graph_mce() {
    }

    void
    build_csr();

    void
    change_to_core_order();

    uint32_t
    degree(uint32_t u);
    uint32_t
    degree2(uint32_t u) {
        return p_idx2[u] - p_idx[u];
    }
    // uint32_t degree_deep(uint32_t deep, uint32_t u) { return p_deep_idx[deep][u] - p_idx[u]; }
    uint32_t
    degree_deep(uint32_t deep, uint32_t u) {
        return deg[deep][u];
    }

    bool
    connect(uint32_t u, uint32_t v);
    bool
    connect2(uint32_t u, uint32_t v);
    bool
    connect_out(uint32_t u, uint32_t v);

    void
    print() {
        for (uint32_t u = 0; u < n; u++) {
            printf("%u:", u);
            for (uint32_t i = p_idx[u]; i < p_idx[u + 1]; i++) {
                printf("%u ", p_edge[i]);
            }
            printf("\n");
        }
    }

private:  //hash
    std::vector<cuckoo_hash> cuhash;

public:
    bool
    connect_hash(uint32_t u, uint32_t v) {
        return cuhash[u].find(v);
    }
    void
    init_hash() {
        cuhash.resize(n);
        for (uint32_t u = 0; u < n; u++) {
            cuhash[u].reserve(p_idx[u + 1] - p_idx[u]);
            for (uint32_t i = p_idx[u]; i < p_idx[u + 1]; i++) {
                cuhash[u].insert(p_edge[i]);
            }
        }
    }
};

// }

}  // namespace vsag::mci
