#pragma once

#include "mci_v3_types.h"
// #include "../tools/fastIO.hpp"
#include <immintrin.h>

#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>
namespace vsag::mci_v3 {

using Pair = std::pair<ui, ui>;

// namespace CCRMCE {

class CuckooHash {
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
    CuckooHash(/* args */) {
        capacity = 0;
        hashtable = nullptr;
        mask = 0;
        size = 0;
    }
    ~CuckooHash() {
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

        // assert(buff_size == 4 && sizeof (int32) == 4);
        __m128i cmp = _mm_set1_epi32(_u);
        // if(buff_size*hs1 >= capacity) {
        // 	printf("hs1 %d, cap %d\n", hs1, capacity);fflush(stdout);
        // }
        // assert(buff_size*hs1 < capacity);
        __m128i b1 = _mm_load_si128((__m128i*)&hashtable[buff_size * hs1]);
        __m128i b2 = _mm_load_si128((__m128i*)&hashtable[buff_size * hs2]);
        __m128i flag = _mm_or_si128(_mm_cmpeq_epi32(cmp, b1), _mm_cmpeq_epi32(cmp, b2));

        return _mm_movemask_epi8(flag) != 0;
    }
    int32
    getcapacity() {
        return capacity;
    }
    int32
    getmask() {
        return mask;
    }
    int32*
    gethashtable() {
        return hashtable;
    }
};

class Graphmce {
public:
    ui n, m;
    ui maxD = 0;
    std::vector<ui> pIdx, pIdx2, pEdge;
    std::vector<std::vector<ui>> pDeepIdx;
    std::vector<std::vector<ui>> deg;
    std::vector<Pair> edges;
    ui coreNumber;

    //mp[i] = v; mp2[v] = i;
    std::vector<ui> mp, mp2;

    Graphmce() {
    }

    void
    buildCSR();

    void
    changeToCoreOrder();

    ui
    degree(ui u);
    ui
    degree2(ui u) {
        return pIdx2[u] - pIdx[u];
    }
    // ui degreeDeep(ui deep, ui u) { return pDeepIdx[deep][u] - pIdx[u]; }
    ui
    degreeDeep(ui deep, ui u) {
        return deg[deep][u];
    }

    bool
    connect(ui u, ui v);
    bool
    connect2(ui u, ui v);
    bool
    connectOut(ui u, ui v);

    void
    print() {
        for (ui u = 0; u < n; u++) {
            printf("%u:", u);
            for (ui i = pIdx[u]; i < pIdx[u + 1]; i++) {
                printf("%u ", pEdge[i]);
            }
            printf("\n");
        }
    }

private:  //hash
    std::vector<CuckooHash> cuhash;

public:
    bool
    connectHash(ui u, ui v) {
        return cuhash[u].find(v);
    }
    void
    initHash() {
        cuhash.resize(n);
        for (ui u = 0; u < n; u++) {
            cuhash[u].reserve(pIdx[u + 1] - pIdx[u]);
            for (ui i = pIdx[u]; i < pIdx[u + 1]; i++) {
                cuhash[u].insert(pEdge[i]);
            }
        }
    }
};

// }

}  // namespace vsag::mci_v3
