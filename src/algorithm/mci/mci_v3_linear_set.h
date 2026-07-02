#pragma once

#include <cassert>
#include <cstring>
#include <tuple>
#include <utility>

#include "mci_v3_types.h"

namespace vsag::mci_v3 {

class LinearSet {
private:
    uint32_t* vSet = nullptr;
    uint32_t* fIndex = nullptr;
    uint32_t sz = 0;
    uint32_t capacity = 0;

public:
    LinearSet() {
    }
    LinearSet(uint32_t sz_) {
        resize(sz_);
    }
    void
    resize(uint32_t sz_) {
        if (capacity < sz_) {
            // sz = sz_;
            if (vSet != nullptr) {
                delete[] vSet;
                vSet = nullptr;
            }
            if (fIndex != nullptr) {
                delete[] fIndex;
                fIndex = nullptr;
            }
            vSet = new uint32_t[sz_];
            fIndex = new uint32_t[sz_];
            capacity = sz_;
        }
        sz = sz_;

        for (uint32_t i = 0; i < sz; i++) {
            vSet[i] = fIndex[i] = i;
        }
    }

    uint32_t
    size() {
        return sz;
    }
    uint32_t
    cap() {
        return capacity;
    }

    ~LinearSet() {
        if (fIndex != nullptr) {
            delete[] fIndex;
            delete[] vSet;
            fIndex = nullptr;
        }
    }

    uint32_t*
    begin() {
        return vSet;
    }

    uint32_t
    operator[](uint32_t i) {
        // if(i >= g->maxSize()) {
        //     printf("error index\n"); return -1;
        // }
        return vSet[i];
    }

    void
    changeTo(uint32_t u, uint32_t p) {
        uint32_t pU = fIndex[u];
        std::swap(fIndex[u], fIndex[vSet[p]]);
        std::swap(vSet[pU], vSet[p]);
    }
    void
    swapByPos(uint32_t pU, uint32_t p) {
        std::swap(fIndex[vSet[pU]], fIndex[vSet[p]]);
        std::swap(vSet[pU], vSet[p]);
    }

    uint32_t
    idx(uint32_t u) {
        return fIndex[u];
    }
    uint32_t
    pos(uint32_t u) {
        return fIndex[u];
    }

    bool
    isin(uint32_t u, uint32_t st, uint32_t ed) {
        return st <= fIndex[u] && fIndex[u] < ed;
    }
    bool
    isIn(uint32_t u, uint32_t st, uint32_t ed) {
        return st <= fIndex[u] && fIndex[u] < ed;
    }

    void
    changeToByPos(uint32_t pU, uint32_t p) {
        std::swap(fIndex[vSet[pU]], fIndex[vSet[p]]);
        std::swap(vSet[pU], vSet[p]);
    }

    void
    copy(uint32_t* p, uint32_t r, uint32_t st = 0) {
        assert(st <= r);
        assert(r <= sz);

        if (r - st >= 4) {
            memcpy(p, vSet + st, sizeof(uint32_t) * (r - st));
        } else {
            for (uint32_t i = st; i < r; i++) {
                p[i - st] = vSet[i];
            }
        }
    }
};

}  // namespace vsag::mci_v3
