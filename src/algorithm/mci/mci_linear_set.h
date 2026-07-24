#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <utility>

namespace vsag::mci {

class linear_set {
private:
    uint32_t* v_set = nullptr;
    uint32_t* f_index = nullptr;
    uint32_t sz = 0;
    uint32_t capacity = 0;

public:
    linear_set() {
    }
    linear_set(uint32_t sz_) {
        resize(sz_);
    }
    void
    resize(uint32_t sz_) {
        if (capacity < sz_) {
            // sz = sz_;
            if (v_set != nullptr) {
                delete[] v_set;
                v_set = nullptr;
            }
            if (f_index != nullptr) {
                delete[] f_index;
                f_index = nullptr;
            }
            v_set = new uint32_t[sz_];
            f_index = new uint32_t[sz_];
            capacity = sz_;
        }
        sz = sz_;

        for (uint32_t i = 0; i < sz; i++) {
            v_set[i] = f_index[i] = i;
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

    ~linear_set() {
        if (f_index != nullptr) {
            delete[] f_index;
            delete[] v_set;
            f_index = nullptr;
        }
    }

    uint32_t*
    begin() {
        return v_set;
    }

    uint32_t
    operator[](uint32_t i) {
        // if(i >= g->maxSize()) {
        //     printf("error index\n"); return -1;
        // }
        return v_set[i];
    }

    void
    change_to(uint32_t u, uint32_t p) {
        uint32_t p_u = f_index[u];
        std::swap(f_index[u], f_index[v_set[p]]);
        std::swap(v_set[p_u], v_set[p]);
    }
    void
    swap_by_pos(uint32_t p_u, uint32_t p) {
        std::swap(f_index[v_set[p_u]], f_index[v_set[p]]);
        std::swap(v_set[p_u], v_set[p]);
    }

    uint32_t
    idx(uint32_t u) {
        return f_index[u];
    }
    uint32_t
    pos(uint32_t u) {
        return f_index[u];
    }

    bool
    isin(uint32_t u, uint32_t st, uint32_t ed) {
        return st <= f_index[u] && f_index[u] < ed;
    }
    bool
    is_in(uint32_t u, uint32_t st, uint32_t ed) {
        return st <= f_index[u] && f_index[u] < ed;
    }

    void
    change_to_by_pos(uint32_t p_u, uint32_t p) {
        std::swap(f_index[v_set[p_u]], f_index[v_set[p]]);
        std::swap(v_set[p_u], v_set[p]);
    }

    void
    copy(uint32_t* p, uint32_t r, uint32_t st = 0) {
        assert(st <= r);
        assert(r <= sz);

        if (r - st >= 4) {
            memcpy(p, v_set + st, sizeof(uint32_t) * (r - st));
        } else {
            for (uint32_t i = st; i < r; i++) {
                p[i - st] = v_set[i];
            }
        }
    }
};

}  // namespace vsag::mci
