#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace vsag::mci {

class list_linear_heap {
public:
    uint32_t n = 0;    // number vertices
    uint32_t key_cap;  // the maximum allowed key value

    uint32_t max_key;  // possible max key
    uint32_t min_key;  // possible min key

    uint32_t* keys = nullptr;  // keys of vertices
        // keys[i] > key_cap if vertex i is not in the data structure

    uint32_t* heads = nullptr;  // head of doubly-linked list for a specific weight
    uint32_t* pres = nullptr;   // pre for doubly-linked list
    uint32_t* nexts = nullptr;  // next for doubly-linked list

public:
    list_linear_heap(uint32_t _n, uint32_t _key_cap) {
        n = _n;
        key_cap = _key_cap;

        min_key = key_cap;
        max_key = 0;

        heads = keys = pres = nexts = nullptr;
    }
    ~list_linear_heap() {
        if (heads != nullptr) {
            delete[] heads;
            heads = nullptr;
        }
        if (pres != nullptr) {
            delete[] pres;
            pres = nullptr;
        }
        if (nexts != nullptr) {
            delete[] nexts;
            nexts = nullptr;
        }
        if (keys != nullptr) {
            delete[] keys;
            keys = nullptr;
        }
    }

    void
    reserve(uint32_t _n, uint32_t _key_cap) {
        if (n < _n) {
            if (heads != nullptr) {
                delete[] heads;
                heads = nullptr;
            }
            if (pres != nullptr) {
                delete[] pres;
                pres = nullptr;
            }
            if (nexts != nullptr) {
                delete[] nexts;
                nexts = nullptr;
            }
            if (keys != nullptr) {
                delete[] keys;
                keys = nullptr;
            }
            // n = _n;
        }
        n = _n;
        if (keys == nullptr)
            keys = new uint32_t[n];
        if (pres == nullptr)
            pres = new uint32_t[n];
        if (nexts == nullptr)
            nexts = new uint32_t[n];
        if (heads == nullptr)
            heads = new uint32_t[key_cap + 1];
    }

    void
    init_raw(uint32_t _n, uint32_t _key_cap, uint32_t* _id_s, uint32_t* _key_s) {
        uint32_t required_n = n;
        for (uint32_t i = 0; i < _n; ++i) {
            required_n = std::max(required_n, _id_s[i] + 1);
        }
        if (n < required_n) {
            if (heads != nullptr) {
                delete[] heads;
                heads = nullptr;
            }
            if (pres != nullptr) {
                delete[] pres;
                pres = nullptr;
            }
            if (nexts != nullptr) {
                delete[] nexts;
                nexts = nullptr;
            }
            if (keys != nullptr) {
                delete[] keys;
                keys = nullptr;
            }
        }
        if (heads != nullptr and key_cap < _key_cap) {
            delete[] heads;
            heads = nullptr;
        }
        n = required_n;
        key_cap = _key_cap;
        if (keys == nullptr)
            keys = new uint32_t[n];
        if (pres == nullptr)
            pres = new uint32_t[n];
        if (nexts == nullptr)
            nexts = new uint32_t[n];
        if (heads == nullptr)
            heads = new uint32_t[key_cap + 1];

        assert(_key_cap <= key_cap);
        min_key = max_key = _key_cap;
        for (uint32_t i = 0; i <= _key_cap; i++) heads[i] = n;

        for (uint32_t i = 0; i < _n; i++) {
            uint32_t id = _id_s[i];
            uint32_t key = _key_s[id];
            assert(id < n);
            assert(key <= _key_cap);

            keys[id] = key;
            pres[id] = n;
            nexts[id] = heads[key];
            if (heads[key] != n)
                pres[heads[key]] = id;
            heads[key] = id;

            if (key < min_key)
                min_key = key;
        }
    }

    // initialize the data structure by (id, key) pairs
    // _n is the number of pairs, _key_cap is the maximum possible key value
    void
    init(uint32_t _n, uint32_t _key_cap, uint32_t* _ids, uint32_t* _keys) {
        n = _n;
        key_cap = _key_cap;
        if (keys == nullptr)
            keys = new uint32_t[n];
        if (pres == nullptr)
            pres = new uint32_t[n];
        if (nexts == nullptr)
            nexts = new uint32_t[n];
        if (heads == nullptr)
            heads = new uint32_t[key_cap + 1];
        assert(_key_cap <= key_cap);
        min_key = _key_cap;
        max_key = 0;
        for (uint32_t i = 0; i <= _key_cap; i++) heads[i] = n;

        for (uint32_t i = 0; i < _n; i++) insert(_ids[i], _keys[i]);
    }

    // insert (id, key) pair into the data structure
    void
    insert(uint32_t id, uint32_t key) {
        assert(id < n);
        assert(key <= key_cap);
        //assert(keys[id] > key_cap);
        // printf("%u %u\n", id, key);
        keys[id] = key;
        pres[id] = n;
        nexts[id] = heads[key];
        if (heads[key] != n)
            pres[heads[key]] = id;
        heads[key] = id;

        if (key < min_key)
            min_key = key;
        if (key > max_key)
            max_key = key;
    }

    // remove a vertex from the data structure
    uint32_t
    remove(uint32_t id) {
        // assert(id < n);
        // assert(keys[id] <= max_key);
        // printf("remove %u %u\n", id, n);fflush(stdout);
        if (pres[id] == n) {
            assert(heads[keys[id]] == id);
            heads[keys[id]] = nexts[id];
            if (nexts[id] != n)
                pres[nexts[id]] = n;
        } else {
            uint32_t pid = pres[id];
            nexts[pid] = nexts[id];
            if (nexts[id] != n)
                pres[nexts[id]] = pid;
        }

        return keys[id];
    }

    uint32_t
    get_n() {
        return n;
    }
    uint32_t
    get_key_cap() {
        return key_cap;
    }
    uint32_t
    get_key(uint32_t id) {
        return keys[id];
    }

    void
    get_ids(std::vector<uint32_t>& ids) {
        ids.clear();
        tighten();
        for (uint32_t i = min_key; i <= max_key; i++) {
            for (uint32_t id = heads[i]; id != n; id = nexts[id]) {
                ids.push_back(id);
            }
        }
    }
    void
    get_ids(uint32_t* vs, uint32_t& vs_size) {
        tighten();
        for (uint32_t i = min_key; i <= max_key; i++) {
            for (uint32_t id = heads[i]; id != n; id = nexts[id]) {
                vs[vs_size++] = id;
            }
        }
    }

    void
    print() {
        tighten();
        for (uint32_t i = min_key; i <= max_key; i++) {
            printf("key %u:", i);
            for (uint32_t id = heads[i]; id != n; id = nexts[id]) {
                printf("%u ", id);
            }
            printf("\n");
        }
    }

    void
    get_ids_keys(std::vector<uint32_t>& ids, std::vector<uint32_t>& _keys) {
        ids.clear();
        _keys.clear();
        tighten();
        for (uint32_t i = min_key; i <= max_key; i++) {
            for (uint32_t id = heads[i]; id != n; id = nexts[id]) {
                ids.push_back(id);
                _keys.push_back(id);
            }
        }
    }

    bool
    empty() {
        tighten();
        return min_key > max_key;
    }

    uint32_t
    size() {
        tighten();
        uint32_t res = 0;
        for (uint32_t i = min_key; i <= max_key; i++)
            for (uint32_t id = heads[i]; id != n; id = nexts[id]) ++res;
        return res;
    }

    // get the (id,key) pair with the maximum key value; return true if success, return false otherwise
    bool
    get_max(uint32_t& id, uint32_t& key) {
        if (empty())
            return false;

        id = heads[max_key];
        key = max_key;
        assert(keys[id] == key);
        return true;
    }

    // pop the (id,key) pair with the maximum key value; return true if success, return false otherwise
    bool
    pop_max(uint32_t& id, uint32_t& key) {
        if (empty())
            return false;

        id = heads[max_key];
        key = max_key;
        assert(keys[id] == key);

        heads[max_key] = nexts[id];
        if (heads[max_key] != n)
            pres[heads[max_key]] = n;

        tighten();
        return true;
    }

    // get the (id,key) pair with the minimum key value; return true if success, return false otherwise
    bool
    get_min(uint32_t& id, uint32_t& key) {
        if (empty())
            return false;

        id = heads[min_key];
        key = min_key;
        assert(keys[id] == key);

        return true;
    }

    // pop the (id,key) pair with the minimum key value; return true if success, return false otherwise
    bool
    pop_min(uint32_t& id, uint32_t& key) {
        if (empty())
            return false;

        id = heads[min_key];
        key = min_key;

        assert(keys[id] == key);
        keys[id] = 0;

        heads[min_key] = nexts[id];
        if (heads[min_key] != n)
            pres[heads[min_key]] = n;

        tighten();
        return true;
    }

    // increment the key of vertex id by inc
    uint32_t
    increment(uint32_t id, uint32_t inc = 1) {
        assert(keys[id] + inc <= key_cap);

        uint32_t new_key = keys[id] + inc;

        remove(id);
        insert(id, new_key);

        return new_key;
    }

    // decrement the key of vertex id by dec
    uint32_t
    decrement(uint32_t id, uint32_t dec = 1) {
        // assert(keys[id] >= dec);
        if (keys[id] < dec)
            return 0;
        uint32_t new_key = keys[id] - dec;

        remove(id);
        insert(id, new_key);

        tighten();
        return new_key;
    }

private:
    void
    tighten() {
        while (min_key <= max_key && heads[min_key] == n) ++min_key;
        while (max_key > 0 && min_key <= max_key && heads[max_key] == n) --max_key;
    }
};

}  // namespace vsag::mci
