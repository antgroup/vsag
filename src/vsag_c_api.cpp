
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

#include <vsag/factory.h>
#include <vsag/index.h>
#include <vsag/vsag_c_api.h>

#include <cstring>
#include <fstream>
#include <streambuf>
#include <type_traits>

Error_t success = {VSAG_SUCCESS, "success"};

class VsagIndex {
public:
    VsagIndex(std::shared_ptr<vsag::Index> index) : index_(std::move(index)) {
    }

    std::shared_ptr<vsag::Index> index_;
};

static Error_t
make_error(const vsag::Error& error) {
    Error_t err;
    err.code = -static_cast<int>(error.type);
    const auto& msg = error.message;
    snprintf(err.message, sizeof(err.message), "%s", msg.c_str());
    return err;
}

static Error_t
make_error(const std::exception& e) {
    Error_t err;
    err.code = VSAG_INTERNAL_ERROR;
    const auto* msg = e.what();
    // Use snprintf for safe string copying
    snprintf(err.message, sizeof(err.message), "%s", msg);
    return err;
}

extern "C" {
vsag_index_t
vsag_index_factory(const char* index_name, const char* index_param) {
    try {
        auto index = vsag::Factory::CreateIndex(index_name, index_param);
        if (index.has_value()) {
            return new VsagIndex(index.value());
        }
        return nullptr;

    } catch (const std::exception& e) {
        return nullptr;
    }
}

Error_t
vsag_index_destroy(vsag_index_t index) {
    try {
        auto* vsag_index = static_cast<VsagIndex*>(index);
        delete vsag_index;
        return success;
    } catch (const std::exception& e) {
        return make_error(e);
    }
}

Error_t
vsag_index_build(
    vsag_index_t index, const float* data, const int64_t* ids, uint64_t dim, uint64_t count) {
    try {
        auto* vsag_index = static_cast<VsagIndex*>(index);
        if (vsag_index != nullptr) {
            auto dataset = vsag::Dataset::Make();
            dataset->Owner(false)
                ->Dim(static_cast<int64_t>(dim))
                ->NumElements(static_cast<int64_t>(count))
                ->Ids(ids)
                ->Float32Vectors(data);
            auto build_result = vsag_index->index_->Build(dataset);
            if (build_result.has_value()) {
                return success;
            }

            return make_error(build_result.error());
        }
        return success;
    } catch (const std::exception& e) {
        return make_error(e);
    }
}

Error_t
vsag_index_knn_search(vsag_index_t index,
                      const float* query,
                      uint64_t dim,
                      int64_t k,
                      const char* parameters,
                      float* results,
                      int64_t* ids) {
    try {
        auto* vsag_index = static_cast<VsagIndex*>(index);
        if (vsag_index != nullptr) {
            auto query_dataset = vsag::Dataset::Make();
            query_dataset->Owner(false)
                ->Dim(static_cast<int64_t>(dim))
                ->NumElements(static_cast<int64_t>(1))
                ->Float32Vectors(query);
            auto result = vsag_index->index_->KnnSearch(query_dataset, k, parameters);
            if (result.has_value()) {
                const auto* ids_view = result.value()->GetIds();
                const auto* dists_view = result.value()->GetDistances();
                auto real_k = result.value()->GetDim();
                for (int i = 0; i < real_k; ++i) {
                    ids[i] = ids_view[i];
                    results[i] = dists_view[i];
                }
            } else {
                return make_error(result.error());
            }
        }
        return success;
    } catch (const std::exception& e) {
        return make_error(e);
    }
}

Error_t
vsag_serialize_file(vsag_index_t index, const char* file_path) {
    try {
        auto* vsag_index = static_cast<VsagIndex*>(index);
        if (vsag_index != nullptr) {
            std::ofstream file(file_path, std::ios::binary);
            auto serialize_result = vsag_index->index_->Serialize(file);
            file.close();
            if (serialize_result.has_value()) {
                return success;
            }
            return make_error(serialize_result.error());
        }
        return success;
    } catch (const std::exception& e) {
        return make_error(e);
    }
}

Error_t
vsag_deserialize_file(vsag_index_t index, const char* file_path) {
    try {
        auto* vsag_index = static_cast<VsagIndex*>(index);
        if (vsag_index != nullptr) {
            std::ifstream file(file_path, std::ios::binary);
            vsag_index->index_->Deserialize(file);
            file.close();
        }
        return success;
    } catch (const std::exception& e) {
        return make_error(e);
    }
}

Error_t
vsag_serialize_write_func(vsag_index_t index,
                          void (*write_func)(OffsetType offset, SizeType size, const void* data)) {
    try {
        auto* vsag_index = static_cast<VsagIndex*>(index);
        if (vsag_index != nullptr) {
            auto result = vsag_index->index_->Serialize(write_func);
            if (not result.has_value()) {
                return make_error(result.error());
            }
        }
        return success;
    } catch (const std::exception& e) {
        return make_error(e);
    }
}

class RandomAccessStreamBuf : public std::streambuf {
public:
    using ReadFunc = void (*)(OffsetType offset, SizeType size, void* data);
    using SizeCallback = std::function<OffsetType()>;

    explicit RandomAccessStreamBuf(ReadFunc read_func, SizeCallback size_cb = nullptr)
        : read_func_(read_func), size_cb_(std::move(size_cb)) {
    }

protected:
    std::streamsize
    xsgetn(char* s, std::streamsize count) override {
        if (read_func_ == nullptr or count <= 0) {
            return 0;
        }

        auto n = static_cast<SizeType>(count);

        read_func_(current_pos_, n, s);

        current_pos_ += n;
        return count;
    }

    pos_type
    seekoff(off_type off,
            std::ios_base::seekdir way,
            std::ios_base::openmode which = std::ios_base::in) override {
        if (which != std::ios_base::in) {
            return {-1};
        }

        OffsetType new_pos;
        switch (way) {
            case std::ios_base::beg:
                if (off < 0) {
                    return {-1};
                }
                new_pos = static_cast<OffsetType>(off);
                break;
            case std::ios_base::cur:
                if (off >= 0) {
                    new_pos = current_pos_ + static_cast<OffsetType>(off);
                } else {
                    auto abs_off = static_cast<OffsetType>(-off);
                    if (abs_off > current_pos_) {
                        return {-1};
                    }
                    new_pos = current_pos_ - abs_off;
                }
                break;
            case std::ios_base::end:
                if (!size_cb_) {
                    return {-1};
                }
                new_pos = size_cb_() + static_cast<OffsetType>(off);
                break;
            default:
                return {-1};
        }

        current_pos_ = new_pos;
        return {static_cast<off_type>(new_pos)};
    }

    pos_type
    seekpos(pos_type sp, std::ios_base::openmode which = std::ios_base::in) override {
        return seekoff(off_type(sp), std::ios_base::beg, which);
    }

private:
    ReadFunc read_func_;
    SizeCallback size_cb_;
    OffsetType current_pos_{0};
};

Error_t
vsag_deserialize_read_func(vsag_index_t index,
                           void (*read_func)(OffsetType offset, SizeType size, void* data),
                           SizeType (*size_func)()) {
    try {
        auto* vsag_index = static_cast<VsagIndex*>(index);
        if (vsag_index != nullptr) {
            RandomAccessStreamBuf stream_buf(read_func, size_func);
            std::istream stream(&stream_buf);
            vsag_index->index_->Deserialize(stream);
        }
        return success;
    } catch (const std::exception& e) {
        return make_error(e);
    }
}
}
