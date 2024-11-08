// Copyright 2023 Greptime Team
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

#include "wrapper.h"
#include "vsag/factory.h"
#include "vsag/index.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

template <typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
  out.write((char *)&podRef, sizeof(T));
}

template <typename T> static void readBinaryPOD(std::istream &in, T &podRef) {
  in.read((char *)&podRef, sizeof(T));
}

extern "C" {

CError *new_error(int type_, const char *msg) {
  CError *err = (CError *)malloc(sizeof(CError));
  if (err == NULL) {
    return NULL;
  }

  size_t msg_size = strlen(msg);
  memcpy(err->message, msg,
         msg_size > VSAG_WRAPPER_MAX_ERROR_MESSAGE_LENGTH
             ? VSAG_WRAPPER_MAX_ERROR_MESSAGE_LENGTH
             : msg_size);

  return err;
}

void free_error(const CError *error) {
  if (error) {
    free(const_cast<CError *>(error)); // Deallocate the error struct
  }
}

const CError *create_index(const char *in_index_type, const char *in_parameters,
                           void **out_index_ptr) {
  if (!in_index_type || !in_parameters || !out_index_ptr) {
    return new_error(static_cast<int>(vsag::ErrorType::INVALID_ARGUMENT),
                     "Invalid null argument.");
  }

  auto result = vsag::Factory::CreateIndex(in_index_type, in_parameters);

  if (!result.has_value()) {
    // Convert C++ error to dynamically allocated CError
    return new_error(static_cast<int>(result.error().type),
                     result.error().message.c_str());
  }

  auto pIndex = new std::shared_ptr<vsag::Index>(result.value());
  *out_index_ptr = static_cast<void *>(pIndex);

  return nullptr; // Success: Return NULL
}

const CError *build_index(void *in_index_ptr, size_t in_num_vectors,
                          size_t in_dim, const int64_t *in_ids,
                          const float *in_vectors,

                          const int64_t **out_failed_ids,
                          size_t *out_num_failed) {
  if (!in_index_ptr || !in_ids || !in_vectors || !out_failed_ids ||
      !out_num_failed) {
    return new_error(static_cast<int>(vsag::ErrorType::INVALID_ARGUMENT),
                     "Invalid null argument.");
  }

  // Cast the void pointer back to the original pointer type,
  // std::shared_ptr<Index>*
  auto pIndex = static_cast<std::shared_ptr<vsag::Index> *>(in_index_ptr);

  auto base = vsag::Dataset::Make();
  base->NumElements(in_num_vectors)
      ->Dim(in_dim)
      ->Ids(in_ids)
      ->Float32Vectors(in_vectors)
      ->Owner(false);
  auto result = (*pIndex)->Build(base);

  if (!result.has_value()) {
    // Convert C++ error to dynamically allocated CError
    return new_error(static_cast<int>(result.error().type),
                     result.error().message.c_str());
  }

  // Copy the failed IDs to the output array
  auto failed_ids = result.value();
  auto failed_ids_array = new int64_t[failed_ids.size()];
  std::copy(failed_ids.begin(), failed_ids.end(), failed_ids_array);
  *out_failed_ids = failed_ids_array;
  *out_num_failed = static_cast<int64_t>(failed_ids.size());

  return nullptr; // Success: Return NULL
}

const CError *knn_search_index(void *in_index_ptr, size_t in_dim,
                               const float *in_query_vector, size_t in_k,
                               const char *in_search_parameters,

                               const int64_t **out_ids,
                               const float **out_distances,
                               size_t *out_num_results) {
  if (!in_index_ptr || !in_query_vector || !in_search_parameters || !out_ids ||
      !out_distances || !out_num_results) {
    return new_error(static_cast<int>(vsag::ErrorType::INVALID_ARGUMENT),
                     "Invalid null argument.");
  }

  // Cast the void pointer back to the original pointer type,
  // std::shared_ptr<Index>*
  auto pIndex = static_cast<std::shared_ptr<vsag::Index> *>(in_index_ptr);

  auto query = vsag::Dataset::Make();
  query->NumElements(1)
      ->Dim(in_dim)
      ->Float32Vectors(in_query_vector)
      ->Owner(false);
  auto result = (*pIndex)->KnnSearch(query, in_k, in_search_parameters);

  if (!result.has_value()) {
    // Convert C++ error to dynamically allocated CError
    return new_error(static_cast<int>(result.error().type),
                     result.error().message.c_str());
  }

  auto dataset = result.value();
  auto num = dataset->GetDim();
  *out_num_results = num;

  auto ids_array = new int64_t[num];
  auto ids = dataset->GetIds();
  std::copy(ids, ids + num, ids_array);
  auto distances_array = new float[num];
  auto distances = dataset->GetDistances();
  std::copy(distances, distances + num, distances_array);

  *out_ids = ids_array;
  *out_distances = distances_array;

  return nullptr; // Success: Return NULL
}

const CError *dump_index(void *in_index_ptr, const char *in_file_path) {
  if (!in_index_ptr || !in_file_path) {
    return new_error(static_cast<int>(vsag::ErrorType::INVALID_ARGUMENT),
                     "Invalid null argument.");
  }

  // Cast the void pointer back to the original pointer type,
  // std::shared_ptr<Index>*
  auto pIndex = static_cast<std::shared_ptr<vsag::Index> *>(in_index_ptr);

  if (auto bs = (*pIndex)->Serialize(); bs.has_value()) {
    auto keys = bs->GetKeys();
    std::vector<uint64_t> offsets;

    std::ofstream file(in_file_path, std::ios::binary);
    uint64_t offset = 0;
    for (auto key : keys) {
      // [len][data...][len][data...]...
      vsag::Binary b = bs->Get(key);
      writeBinaryPOD(file, b.size);
      file.write((const char *)b.data.get(), b.size);
      offsets.push_back(offset);
      offset += sizeof(b.size) + b.size;
    }
    // footer
    for (uint64_t i = 0; i < keys.size(); ++i) {
      // [len][key...][offset][len][key...][offset]...
      const auto &key = keys[i];
      int64_t len = key.length();
      writeBinaryPOD(file, len);
      file.write(key.c_str(), key.length());
      writeBinaryPOD(file, offsets[i]);
    }
    // [num_keys][footer_offset]$
    writeBinaryPOD(file, keys.size());
    writeBinaryPOD(file, offset);
    file.close();
  } else {
    auto err = bs.error();
    return new_error(static_cast<int>(err.type), err.message.c_str());
  }

  return nullptr; // Success: Return NULL
}

const CError *load_index(const char *in_file_path, const char *in_index_type,
                         const char *in_parameters,

                         void **out_index_ptr) {
  if (!in_file_path || !in_index_type || !in_parameters || !out_index_ptr) {
    return new_error(static_cast<int>(vsag::ErrorType::INVALID_ARGUMENT),
                     "Invalid null argument.");
  }

  std::ifstream file(in_file_path, std::ios::in);
  file.seekg(-sizeof(uint64_t) * 2, std::ios::end);
  uint64_t num_keys, footer_offset;
  readBinaryPOD(file, num_keys);
  readBinaryPOD(file, footer_offset);
  // std::cout << "num_keys: " << num_keys << std::endl;
  // std::cout << "footer_offset: " << footer_offset << std::endl;
  file.seekg(footer_offset, std::ios::beg);

  std::vector<std::string> keys;
  std::vector<uint64_t> offsets;
  for (uint64_t i = 0; i < num_keys; ++i) {
    int64_t key_len;
    readBinaryPOD(file, key_len);
    // std::cout << "key_len: " << key_len << std::endl;
    char key_buf[key_len + 1];
    memset(key_buf, 0, key_len + 1);
    file.read(key_buf, key_len);
    // std::cout << "key: " << key_buf << std::endl;
    keys.push_back(key_buf);

    uint64_t offset;
    readBinaryPOD(file, offset);
    // std::cout << "offset: " << offset << std::endl;
    offsets.push_back(offset);
  }

  vsag::ReaderSet rs;
  for (uint64_t i = 0; i < num_keys; ++i) {
    int64_t size = 0;
    if (i + 1 == num_keys) {
      size = footer_offset;
    } else {
      size = offsets[i + 1];
    }
    size -= (offsets[i] + sizeof(uint64_t));
    auto file_reader = vsag::Factory::CreateLocalFileReader(
        in_file_path, offsets[i] + sizeof(uint64_t), size);
    rs.Set(keys[i], file_reader);
  }

  std::shared_ptr<vsag::Index> hnsw;
  if (auto index = vsag::Factory::CreateIndex(in_index_type, in_parameters);
      index.has_value()) {
    hnsw = index.value();
  } else {
    auto err = index.error();
    return new_error(static_cast<int>(err.type), err.message.c_str());
  }
  auto res = hnsw->Deserialize(rs);
  if (!res.has_value()) {
    auto err = res.error();
    return new_error(static_cast<int>(err.type), err.message.c_str());
  }

  auto pIndex = new std::shared_ptr<vsag::Index>(hnsw);
  *out_index_ptr = static_cast<void *>(pIndex);

  return nullptr; // Success: Return NULL
}

void free_index(void *index_ptr) {
  if (index_ptr) {
    // Cast the void pointer back to the original pointer type,
    // std::shared_ptr<Index>*
    std::shared_ptr<vsag::Index> *pIndex =
        static_cast<std::shared_ptr<vsag::Index> *>(index_ptr);

    // Delete the std::shared_ptr<Index> which was dynamically allocated
    delete pIndex;

    // Note: Deleting the std::shared_ptr<Index> will automatically handle
    // the decrement of the reference count and will delete the managed Index
    // object if the reference count goes to zero.
  }
}

void free_i64_vector(int64_t *vector) {
  if (vector) {
    delete[] vector;
  }
}
void free_f32_vector(float *vector) {
  if (vector) {
    delete[] vector;
  }
}

} // extern "C"
