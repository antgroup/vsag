#pragma once

#include <fstream>
#include <iostream>


namespace vsag {

template <typename T>
inline void load_aligned_fvecs(const std::string &fvecs_file, T *&data, uint32_t &npts, uint32_t &dim)
{
    int datasize = sizeof(T);

    std::ifstream reader(fvecs_file, std::ios::binary | std::ios::ate);
    size_t fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);

    reader.read((char *)&dim, sizeof(dim));
    reader.seekg(0, std::ios::beg);

    npts = fsize / (dim * datasize + sizeof(uint32_t));
    std::cout << fsize << " " << dim << " " << datasize << std::endl;
    std::cout << "Dataset: #pts = " << npts << ", # dims = " << dim << std::endl;

    uint32_t allocSize = npts * dim * datasize;
    std::cout << "allocating aligned memory of " << allocSize << " bytes" << std::endl;
    data = new T[npts * dim];
    T *read_buf = new T[npts * (dim + 1)];
    reader.read((char *)read_buf, npts * (dim + 1) * datasize);

    for (size_t i = 0; i < npts; i++)
    {
        memcpy(data + i * dim, read_buf + i * (dim + 1) + 1, dim * datasize);
    }

    delete [] read_buf;
    reader.close();
}


template <typename T>
inline void load_aligned_bin(const std::string &bin_file, T *&data, uint32_t &npts, uint32_t &dim)
{
    std::ifstream reader;
    reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    reader.open(bin_file, std::ios::binary | std::ios::ate);
    size_t actual_file_size = reader.tellg();
    std::cout << "==== Reading (with alignment) bin file: " << bin_file << " ====" << std::endl;
    std::cout << "actual file size: " << actual_file_size << std::endl;

    reader.seekg(0);
    reader.read((char *)&npts, sizeof(uint32_t));
    reader.read((char *)&dim, sizeof(uint32_t));

    size_t expected_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    assert (actual_file_size == expected_file_size);

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim  << std::endl;
    uint32_t allocSize = npts * dim * sizeof(T);
    std::cout << "allocating aligned memory of " << allocSize << " bytes" << std::endl;

    data = new T[npts * dim];
    reader.read((char *)(data), npts * dim * sizeof(T));
}

template <typename T>
inline void load_vector(const std::string &bin_file, T *&data, uint32_t dim, uint32_t id, uint32_t offset = 2 * sizeof(int)) {
    std::ifstream reader;
    reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    
    reader.open(bin_file, std::ios::binary | std::ios::ate);
    size_t actual_file_size = reader.tellg();
    assert (actual_file_size > offset + (id + 1) * dim * sizeof(T));

    reader.seekg(offset + id * dim * sizeof(T));
    reader.read((char *)(data), dim * sizeof(T));
}

template <typename T>
inline void save_aligned_bin(const std::string &bin_file, T *data, uint32_t npts, uint32_t dim) {
    std::ofstream writer(bin_file, std::ios::binary);
    writer.write(reinterpret_cast<const char*>(&npts), sizeof(uint32_t));
    writer.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
    writer.write(reinterpret_cast<const char*>(data), sizeof(T) * dim * npts);
    writer.close();
}

inline void load_truthset(const std::string &bin_file, uint32_t *&ids, float *&dists, uint32_t &npts, uint32_t &dim)
{
    std::ifstream reader;
    reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    reader.open(bin_file, std::ios::binary | std::ios::ate);
    size_t actual_file_size = reader.tellg();

    std::cout << "==== Reading truthset file :" << bin_file.c_str() << " ====" << std::endl;
    std::cout << "actual file size: " << actual_file_size << std::endl;

    reader.seekg(0);
    reader.read((char *)&npts, sizeof(uint32_t));
    reader.read((char *)&dim, sizeof(uint32_t));

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << std::endl;

    int truthset_type = -1; // 1 means truthset has ids and distances, 2 means only ids, -1 is error

    size_t expected_file_size_with_dists = 2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists)
        truthset_type = 1;

    size_t expected_file_size_just_ids = npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_just_ids)
        truthset_type = 2;

    if (truthset_type == -1)
    {
        throw std::runtime_error("read gt failed");
    }

    ids = new uint32_t[npts * dim];
    reader.read((char *)ids, npts * dim * sizeof(uint32_t));

    if (truthset_type == 1)
    {
        dists = new float[npts * dim];
        reader.read((char *)dists, npts * dim * sizeof(float));
    }
}


template <typename T>
static void
writeBinaryPOD(std::ostream& out, const T& podRef) {
    out.write((char*)&podRef, sizeof(T));
}

template <typename T>
static void
readBinaryPOD(std::istream& in, T& podRef) {
    in.read((char*)&podRef, sizeof(T));
}

void serialize(std::shared_ptr<vsag::Index> index, std::string index_path) {
    std::fstream file(index_path, std::ios::out | std::ios::binary);
    if (auto bs = index->Serialize(); bs.has_value()) {
        auto keys = bs->GetKeys();
        std::vector<uint64_t> offsets;

        uint64_t offset = 0;
        for (auto key : keys) {
            // [len][data...][len][data...]...
            vsag::Binary b = bs->Get(key);
            writeBinaryPOD(file, b.size);
            file.write((const char*)b.data.get(), b.size);
            offsets.push_back(offset);
            offset += sizeof(b.size) + b.size;
        }
        // footer
        for (uint64_t i = 0; i < keys.size(); ++i) {
            // [len][key...][offset][len][key...][offset]...
            const auto& key = keys[i];
            int64_t len = key.length();
            writeBinaryPOD(file, len);
            file.write(key.c_str(), key.length());
            writeBinaryPOD(file, offsets[i]);
        }
        // [num_keys][footer_offset]$
        writeBinaryPOD(file, keys.size());
        writeBinaryPOD(file, offset);
        file.close();
    } else if (bs.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY) {
        std::cerr << "no enough memory to serialize index" << std::endl;
    }
}

void deserialize(std::shared_ptr<vsag::Index> index, std::string index_path) {
    std::ifstream file(index_path, std::ios::in | std::ios::binary);
    file.seekg(-sizeof(uint64_t) * 2, std::ios::end);
    uint64_t num_keys, footer_offset;
    readBinaryPOD(file, num_keys);
    readBinaryPOD(file, footer_offset);
    file.seekg(footer_offset, std::ios::beg);

    std::vector<std::string> keys;  // todo: mem peak here
    std::vector<uint64_t> offsets;
    for (uint64_t i = 0; i < num_keys; ++i) {
        int64_t key_len;
        readBinaryPOD(file, key_len);
        char key_buf[key_len + 1];
        memset(key_buf, 0, key_len + 1);
        file.read(key_buf, key_len);
        keys.push_back(key_buf);

        uint64_t offset;
        readBinaryPOD(file, offset);
        offsets.push_back(offset);
    }

    vsag::BinarySet bs;
    for (uint64_t i = 0; i < num_keys; ++i) {
        file.seekg(offsets[i], std::ios::beg);
        vsag::Binary b;
        readBinaryPOD(file, b.size);
        b.data.reset(new int8_t[b.size]);
        file.read((char*)b.data.get(), b.size);
        bs.Set(keys[i], b);
    }
    file.close();

    index->Deserialize(bs);
}




}