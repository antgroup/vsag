// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "vsag/lite/persistent_index.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#include "lite/backend_utils.h"

namespace vsag::lite {
namespace {

using detail::failure;

constexpr char CHECKPOINT_FILE[] = "checkpoint.vsag";
constexpr char LOG_FILE[] = "add.log";
// The log starts with a magic, checkpoint size, dimension, checkpoint CRC and
// header CRC. Each little-endian record contains an ID, FP32 inputs and a CRC.
constexpr std::array<uint8_t, 8> MAGIC{'V', 'S', 'A', 'G', 'L', 'A', '0', '1'};
constexpr uint64_t HEADER_BYTES = 32;
constexpr uint64_t ID_BYTES = 8;
constexpr uint64_t CRC_BYTES = 4;
constexpr uint64_t FLOAT_BYTES = 4;
constexpr uint32_t CRC_INITIAL = UINT32_MAX;
static_assert(sizeof(float) == FLOAT_BYTES and std::numeric_limits<float>::is_iec559);

class File {
public:
    explicit File(int fd) : fd_(fd) {
    }
    File(const File&) = delete;
    File&
    operator=(const File&) = delete;
    File(File&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {
    }

    ~File() {
        Close();
    }

    [[nodiscard]] int
    Get() const {
        return fd_;
    }

    void
    Close() {
        if (fd_ >= 0) {
            ::close(std::exchange(fd_, -1));
        }
    }

private:
    int fd_;
};

class NewDirectory {
public:
    NewDirectory(const std::string& path,
                 const std::string& log_path,
                 const std::string& checkpoint_path)
        : path_(path), log_path_(log_path), checkpoint_path_(checkpoint_path) {
    }

    ~NewDirectory() {
        if (keep_) {
            return;
        }
        ::unlink(log_path_.c_str());
        ::unlink(checkpoint_path_.c_str());
        ::rmdir(path_.c_str());
    }

    void
    Keep() {
        keep_ = true;
    }

private:
    const std::string& path_;
    const std::string& log_path_;
    const std::string& checkpoint_path_;
    bool keep_{false};
};

bool
valid_path(const std::string& path) {
    return not path.empty() and path.find('\0') == std::string::npos;
}

bool
lock_file(int fd) {
    int result;
    do {
        result = ::flock(fd, LOCK_EX | LOCK_NB);
    } while (result < 0 and errno == EINTR);
    return result == 0;
}

bool
sync_file(int fd) {
    int result;
    do {
        result = ::fsync(fd);
    } while (result < 0 and errno == EINTR);
    return result == 0;
}

bool
truncate_log(int fd, off_t size) {
    int result;
    do {
        result = ::ftruncate(fd, size);
    } while (result < 0 and errno == EINTR);
    return result == 0 and sync_file(fd);
}

template <typename Transfer>
bool
transfer_all(uint64_t bytes, Transfer transfer) {
    uint64_t offset = 0;
    while (offset < bytes) {
        const auto count = transfer(offset, bytes - offset);
        if (count < 0 and errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        offset += static_cast<uint64_t>(count);
    }
    return true;
}

bool
read_all(int fd, uint8_t* data, uint64_t size) {
    return transfer_all(
        size, [&](uint64_t offset, uint64_t bytes) { return ::read(fd, data + offset, bytes); });
}

bool
write_all(int fd, const uint8_t* data, uint64_t size) {
    return transfer_all(
        size, [&](uint64_t offset, uint64_t bytes) { return ::write(fd, data + offset, bytes); });
}

template <typename UInt>
void
put_le(uint8_t* out, UInt value) {
    for (uint64_t i = 0; i < sizeof(UInt); ++i) {
        out[i] = static_cast<uint8_t>(value >> (8U * i));
    }
}

template <typename UInt>
UInt
get_le(const uint8_t* in) {
    UInt value = 0;
    for (uint64_t i = 0; i < sizeof(UInt); ++i) {
        value |= static_cast<UInt>(in[i]) << (8U * i);
    }
    return value;
}

constexpr auto
make_crc_table() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < table.size(); ++i) {
        uint32_t crc = i;
        for (uint32_t bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1U) ^ ((crc & 1U) == 0 ? 0U : 0xEDB88320U);
        }
        table[i] = crc;
    }
    return table;
}

uint32_t
update_crc(uint32_t crc, const uint8_t* data, uint64_t size) {
    static constexpr auto table = make_crc_table();
    for (uint64_t i = 0; i < size; ++i) {
        crc = (crc >> 8U) ^ table[(crc ^ data[i]) & 0xFFU];
    }
    return crc;
}

uint32_t
checksum(const uint8_t* data, uint64_t size) {
    return ~update_crc(CRC_INITIAL, data, size);
}

struct CheckpointDigest {
    uint64_t size{};
    uint32_t crc{};
};

tl::expected<CheckpointDigest, Error>
digest_checkpoint(std::ifstream& input) {
    std::array<uint8_t, uint64_t{64} * 1024> buffer{};
    uint64_t size = 0;
    uint32_t crc = CRC_INITIAL;
    while (input.read(reinterpret_cast<char*>(buffer.data()), buffer.size()) or
           input.gcount() > 0) {
        const auto bytes = static_cast<uint64_t>(input.gcount());
        size += bytes;
        crc = update_crc(crc, buffer.data(), bytes);
    }
    if (input.bad() or not input.eof()) {
        return failure(ErrorType::READ_ERROR, "cannot read the Lite checkpoint");
    }
    input.clear();
    input.seekg(0);
    if (not input) {
        return failure(ErrorType::READ_ERROR, "cannot rewind the Lite checkpoint");
    }
    return CheckpointDigest{size, ~crc};
}

using Header = std::array<uint8_t, HEADER_BYTES>;

Header
make_header(const CheckpointDigest& digest, uint64_t dim) {
    Header header{};
    std::copy(MAGIC.begin(), MAGIC.end(), header.begin());
    put_le(header.data() + 8, digest.size);
    put_le(header.data() + 16, dim);
    put_le(header.data() + 24, digest.crc);
    put_le(header.data() + 28, checksum(header.data(), 28));
    return header;
}

tl::expected<uint64_t, Error>
record_bytes(uint64_t dim) {
    const uint64_t limit = std::numeric_limits<ssize_t>::max();
    if (dim == 0 or dim > (limit - ID_BYTES - CRC_BYTES) / FLOAT_BYTES) {
        return failure(ErrorType::INVALID_ARGUMENT, "persistent Add dimension is out of range");
    }
    return ID_BYTES + FLOAT_BYTES * dim + CRC_BYTES;
}

void
encode_add(uint8_t* record, int64_t id, const float* vector, uint64_t dim, uint64_t size) {
    uint64_t id_bits;
    std::memcpy(&id_bits, &id, sizeof(id_bits));
    put_le(record, id_bits);
    for (uint64_t i = 0; i < dim; ++i) {
        uint32_t bits;
        std::memcpy(&bits, vector + i, sizeof(bits));
        put_le(record + ID_BYTES + FLOAT_BYTES * i, bits);
    }
    put_le(record + size - CRC_BYTES, checksum(record, size - CRC_BYTES));
}

tl::expected<void, Error>
replay_add(Index& index, const std::vector<uint8_t>& record, std::vector<float>& vector) {
    const auto size = record.size();
    if (get_le<uint32_t>(record.data() + size - CRC_BYTES) !=
        checksum(record.data(), size - CRC_BYTES)) {
        return failure(ErrorType::INVALID_BINARY, "persistent Add record checksum mismatch");
    }
    const auto id_bits = get_le<uint64_t>(record.data());
    int64_t id;
    std::memcpy(&id, &id_bits, sizeof(id));
    for (uint64_t i = 0; i < vector.size(); ++i) {
        const auto bits = get_le<uint32_t>(record.data() + ID_BYTES + FLOAT_BYTES * i);
        std::memcpy(vector.data() + i, &bits, sizeof(bits));
    }
    const auto added = index.Add(id, vector.data(), vector.size());
    if (not added) {
        if (added.error().type == ErrorType::NO_ENOUGH_MEMORY) {
            return tl::unexpected(added.error());
        }
        return failure(ErrorType::INVALID_BINARY, "persistent Add record cannot be applied");
    }
    return {};
}

struct Recovery {
    std::unique_ptr<Index> index;
    std::vector<uint8_t> record;
    off_t committed;
};

using SnapshotLoader = tl::expected<std::unique_ptr<Index>, Error> (*)(std::istream&, uint64_t);

tl::expected<Recovery, Error>
recover(const std::string& directory, int fd, SnapshotLoader load_snapshot) {
    struct stat info{};
    Header header{};
    if (::fstat(fd, &info) != 0 or not S_ISREG(info.st_mode) or
        info.st_size < static_cast<off_t>(header.size()) or
        not read_all(fd, header.data(), header.size())) {
        return failure(ErrorType::READ_ERROR, "cannot read the persistent Add log header");
    }
    if (not std::equal(MAGIC.begin(), MAGIC.end(), header.begin()) or
        get_le<uint32_t>(header.data() + 28) != checksum(header.data(), 28)) {
        return failure(ErrorType::INVALID_BINARY, "invalid persistent Add log header");
    }

    std::ifstream input(directory + "/" + CHECKPOINT_FILE, std::ios::binary);
    if (not input) {
        return failure(ErrorType::MISSING_FILE, "cannot open the Lite checkpoint");
    }
    const auto digest = digest_checkpoint(input);
    if (not digest) {
        return tl::unexpected(digest.error());
    }
    if (digest->size != get_le<uint64_t>(header.data() + 8) or
        digest->crc != get_le<uint32_t>(header.data() + 24)) {
        return failure(ErrorType::INVALID_BINARY, "persistent Add log does not match checkpoint");
    }

    const auto dim = get_le<uint64_t>(header.data() + 16);
    const auto size = record_bytes(dim);
    if (not size) {
        return tl::unexpected(size.error());
    }
    off_t committed = header.size();
    const auto record_size = static_cast<off_t>(*size);
    const uint64_t complete_records = (info.st_size - committed) / record_size;
    // Reserve the final vector payload before loading it. Otherwise the first
    // replayed Add can grow a full buffer and temporarily duplicate the payload.
    auto index = load_snapshot(input, complete_records);
    if (not index) {
        return tl::unexpected(index.error());
    }
    if ((*index)->Dim() != dim) {
        return failure(ErrorType::INVALID_BINARY, "persistent Add log dimension mismatch");
    }
    std::vector<uint8_t> record(*size);
    std::vector<float> vector(dim);
    while (info.st_size - committed >= record_size) {
        if (not read_all(fd, record.data(), record.size())) {
            return failure(ErrorType::READ_ERROR, "cannot read a complete persistent Add record");
        }
        const auto applied = replay_add(**index, record, vector);
        if (not applied) {
            return tl::unexpected(applied.error());
        }
        committed += record_size;
    }
    if (committed != info.st_size and not truncate_log(fd, committed)) {
        return failure(ErrorType::INTERNAL_ERROR, "cannot truncate the incomplete Add log tail");
    }
    return Recovery{std::move(*index), std::move(record), committed};
}

}  // namespace

struct PersistentIndex::Impl {
    File log;
    Recovery state;
};

PersistentIndex::PersistentIndex(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}

PersistentIndex::~PersistentIndex() = default;

tl::expected<std::unique_ptr<PersistentIndex>, Error>
PersistentIndex::Create(const std::string& directory, std::unique_ptr<Index> checkpoint) {
    if (not valid_path(directory) or checkpoint == nullptr) {
        return failure(ErrorType::INVALID_ARGUMENT, "invalid persistent index directory or seed");
    }
    try {
        const auto checkpoint_path = directory + "/" + CHECKPOINT_FILE;
        const auto log_path = directory + "/" + LOG_FILE;
        if (::mkdir(directory.c_str(), 0700) != 0) {
            return failure(ErrorType::INTERNAL_ERROR,
                           "cannot create a new persistent index directory");
        }
        NewDirectory creation(directory, log_path, checkpoint_path);
        File log(::open(log_path.c_str(), O_RDWR | O_APPEND | O_CREAT | O_EXCL | O_CLOEXEC, 0600));
        if (log.Get() < 0 or not lock_file(log.Get())) {
            return failure(ErrorType::INTERNAL_ERROR, "cannot create and lock the Add log");
        }

        std::ofstream output(checkpoint_path, std::ios::binary);
        if (not output) {
            return failure(ErrorType::INTERNAL_ERROR, "cannot create the Lite checkpoint");
        }
        const auto saved = checkpoint->Save(output);
        if (not saved) {
            return tl::unexpected(saved.error());
        }
        output.close();
        File file(::open(checkpoint_path.c_str(), O_RDWR | O_CLOEXEC));
        if (not output or file.Get() < 0 or not sync_file(file.Get())) {
            return failure(ErrorType::INTERNAL_ERROR, "cannot sync the Lite checkpoint");
        }
        const auto dim = checkpoint->Dim();
        checkpoint.reset();
        std::ifstream input(checkpoint_path, std::ios::binary);
        const auto digest = digest_checkpoint(input);
        if (not digest) {
            return tl::unexpected(digest.error());
        }
        // Start with the same reconstructed state used during replay.
        auto index = Index::Load(input);
        if (not index) {
            return tl::unexpected(index.error());
        }
        const auto size = record_bytes(dim);
        if (not size) {
            return tl::unexpected(size.error());
        }
        Recovery state{std::move(*index), std::vector<uint8_t>(*size), HEADER_BYTES};
        const auto header = make_header(*digest, dim);
        File dir(::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
        File parent(dir.Get() < 0 ? -1
                                  : ::openat(dir.Get(), "..", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
        if (parent.Get() < 0 or not write_all(log.Get(), header.data(), header.size()) or
            not sync_file(log.Get()) or not sync_file(dir.Get()) or not sync_file(parent.Get())) {
            return failure(ErrorType::INTERNAL_ERROR, "cannot initialize the persistent Add log");
        }
        auto result = std::unique_ptr<PersistentIndex>(
            new PersistentIndex(std::make_unique<Impl>(Impl{std::move(log), std::move(state)})));
        creation.Keep();
        return result;
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "cannot create persistent Lite index");
    } catch (const std::length_error&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY,
                       "persistent Lite index allocation is too large");
    }
}

tl::expected<std::unique_ptr<PersistentIndex>, Error>
PersistentIndex::Open(const std::string& directory) {
    if (not valid_path(directory)) {
        return failure(ErrorType::INVALID_ARGUMENT, "invalid persistent index directory");
    }
    try {
        const auto log_path = directory + "/" + LOG_FILE;
        File log(::open(log_path.c_str(), O_RDWR | O_APPEND | O_CLOEXEC | O_NOFOLLOW));
        if (log.Get() < 0) {
            return failure(ErrorType::MISSING_FILE, "cannot open the persistent Add log");
        }
        if (not lock_file(log.Get())) {
            return failure(ErrorType::WRONG_STATUS, "persistent Lite directory is already in use");
        }
        auto state = recover(directory, log.Get(), Index::load_with_extra_rows);
        if (not state) {
            return tl::unexpected(state.error());
        }
        return std::unique_ptr<PersistentIndex>(
            new PersistentIndex(std::make_unique<Impl>(Impl{std::move(log), std::move(*state)})));
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "cannot open persistent Lite index");
    } catch (const std::length_error&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY,
                       "persistent Lite index allocation is too large");
    }
}

tl::expected<void, Error>
PersistentIndex::Add(int64_t id, const float* vector, uint64_t dim) {
    auto& state = impl_->state;
    if (state.index == nullptr) {
        return failure(ErrorType::WRONG_STATUS, "reopen the persistent Lite index before use");
    }
    const auto size = static_cast<off_t>(state.record.size());
    if (state.committed > std::numeric_limits<off_t>::max() - size) {
        return failure(ErrorType::INTERNAL_ERROR, "persistent Add log size limit reached");
    }
    const auto added = state.index->Add(id, vector, dim);
    if (not added) {
        return tl::unexpected(added.error());
    }
    encode_add(state.record.data(), id, vector, dim, state.record.size());
    const int fd = impl_->log.Get();
    if (not write_all(fd, state.record.data(), state.record.size()) or not sync_file(fd)) {
        const bool rolled_back = truncate_log(fd, state.committed);
        state.index.reset();
        impl_->log.Close();
        return failure(ErrorType::INTERNAL_ERROR,
                       rolled_back ? "persistent Add failed; reopen the index"
                                   : "persistent Add outcome uncertain; reopen and check the ID");
    }
    state.committed += size;
    return {};
}

tl::expected<uint64_t, Error>
PersistentIndex::Size() const {
    if (impl_->state.index == nullptr) {
        return failure(ErrorType::WRONG_STATUS, "reopen the persistent Lite index before use");
    }
    return impl_->state.index->Size();
}

tl::expected<std::vector<Neighbor>, Error>
PersistentIndex::Search(const float* query,
                        uint64_t dim,
                        uint64_t k,
                        const SearchOptions& options,
                        const IdFilter& filter) const {
    if (impl_->state.index == nullptr) {
        return failure(ErrorType::WRONG_STATUS, "reopen the persistent Lite index before use");
    }
    return impl_->state.index->SearchWithOptions(query, dim, k, options, filter);
}

tl::expected<void, Error>
PersistentIndex::Save(std::ostream& output) const {
    if (impl_->state.index == nullptr) {
        return failure(ErrorType::WRONG_STATUS, "reopen the persistent Lite index before use");
    }
    return impl_->state.index->Save(output);
}

}  // namespace vsag::lite
