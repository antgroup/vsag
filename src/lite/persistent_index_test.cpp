// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "vsag/lite/persistent_index.h"

#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace vsag::lite {
namespace {

namespace fs = std::filesystem;
constexpr uint64_t DIM = 64;
constexpr int64_t SEED_ROWS = 64;
constexpr uint64_t LOG_HEADER = 32;
using Vector = std::array<float, DIM>;

class TempDirectory {
public:
    TempDirectory() {
        auto pattern = (fs::temp_directory_path() / "vsag-lite-persistence-XXXXXX").string();
        const char* path = ::mkdtemp(pattern.data());
        REQUIRE(path != nullptr);
        path_ = path;
    }

    ~TempDirectory() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    fs::path
    Store() const {
        return path_ / "store";
    }

    fs::path
    Snapshot() const {
        return path_ / "ordinary.vsag";
    }

private:
    fs::path path_;
};

Vector
make_vector(int64_t id) {
    Vector vector{};
    for (uint64_t i = 0; i < DIM; ++i) {
        const auto code = static_cast<uint64_t>(id) * (23 + i * 17) + i * i * 13;
        vector[i] = static_cast<float>(code % 257) / 129.0F - 1.0F;
    }
    return vector;
}

std::unique_ptr<Index>
make_seed(int mode) {
    auto index = Index::Create(DIM);
    REQUIRE(index.has_value());
    if (mode == 0) {
        return std::move(*index);
    }
    for (int64_t i = 0; i < SEED_ROWS; ++i) {
        const auto vector = make_vector(i);
        REQUIRE((*index)->Add(i, vector.data(), DIM).has_value());
    }
    const auto storage = mode == 1   ? VectorStorage::FP32
                         : mode == 2 ? VectorStorage::FP16
                                     : VectorStorage::RABITQ8;
    REQUIRE((*index)->BuildGraph(storage, 8, 64).has_value());
    return std::move(*index);
}

template <typename IndexType>
std::string
snapshot(const IndexType& index) {
    std::ostringstream output(std::ios::binary);
    REQUIRE(index.Save(output).has_value());
    return output.str();
}

void
expect_same(const Index& reference, const PersistentIndex& durable) {
    REQUIRE(durable.Size().value() == reference.Size());
    REQUIRE(snapshot(durable) == snapshot(reference));
    for (const int64_t id : {2, 20, 100, 1002, -987}) {
        const auto query = make_vector(id);
        const auto expected = reference.SearchWithOptions(query.data(), DIM, 10, {128});
        const auto actual = durable.Search(query.data(), DIM, 10, {128});
        REQUIRE(expected.has_value());
        REQUIRE(actual.has_value());
        REQUIRE(actual->size() == expected->size());
        for (uint64_t i = 0; i < actual->size(); ++i) {
            CHECK((*actual)[i].id == (*expected)[i].id);
            CHECK((*actual)[i].distance == (*expected)[i].distance);
        }
    }
}

template <typename Action>
void
kill_after_add(Action action) {
    int ack[2];
    REQUIRE(::pipe(ack) == 0);
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::close(ack[0]);
        const auto handle = action();
        if (not handle) {
            ::_exit(1);
        }
        const char marker = 'A';
        if (::write(ack[1], &marker, 1) != 1) {
            ::_exit(2);
        }
        ::close(ack[1]);
        for (;;) {
            ::pause();
        }
    }
    ::close(ack[1]);
    if (pid < 0) {
        ::close(ack[0]);
        FAIL("cannot fork persistence test");
    }
    char marker{};
    ssize_t read;
    do {
        read = ::read(ack[0], &marker, 1);
    } while (read < 0 and errno == EINTR);
    ::close(ack[0]);
    const int killed = ::kill(pid, SIGKILL);
    int status{};
    REQUIRE(::waitpid(pid, &status, 0) == pid);
    REQUIRE(read == 1);
    REQUIRE(marker == 'A');
    REQUIRE(killed == 0);
    REQUIRE(WIFSIGNALED(status));
    REQUIRE(WTERMSIG(status) == SIGKILL);
}

void
flip_byte(const fs::path& path, uint64_t offset) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekg(offset);
    const int value = file.get();
    REQUIRE(value != std::char_traits<char>::eof());
    file.seekp(offset);
    file.put(static_cast<char>(value ^ 1));
    file.close();
    REQUIRE(file.good());
}

}  // namespace

TEST_CASE("Lite Add survives SIGKILL after acknowledgement when persistence is enabled",
          "[lite-persistent-add]") {
#ifdef VSAG_LITE_HAS_RABITQ_BACKEND
    const int mode = GENERATE(0, 1, 2, 3);
#else
    const int mode = GENERATE(0, 1, 2);
#endif
    CAPTURE(mode);
    TempDirectory temporary;
    const auto path = temporary.Store().string();
    auto seed = make_seed(mode);
    std::istringstream input(snapshot(*seed), std::ios::binary);
    auto reference = Index::Load(input);
    REQUIRE(reference.has_value());
    auto durable = PersistentIndex::Create(path, std::move(seed));
    REQUIRE(durable.has_value());
    REQUIRE_FALSE(PersistentIndex::Create(path, make_seed(0)).has_value());
    REQUIRE_FALSE(PersistentIndex::Open(path).has_value());

    const auto first = make_vector(-987);
    REQUIRE((*durable)->Add(-987, first.data(), DIM).has_value());
    REQUIRE((*reference)->Add(-987, first.data(), DIM).has_value());
    REQUIRE_FALSE((*durable)->Add(-987, first.data(), DIM).has_value());
    REQUIRE_FALSE((*durable)->Add(999, first.data(), DIM - 1).has_value());
    REQUIRE_FALSE((*durable)->Add(999, nullptr, DIM).has_value());
    expect_same(**reference, **durable);
    durable->reset();

    const auto second = make_vector(1002);
    kill_after_add([&]() -> std::unique_ptr<PersistentIndex> {
        auto child = PersistentIndex::Open(path);
        if (not child or not(*child)->Add(1002, second.data(), DIM)) {
            return nullptr;
        }
        return std::move(*child);
    });

    durable = PersistentIndex::Open(path);
    REQUIRE(durable.has_value());
    REQUIRE((*reference)->Add(1002, second.data(), DIM).has_value());
    expect_same(**reference, **durable);

    const auto third = make_vector(100);
    REQUIRE((*durable)->Add(100, third.data(), DIM).has_value());
    REQUIRE((*reference)->Add(100, third.data(), DIM).has_value());
    durable->reset();
    durable = PersistentIndex::Open(path);
    REQUIRE(durable.has_value());
    expect_same(**reference, **durable);
}

TEST_CASE("An ordinary Lite snapshot retains its last explicit save", "[lite-persistent-add]") {
    TempDirectory temporary;
    const auto file = temporary.Snapshot();
    {
        auto index = make_seed(0);
        std::ofstream output(file, std::ios::binary);
        REQUIRE(index->Save(output).has_value());
    }
    kill_after_add([&]() -> std::unique_ptr<Index> {
        std::ifstream input(file, std::ios::binary);
        auto child = Index::Load(input);
        const auto vector = make_vector(100);
        if (not child or not(*child)->Add(100, vector.data(), DIM) or (*child)->Size() != 1) {
            return nullptr;
        }
        return std::move(*child);
    });
    std::ifstream input(file, std::ios::binary);
    auto reopened = Index::Load(input);
    REQUIRE(reopened.has_value());
    CHECK((*reopened)->Size() == 0);
}

TEST_CASE("Lite recovery handles an incomplete tail and reports corrupt records",
          "[lite-persistent-add]") {
    TempDirectory temporary;
    const auto path = temporary.Store().string();
    const auto log = temporary.Store() / "add.log";
    const auto vector = make_vector(10);
    auto index = PersistentIndex::Create(path, make_seed(0));
    REQUIRE(index.has_value());
    REQUIRE((*index)->Add(10, vector.data(), DIM).has_value());
    index->reset();
    const auto complete_size = fs::file_size(log);
    REQUIRE(complete_size == LOG_HEADER + 12 + DIM * sizeof(float));

    SECTION("truncate incomplete suffix and accept the next append") {
        {
            std::ofstream output(log, std::ios::binary | std::ios::app);
            output.write("partial", 7);
        }
        auto reopened = PersistentIndex::Open(path);
        REQUIRE(reopened.has_value());
        CHECK((*reopened)->Size().value() == 1);
        CHECK(fs::file_size(log) == complete_size);
        REQUIRE((*reopened)->Add(11, vector.data(), DIM).has_value());
        reopened->reset();
        reopened = PersistentIndex::Open(path);
        REQUIRE(reopened.has_value());
        CHECK((*reopened)->Size().value() == 2);
    }

    SECTION("keep a complete corrupted record for diagnosis") {
        flip_byte(log, LOG_HEADER + 8);
        const auto reopened = PersistentIndex::Open(path);
        REQUIRE_FALSE(reopened.has_value());
        CHECK(reopened.error().type == ErrorType::INVALID_BINARY);
        CHECK(fs::file_size(log) == complete_size);
    }

    SECTION("detect a mismatched checkpoint") {
        flip_byte(temporary.Store() / "checkpoint.vsag", 0);
        const auto reopened = PersistentIndex::Open(path);
        REQUIRE_FALSE(reopened.has_value());
        CHECK(reopened.error().type == ErrorType::INVALID_BINARY);
        CHECK(fs::file_size(log) == complete_size);
    }
}

TEST_CASE("Lite rejects an incomplete persistent directory", "[lite-persistent-add]") {
    TempDirectory temporary;
    const auto path = temporary.Store().string();
    const auto log = temporary.Store() / "add.log";

    SECTION("invalid arguments and an absent directory") {
        const auto empty_path = PersistentIndex::Open("");
        REQUIRE_FALSE(empty_path.has_value());
        CHECK(empty_path.error().type == ErrorType::INVALID_ARGUMENT);
        const auto absent = PersistentIndex::Open(path);
        REQUIRE_FALSE(absent.has_value());
        CHECK(absent.error().type == ErrorType::MISSING_FILE);
        REQUIRE_FALSE(PersistentIndex::Create(path, nullptr).has_value());
        CHECK_FALSE(fs::exists(path));
    }

    SECTION("a truncated header stays available for diagnosis") {
        REQUIRE(fs::create_directory(path));
        {
            std::ofstream output(log, std::ios::binary);
            output.write("VSAG", 4);
        }
        const auto result = PersistentIndex::Open(path);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().type == ErrorType::READ_ERROR);
        CHECK(fs::file_size(log) == 4);
    }

    SECTION("a corrupted header is rejected before replay") {
        auto index = PersistentIndex::Create(path, make_seed(0));
        REQUIRE(index.has_value());
        index->reset();
        flip_byte(log, 0);
        const auto result = PersistentIndex::Open(path);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().type == ErrorType::INVALID_BINARY);
    }

    SECTION("a missing checkpoint is reported without trimming the log") {
        auto index = PersistentIndex::Create(path, make_seed(0));
        REQUIRE(index.has_value());
        index->reset();
        REQUIRE(fs::remove(temporary.Store() / "checkpoint.vsag"));
        const auto result = PersistentIndex::Open(path);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().type == ErrorType::MISSING_FILE);
        CHECK(fs::file_size(log) == LOG_HEADER);
    }
}

TEST_CASE("Lite cleans up after a failed initial checkpoint write", "[lite-persistent-add]") {
    TempDirectory temporary;
    const auto path = temporary.Store().string();
    auto seed = make_seed(0);
    const auto pid = ::fork();
    if (pid == 0) {
        struct rlimit limit{};
        if (::getrlimit(RLIMIT_FSIZE, &limit) != 0 or limit.rlim_max < 8) {
            ::_exit(1);
        }
        limit.rlim_cur = 8;
        if (::signal(SIGXFSZ, SIG_IGN) == SIG_ERR or ::setrlimit(RLIMIT_FSIZE, &limit) != 0) {
            ::_exit(2);
        }
        const auto failed = PersistentIndex::Create(path, std::move(seed));
        ::_exit(not failed and not fs::exists(path) ? 0 : 3);
    }
    REQUIRE(pid > 0);
    int status{};
    REQUIRE(::waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
    CHECK_FALSE(fs::exists(path));
    REQUIRE(PersistentIndex::Create(path, std::move(seed)).has_value());
}

TEST_CASE("A short Add write rolls back the log and closes the writer", "[lite-persistent-add]") {
    TempDirectory temporary;
    const auto path = temporary.Store().string();
    const auto log = temporary.Store() / "add.log";
    const auto vector = make_vector(10);
    auto index = PersistentIndex::Create(path, make_seed(1));
    REQUIRE(index.has_value());
    REQUIRE((*index)->Add(100, vector.data(), DIM).has_value());
    index->reset();
    const auto complete_size = fs::file_size(log);

    const auto pid = ::fork();
    if (pid == 0) {
        auto opened = PersistentIndex::Open(path);
        if (not opened) {
            ::_exit(1);
        }
        struct rlimit limit{};
        if (::getrlimit(RLIMIT_FSIZE, &limit) != 0 or complete_size + 3 > limit.rlim_max) {
            ::_exit(2);
        }
        limit.rlim_cur = complete_size + 3;
        if (::signal(SIGXFSZ, SIG_IGN) == SIG_ERR or ::setrlimit(RLIMIT_FSIZE, &limit) != 0) {
            ::_exit(3);
        }
        const auto failed = (*opened)->Add(101, vector.data(), DIM);
        const auto invalidated = (*opened)->Size();
        if (failed or failed.error().type != ErrorType::INTERNAL_ERROR or invalidated or
            invalidated.error().type != ErrorType::WRONG_STATUS) {
            ::_exit(4);
        }
        const auto restored = PersistentIndex::Open(path);
        ::_exit(restored and (*restored)->Size().value() == SEED_ROWS + 1 ? 0 : 5);
    }
    REQUIRE(pid > 0);
    int status{};
    REQUIRE(::waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
    CHECK(fs::file_size(log) == complete_size);

    index = PersistentIndex::Open(path);
    REQUIRE(index.has_value());
    REQUIRE((*index)->Size().value() == SEED_ROWS + 1);
    REQUIRE((*index)->Add(101, vector.data(), DIM).has_value());
}

}  // namespace vsag::lite
