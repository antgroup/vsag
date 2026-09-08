#include "label_table.h"

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "impl/allocator/default_allocator.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag_exception.h"

using namespace vsag;

namespace {

using DuplicateGroup = std::pair<InnerIdType, std::vector<InnerIdType>>;

class ForwardOnlyStreamReader final : public StreamReader {
public:
    explicit ForwardOnlyStreamReader(std::string data) : data_(std::move(data)) {
    }

    void
    Read(char* data, uint64_t size) override {
        if (cursor_ > data_.size() or size > data_.size() - cursor_) {
            throw std::runtime_error("read exceeds stream boundary");
        }
        std::memcpy(data, data_.data() + cursor_, size);
        cursor_ += size;
    }

    void
    Seek(uint64_t cursor) override {
        (void)cursor;
        throw std::runtime_error("seek is not supported");
    }

    [[nodiscard]] uint64_t
    GetCursor() const override {
        return cursor_;
    }

    [[nodiscard]] uint64_t
    Length() override {
        throw std::runtime_error("length is not supported");
    }

private:
    std::string data_;
    uint64_t cursor_{0};
};

std::stringstream
create_serialized_label_table(const std::vector<LabelType>& labels,
                              const std::vector<DuplicateGroup>& duplicate_groups) {
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    IOStreamWriter writer(stream);
    StreamWriter::WriteVector(writer, labels);
    const uint64_t duplicate_count = duplicate_groups.size();
    StreamWriter::WriteObj(writer, duplicate_count);
    for (const auto& [head_id, duplicate_ids] : duplicate_groups) {
        StreamWriter::WriteObj(writer, head_id);
        StreamWriter::WriteVector(writer, duplicate_ids);
    }
    stream.seekg(0);
    return stream;
}

std::stringstream
create_serialized_duplicate_records(const std::vector<DuplicateGroup>& duplicate_groups) {
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    IOStreamWriter writer(stream);
    const uint64_t duplicate_count = duplicate_groups.size();
    StreamWriter::WriteObj(writer, duplicate_count);
    for (const auto& [head_id, duplicate_ids] : duplicate_groups) {
        StreamWriter::WriteObj(writer, head_id);
        StreamWriter::WriteVector(writer, duplicate_ids);
    }
    stream.seekg(0);
    return stream;
}

void
deserialize_label_table(LabelTable& label_table, std::stringstream& stream) {
    IOStreamReader reader(stream);
    label_table.Deserialize(reader);
}

void
deserialize_duplicate_records(LabelTable& label_table,
                              std::stringstream& stream,
                              uint64_t logical_element_count) {
    IOStreamReader reader(stream);
    label_table.DeserializeDuplicateRecords(reader, logical_element_count);
}

}  // namespace

TEST_CASE("LabelTable Supports Configurable Remap Implementation", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("robin remap works") {
        LabelTable label_table(allocator.get(), true, false, LabelRemapType::ROBIN);
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        REQUIRE(label_table.GetRemapSize() == 2);
        REQUIRE(label_table.GetIdByLabel(100) == 0);
        REQUIRE(label_table.GetIdByLabel(200) == 1);
    }

    SECTION("pg remap remains default") {
        LabelTable label_table(allocator.get(), true, false, LabelRemapType::PG);
        label_table.Insert(0, 100);

        REQUIRE(label_table.GetRemapSize() == 1);
        REQUIRE(label_table.GetIdByLabel(100) == 0);
    }
}

TEST_CASE("LabelTable deserializes duplicate groups", "[ut][LabelTable][duplicate]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get(), true, true);
    auto stream = create_serialized_label_table({10, 11, 12, 13, 14}, {{0, {1, 2}}, {3, {4}}});

    deserialize_label_table(label_table, stream);

    const auto first_group = label_table.GetDuplicateId(0);
    REQUIRE(first_group.size() == 2);
    REQUIRE(first_group.contains(1));
    REQUIRE(first_group.contains(2));
    const auto second_group = label_table.GetDuplicateId(3);
    REQUIRE(second_group.size() == 1);
    REQUIRE(second_group.contains(4));
    REQUIRE(label_table.GetDuplicateId(1).empty());
    REQUIRE(label_table.GetDuplicateId(2).empty());
    REQUIRE(label_table.GetDuplicateId(4).empty());
}

TEST_CASE("LabelTable reads duplicate groups sequentially", "[ut][LabelTable][duplicate]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable source(allocator.get(), true, true);
    for (InnerIdType id = 0; id < 5; ++id) {
        source.Insert(id, 10 + id);
    }
    source.Resize(5);
    source.SetDuplicateId(0, 1);
    source.SetDuplicateId(0, 2);
    source.SetDuplicateId(3, 4);

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    IOStreamWriter writer(stream);
    source.Serialize(writer);

    ForwardOnlyStreamReader reader(stream.str());
    LabelTable restored(allocator.get(), true, true);
    REQUIRE_NOTHROW(restored.Deserialize(reader));
    REQUIRE(reader.GetCursor() == stream.str().size());

    const auto first_group = restored.GetDuplicateId(0);
    REQUIRE(first_group.size() == 2);
    REQUIRE(first_group.contains(1));
    REQUIRE(first_group.contains(2));
    const auto second_group = restored.GetDuplicateId(3);
    REQUIRE(second_group.size() == 1);
    REQUIRE(second_group.contains(4));
}

TEST_CASE("LabelTable rejects out-of-range duplicate members",
          "[ut][LabelTable][duplicate][invalid-member]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get(), true, true);
    auto stream = create_serialized_label_table({10, 11, 12}, {{0, {3}}});

    REQUIRE_THROWS_AS(deserialize_label_table(label_table, stream), VsagException);
}

TEST_CASE("LabelTable rejects out-of-range duplicate heads",
          "[ut][LabelTable][duplicate][invalid-head]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get(), true, true);
    auto stream = create_serialized_label_table({10, 11, 12}, {{3, {1}}});

    REQUIRE_THROWS_AS(deserialize_label_table(label_table, stream), VsagException);
}

TEST_CASE("LabelTable rejects overlapping duplicate groups",
          "[ut][LabelTable][duplicate][overlap]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    const auto require_invalid = [&](const std::vector<DuplicateGroup>& groups) {
        LabelTable label_table(allocator.get(), true, true);
        auto stream = create_serialized_label_table({10, 11, 12, 13, 14, 15}, groups);
        REQUIRE_THROWS_AS(deserialize_label_table(label_table, stream), VsagException);
    };

    SECTION("head is repeated") {
        require_invalid({{0, {1}}, {0, {2}}});
    }

    SECTION("member is repeated in one group") {
        require_invalid({{0, {1, 1}}});
    }

    SECTION("member is shared by groups") {
        require_invalid({{0, {1}}, {2, {1}}});
    }

    SECTION("member later becomes a head") {
        require_invalid({{0, {1}}, {1, {2}}});
    }

    SECTION("head later becomes a member") {
        require_invalid({{0, {1}}, {2, {0}}});
    }

    SECTION("head contains itself") {
        require_invalid({{0, {0}}});
    }
}

TEST_CASE("LabelTable validates duplicate records against logical count",
          "[ut][LabelTable][duplicate][logical-count]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get(), true, true);
    for (InnerIdType id = 0; id < 3; ++id) {
        label_table.Insert(id, 10 + id);
    }
    label_table.Resize(1024);

    SECTION("head within capacity but outside logical elements") {
        auto stream = create_serialized_duplicate_records({{3, {1}}});
        REQUIRE_THROWS_AS(deserialize_duplicate_records(label_table, stream, 3), VsagException);
    }

    SECTION("member within capacity but outside logical elements") {
        auto stream = create_serialized_duplicate_records({{0, {3}}});
        REQUIRE_THROWS_AS(deserialize_duplicate_records(label_table, stream, 3), VsagException);
    }
}

TEST_CASE("LabelTable validates duplicate record counts before allocation",
          "[ut][LabelTable][duplicate][invalid-count]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get(), true, true);
    for (InnerIdType id = 0; id < 5; ++id) {
        label_table.Insert(id, 10 + id);
    }
    label_table.Resize(1024);

    SECTION("group count") {
        auto stream = create_serialized_duplicate_records({{0, {1}}, {2, {3}}, {4, {}}});
        REQUIRE_THROWS_AS(deserialize_duplicate_records(label_table, stream, 5), VsagException);
    }

    SECTION("empty group") {
        auto stream = create_serialized_duplicate_records({{0, {}}});
        REQUIRE_THROWS_AS(deserialize_duplicate_records(label_table, stream, 5), VsagException);
    }

    SECTION("member count exceeds remaining logical elements") {
        auto stream = create_serialized_duplicate_records({{0, {1, 2}}, {3, {4, 4}}});
        REQUIRE_THROWS_AS(deserialize_duplicate_records(label_table, stream, 5), VsagException);
    }
}

TEST_CASE("LabelTable keeps duplicate records unchanged after invalid input",
          "[ut][LabelTable][duplicate][atomic-publish]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get(), true, true);
    for (InnerIdType id = 0; id < 4; ++id) {
        label_table.Insert(id, 10 + id);
    }
    label_table.Resize(1024);
    label_table.SetDuplicateId(0, 1);

    auto stream = create_serialized_duplicate_records({{2, {4}}});
    REQUIRE_THROWS_AS(deserialize_duplicate_records(label_table, stream, 4), VsagException);

    REQUIRE(label_table.duplicate_count_ == 1);
    const auto duplicate_ids = label_table.GetDuplicateId(0);
    REQUIRE(duplicate_ids.size() == 1);
    REQUIRE(duplicate_ids.contains(1));
}
