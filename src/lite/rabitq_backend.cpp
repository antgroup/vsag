// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <new>

#include "lite/backend.h"
#include "lite/backend_utils.h"
#include "lite/rabitq_snapshot.h"

namespace vsag::lite::detail {
namespace {

template <typename Function>
auto
guarded(Function function, ErrorType fallback = ErrorType::INTERNAL_ERROR) -> decltype(function()) {
    try {
        return function();
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "RaBitQ allocation failed");
    } catch (const std::length_error&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "RaBitQ capacity exceeded");
    } catch (const std::exception&) {
        return failure(fallback, "RaBitQ operation failed");
    } catch (...) {
        return failure(fallback, "RaBitQ operation failed");
    }
}

class RaBitQBackend final : public Backend {
public:
    explicit RaBitQBackend(rabitq::MutableGraphState state) : state_(std::move(state)) {
    }

    tl::expected<void, Error>
    Add(int64_t id, const float* vector, uint64_t dim) override {
        auto valid = validate(vector, dim, Dim());
        if (not valid) {
            return valid;
        }
        if (state_.Contains(id)) {
            return failure(ErrorType::INVALID_ARGUMENT, "duplicate ID");
        }
        if (Size() >= 1000000) {
            return failure(ErrorType::INVALID_ARGUMENT, "RaBitQ record limit exceeded");
        }
        return guarded([&]() -> tl::expected<void, Error> {
            if (not state_.AddTransactional(id, vector)) {
                return failure(ErrorType::INVALID_ARGUMENT, "duplicate ID");
            }
            return {};
        });
    }

    tl::expected<void, Error>
    Update(int64_t id, const float* vector, uint64_t dim) override {
        auto valid = validate(vector, dim, Dim());
        if (not valid) {
            return valid;
        }
        if (not state_.Contains(id)) {
            return failure(ErrorType::INVALID_ARGUMENT, "missing ID");
        }
        return guarded([&]() -> tl::expected<void, Error> {
            auto prepared = rabitq::prepare_encoding(state_.GetModel(), vector);
            if (state_.SameEncoding(id, prepared.code)) {
                return {};
            }
            if (not state_.UpdateTransactional(id, std::move(prepared))) {
                return failure(ErrorType::INVALID_ARGUMENT, "missing ID");
            }
            return {};
        });
    }

    bool
    Remove(int64_t id) override {
        if (not state_.Contains(id)) {
            return false;
        }
        try {
            return state_.RemoveTransactional(id);
        } catch (...) {
            // The existing bool API cannot distinguish missing IDs from an
            // allocation failure. The original state remains intact.
            return false;
        }
    }

    tl::expected<std::vector<Neighbor>, Error>
    Search(const float* query, uint64_t dim, uint64_t k) const override {
        return SearchWithOptions(query, dim, k, {}, {});
    }

    tl::expected<std::vector<Neighbor>, Error>
    Search(const float* query, uint64_t dim, uint64_t k, const IdFilter& filter) const override {
        return SearchWithOptions(query, dim, k, {}, filter);
    }

    tl::expected<std::vector<Neighbor>, Error>
    SearchWithOptions(const float* query,
                      uint64_t dim,
                      uint64_t k,
                      const SearchOptions& options,
                      const IdFilter& filter) const override {
        auto valid = validate(query, dim, Dim());
        if (not valid) {
            return tl::unexpected(valid.error());
        }
        return guarded([&]() -> tl::expected<std::vector<Neighbor>, Error> {
            const auto found = state_.SearchWithOptions(query, k, options.ef_search, filter);
            std::vector<Neighbor> result;
            result.reserve(found.neighbors.size());
            for (const auto& value : found.neighbors) {
                result.push_back({state_.IdAt(value.id), value.distance});
            }
            return result;
        });
    }

    tl::expected<void, Error>
    SaveEncoded(std::ostream& output) const override {
        return guarded(
            [&]() -> tl::expected<void, Error> {
                rabitq::save_mutable_snapshot(output, state_);
                return {};
            },
            ErrorType::READ_ERROR);
    }

    uint64_t
    Size() const override {
        return state_.Size();
    }
    uint64_t
    Dim() const override {
        return state_.GetModel().dim;
    }
    int64_t
    IdAt(uint64_t slot) const override {
        return state_.IdAt(slot);
    }
    const float*
    VectorAt(uint64_t slot, std::vector<float>& scratch) const override {
        return rabitq::decode(state_.GetModel(), state_.GetCodes().At(slot), scratch);
    }
    BackendKind
    Kind() const override {
        return BackendKind::GRAPH;
    }
    VectorStorage
    Storage() const override {
        return VectorStorage::RABITQ8;
    }
    uint64_t
    MaxDegree() const override {
        return state_.GetMaxDegree();
    }
    uint64_t
    EfSearch() const override {
        return state_.GetEfSearch();
    }

    uint64_t
    LinkCountAt(uint64_t slot) const override {
        return state_.LinkCountAt(slot);
    }
    uint64_t
    LinkAt(uint64_t slot, uint64_t edge) const override {
        return state_.LinkAt(slot, edge);
    }

private:
    rabitq::MutableGraphState state_;
};

}  // namespace

tl::expected<std::unique_ptr<Backend>, Error>
make_rabitq_graph_backend(const Backend& source, uint64_t max_degree, uint64_t ef_search) {
    if (source.Size() == 0 or source.Size() > 1000000 or source.Dim() > (1U << 20U) or
        max_degree < 2 or max_degree > 64 or ef_search < max_degree) {
        return failure(ErrorType::INVALID_ARGUMENT, "invalid RaBitQ training or graph options");
    }
    return guarded(
        [&]() -> tl::expected<std::unique_ptr<Backend>, Error> {
            std::vector<float> base;
            std::vector<int64_t> ids;
            if (source.Dim() == 0 or source.Size() > base.max_size() / source.Dim()) {
                return failure(ErrorType::INVALID_ARGUMENT, "RaBitQ training size overflow");
            }
            base.reserve(source.Size() * source.Dim());
            ids.reserve(source.Size());
            std::vector<float> scratch;
            for (uint64_t slot = 0; slot < source.Size(); ++slot) {
                const auto* vector = source.VectorAt(slot, scratch);
                base.insert(base.end(), vector, vector + source.Dim());
                ids.push_back(source.IdAt(slot));
            }
            auto model = rabitq::train(base, source.Size(), source.Dim(), 47);
            rabitq::EncodedRecords codes(source.Dim());
            codes.Reserve(source.Size());
            for (uint64_t slot = 0; slot < source.Size(); ++slot) {
                codes.Append(rabitq::encode(model, base.data() + slot * source.Dim()));
            }
            auto graph = make_graph_backend(source, max_degree, ef_search);
            if (not graph) {
                return tl::unexpected(graph.error());
            }
            rabitq::GraphTopology topology;
            topology.offsets.push_back(0);
            for (uint64_t slot = 0; slot < source.Size(); ++slot) {
                for (uint64_t edge = 0; edge < (*graph)->LinkCountAt(slot); ++edge) {
                    topology.neighbors.push_back((*graph)->LinkAt(slot, edge));
                }
                topology.offsets.push_back(topology.neighbors.size());
            }
            rabitq::MutableGraphState state(std::move(model),
                                            std::move(codes),
                                            topology,
                                            std::move(ids),
                                            max_degree,
                                            ef_search);
            return std::unique_ptr<Backend>(new RaBitQBackend(std::move(state)));
        },
        ErrorType::INVALID_ARGUMENT);
}

tl::expected<std::unique_ptr<Backend>, Error>
load_rabitq_graph_backend(std::istream& input) {
    return guarded(
        [&]() -> tl::expected<std::unique_ptr<Backend>, Error> {
            return std::unique_ptr<Backend>(
                new RaBitQBackend(rabitq::load_mutable_snapshot(input)));
        },
        ErrorType::INVALID_BINARY);
}

}  // namespace vsag::lite::detail
