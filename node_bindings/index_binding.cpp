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

#include <napi.h>

#include <fstream>
#include <vector>

#include "binding.h"
#include "vsag/dataset.h"
#include "vsag/vsag.h"

namespace {

int64_t
to_int64(uint64_t value) {
    return static_cast<int64_t>(value);
}

class VsagIndex : public Napi::ObjectWrap<VsagIndex> {
public:
    static Napi::Function
    GetClass(Napi::Env env) {
        return DefineClass(
            env,
            "Index",
            {
                InstanceMethod("build", &VsagIndex::Build),
                InstanceMethod("add", &VsagIndex::Add),
                InstanceMethod("remove", &VsagIndex::Remove),
                InstanceMethod("knnSearch", &VsagIndex::KnnSearch),
                InstanceMethod("rangeSearch", &VsagIndex::RangeSearch),
                InstanceMethod("save", &VsagIndex::Save),
                InstanceMethod("load", &VsagIndex::Load),
                InstanceMethod("getNumElements", &VsagIndex::GetNumElements),
                InstanceMethod("getMemoryUsage", &VsagIndex::GetMemoryUsage),
                InstanceMethod("checkIdExist", &VsagIndex::CheckIdExist),
                InstanceMethod("getMinMaxId", &VsagIndex::GetMinMaxId),
                InstanceMethod("calDistanceById", &VsagIndex::CalDistanceById),
            });
    }

    VsagIndex(const Napi::CallbackInfo& info) : Napi::ObjectWrap<VsagIndex>(info) {
        Napi::Env env = info.Env();

        if (info.Length() < 2 || !info[0].IsString() || !info[1].IsString()) {
            Napi::TypeError::New(env, "Expected (name: string, parameters: string)")
                .ThrowAsJavaScriptException();
            return;
        }

        std::string name = info[0].As<Napi::String>().Utf8Value();
        std::string parameters = info[1].As<Napi::String>().Utf8Value();

        auto index = vsag::Factory::CreateIndex(name, parameters);
        if (index.has_value()) {
            index_ = index.value();
        } else {
            const vsag::Error error_code = index.error();
            switch (error_code.type) {
                case vsag::ErrorType::UNSUPPORTED_INDEX:
                    Napi::Error::New(env, "error type: UNSUPPORTED_INDEX")
                        .ThrowAsJavaScriptException();
                    break;
                case vsag::ErrorType::INVALID_ARGUMENT:
                    Napi::Error::New(env, "error type: invalid_parameter")
                        .ThrowAsJavaScriptException();
                    break;
                default:
                    Napi::Error::New(env, "error type: unexpectedError")
                        .ThrowAsJavaScriptException();
                    break;
            }
        }
    }

private:
    Napi::Value
    Build(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        if (info.Length() < 4) {
            Napi::TypeError::New(env,
                                 "Expected (vectors: Float32Array, ids: BigInt64Array, "
                                 "numElements: number, dim: number)")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }

        auto vectors = info[0].As<Napi::Float32Array>();
        auto ids = info[1].As<Napi::BigInt64Array>();
        uint64_t num_elements = info[2].As<Napi::Number>().Int64Value();
        uint64_t dim = info[3].As<Napi::Number>().Int64Value();

        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(to_int64(dim))
            ->NumElements(to_int64(num_elements))
            ->Ids(ids.Data())
            ->Float32Vectors(vectors.Data());

        auto result = index_->Build(dataset);
        if (!result.has_value()) {
            Napi::Error::New(env, "Failed to build index").ThrowAsJavaScriptException();
        }

        return env.Undefined();
    }

    Napi::Value
    Add(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        if (info.Length() < 4) {
            Napi::TypeError::New(env,
                                 "Expected (vectors: Float32Array, ids: BigInt64Array, "
                                 "numElements: number, dim: number)")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }

        auto vectors = info[0].As<Napi::Float32Array>();
        auto ids = info[1].As<Napi::BigInt64Array>();
        uint64_t num_elements = info[2].As<Napi::Number>().Int64Value();
        uint64_t dim = info[3].As<Napi::Number>().Int64Value();

        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(to_int64(dim))
            ->NumElements(to_int64(num_elements))
            ->Ids(ids.Data())
            ->Float32Vectors(vectors.Data());

        auto result = index_->Add(dataset);
        if (!result.has_value()) {
            Napi::Error::New(env, "Failed to add vectors to index").ThrowAsJavaScriptException();
        }

        return env.Undefined();
    }

    Napi::Value
    Remove(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        if (info.Length() < 1) {
            Napi::TypeError::New(env, "Expected (ids: BigInt64Array)")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }

        auto ids = info[0].As<Napi::BigInt64Array>();
        uint64_t count = ids.ElementLength();

        std::vector<int64_t> ids_vec(ids.Data(), ids.Data() + count);
        auto result = index_->Remove(ids_vec);
        if (result.has_value()) {
            return Napi::Number::New(env, result.value());
        }
        return Napi::Number::New(env, 0);
    }

    Napi::Value
    KnnSearch(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        if (info.Length() < 3 || !info[2].IsString()) {
            Napi::TypeError::New(env,
                                 "Expected (vector: Float32Array, k: number, parameters: string)")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }

        auto vector = info[0].As<Napi::Float32Array>();
        uint64_t k = info[1].As<Napi::Number>().Int64Value();
        std::string parameters = info[2].As<Napi::String>().Utf8Value();

        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(to_int64(vector.ElementLength()))
            ->Float32Vectors(vector.Data())
            ->Owner(false);

        auto result_ids = Napi::BigInt64Array::New(env, k);
        auto result_dists = Napi::Float32Array::New(env, k);

        auto result = index_->KnnSearch(query, to_int64(k), parameters);
        if (result.has_value()) {
            const auto* vsag_ids = result.value()->GetIds();
            const auto* vsag_distances = result.value()->GetDistances();
            for (uint64_t i = 0; i < k; ++i) {
                result_ids[i] = vsag_ids[i];
                result_dists[i] = vsag_distances[i];
            }
        }

        auto obj = Napi::Object::New(env);
        obj.Set("ids", result_ids);
        obj.Set("distances", result_dists);
        return obj;
    }

    Napi::Value
    RangeSearch(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        if (info.Length() < 3 || !info[2].IsString()) {
            Napi::TypeError::New(
                env, "Expected (vector: Float32Array, threshold: number, parameters: string)")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }

        auto vector = info[0].As<Napi::Float32Array>();
        float threshold = info[1].As<Napi::Number>().FloatValue();
        std::string parameters = info[2].As<Napi::String>().Utf8Value();

        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(to_int64(vector.ElementLength()))
            ->Float32Vectors(vector.Data())
            ->Owner(false);

        auto result = index_->RangeSearch(query, threshold, parameters);
        if (result.has_value()) {
            const auto* ids = result.value()->GetIds();
            const auto* distances = result.value()->GetDistances();
            const auto count = static_cast<uint64_t>(result.value()->GetDim());

            auto result_ids = Napi::BigInt64Array::New(env, count);
            auto result_dists = Napi::Float32Array::New(env, count);

            for (uint64_t i = 0; i < count; ++i) {
                result_ids[i] = ids[i];
                result_dists[i] = distances[i];
            }

            auto obj = Napi::Object::New(env);
            obj.Set("ids", result_ids);
            obj.Set("distances", result_dists);
            return obj;
        }

        auto obj = Napi::Object::New(env);
        obj.Set("ids", Napi::BigInt64Array::New(env, 0));
        obj.Set("distances", Napi::Float32Array::New(env, 0));
        return obj;
    }

    Napi::Value
    Save(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsString()) {
            Napi::TypeError::New(env, "Expected (filename: string)")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }

        std::string filename = info[0].As<Napi::String>().Utf8Value();
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            Napi::Error::New(env, "Failed to open file for writing").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        index_->Serialize(file);
        file.close();

        return env.Undefined();
    }

    Napi::Value
    Load(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsString()) {
            Napi::TypeError::New(env, "Expected (filename: string)")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }

        std::string filename = info[0].As<Napi::String>().Utf8Value();
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            Napi::Error::New(env, "Failed to open file for reading").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        index_->Deserialize(file);
        file.close();

        return env.Undefined();
    }

    Napi::Value
    GetNumElements(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), index_->GetNumElements());
    }

    Napi::Value
    GetMemoryUsage(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), index_->GetMemoryUsage());
    }

    Napi::Value
    CheckIdExist(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected (id: number)").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        int64_t id = info[0].As<Napi::Number>().Int64Value();
        return Napi::Boolean::New(env, index_->CheckIdExist(id));
    }

    Napi::Value
    GetMinMaxId(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        auto result = index_->GetMinAndMaxId();
        auto obj = Napi::Object::New(env);
        if (result.has_value()) {
            auto [min_id, max_id] = result.value();
            obj.Set("minId", Napi::Number::New(env, min_id));
            obj.Set("maxId", Napi::Number::New(env, max_id));
        } else {
            obj.Set("minId", Napi::Number::New(env, -1));
            obj.Set("maxId", Napi::Number::New(env, -1));
        }
        return obj;
    }

    Napi::Value
    CalDistanceById(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        if (info.Length() < 2) {
            Napi::TypeError::New(env, "Expected (query: Float32Array, ids: BigInt64Array)")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }

        auto query = info[0].As<Napi::Float32Array>();
        auto ids = info[1].As<Napi::BigInt64Array>();
        int64_t count = static_cast<int64_t>(ids.ElementLength());

        auto distances = Napi::Float32Array::New(env, count);
        for (int64_t i = 0; i < count; ++i) {
            distances[i] = -1.0F;
        }

        auto result = index_->CalDistanceById(query.Data(), ids.Data(), count);
        if (result.has_value()) {
            const auto* dist_data = result.value()->GetDistances();
            for (int64_t i = 0; i < count; ++i) {
                distances[i] = dist_data[i];
            }
        }

        return distances;
    }

    std::shared_ptr<vsag::Index> index_;
};

}  // namespace

void
InitIndex(Napi::Env env, Napi::Object exports) {
    exports.Set("Index", VsagIndex::GetClass(env));
}
