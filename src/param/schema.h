
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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common.h"
#include "json_types.h"
#include "param/validators.h"

namespace vsag::param {

// Type-erased descriptor for a single parameter field. A schema is a list of
// these descriptors; the descriptors carry the knowledge of how to parse a
// JSON value into a member, how to serialize a member back to JSON, and how
// to compare the member between two instances.
class FieldBase {
public:
    virtual ~FieldBase() = default;

    virtual const std::string&
    Key() const = 0;

    virtual void
    Parse(void* instance, const JsonType& parent) const = 0;

    virtual void
    Serialize(const void* instance, JsonType& parent) const = 0;

    virtual bool
    Equal(const void* lhs, const void* rhs) const = 0;
};

using FieldPtr = std::shared_ptr<FieldBase>;

// A schema groups every field declared on a parameter class. It is the
// engine driving the auto-generated FromJson / ToJson / CheckCompatibility
// helpers exposed via the ``VSAG_PARAM_SCHEMA`` macro family.
class Schema {
public:
    void
    AddField(FieldPtr field);

    void
    Parse(void* instance, const JsonType& parent) const;

    void
    Serialize(const void* instance, JsonType& parent) const;

    bool
    Equal(const void* lhs, const void* rhs) const;

    // Reject any JSON key that is neither registered as a schema field nor
    // listed in ``extra_known_keys``. Used when a parameter class is fully
    // schema-driven and we want to flag typos from callers.
    void
    CheckUnknownKeys(const JsonType& parent,
                     const std::unordered_set<std::string>& extra_known_keys = {}) const;

    const std::unordered_set<std::string>&
    KnownKeys() const {
        return this->known_keys_;
    }

private:
    std::vector<FieldPtr> fields_;
    std::unordered_set<std::string> known_keys_;
};

// Read a scalar JSON value as ``T``. Specialized for the scalar types the
// schema supports (uint64_t, int64_t, std::string, bool, float).
template <typename T>
T
ReadJsonValue(const JsonType& json);

// Write a scalar ``T`` into ``parent[key]``. Specialized for the same set
// of scalar types as ``ReadJsonValue``.
template <typename T>
void
WriteJsonValueAt(JsonType& parent, const std::string& key, const T& value);

// Field descriptor for a scalar member (``T Owner::*``). Supports an optional
// ``ValidatorFn<T>`` invoked after the JSON value is decoded.
template <typename T, typename Owner>
class ScalarField : public FieldBase {
public:
    ScalarField(std::string key, T Owner::*member, T default_value, ValidatorFn<T> validator)
        : key_(std::move(key)),
          member_(member),
          default_value_(std::move(default_value)),
          validator_(std::move(validator)) {
    }

    const std::string&
    Key() const override {
        return this->key_;
    }

    void
    Parse(void* instance, const JsonType& parent) const override {
        auto* obj = static_cast<Owner*>(instance);
        if (parent.Contains(this->key_)) {
            T value = ReadJsonValue<T>(parent[this->key_]);
            if (this->validator_) {
                this->validator_(this->key_, value);
            }
            obj->*member_ = std::move(value);
        } else {
            obj->*member_ = this->default_value_;
        }
    }

    void
    Serialize(const void* instance, JsonType& parent) const override {
        const auto* obj = static_cast<const Owner*>(instance);
        WriteJsonValueAt<T>(parent, this->key_, obj->*member_);
    }

    bool
    Equal(const void* lhs, const void* rhs) const override {
        const auto* l = static_cast<const Owner*>(lhs);
        const auto* r = static_cast<const Owner*>(rhs);
        return (l->*member_) == (r->*member_);
    }

private:
    std::string key_;
    T Owner::*member_;
    T default_value_;
    ValidatorFn<T> validator_;
};

// Field descriptor for a member holding a ``std::shared_ptr<Sub>`` where
// ``Sub`` derives from ``Parameter``. The user supplies a factory that
// inspects the JSON sub-object and returns the appropriate concrete
// instance (e.g. ``CreateFlattenParam``).
template <typename Sub, typename Owner>
class SubParamField : public FieldBase {
public:
    using SubFactory = std::function<std::shared_ptr<Sub>(const JsonType&)>;
    using MemberPtr = std::shared_ptr<Sub> Owner::*;

    SubParamField(std::string key, MemberPtr member, SubFactory factory, bool required)
        : key_(std::move(key)), member_(member), factory_(std::move(factory)), required_(required) {
    }

    const std::string&
    Key() const override {
        return this->key_;
    }

    void
    Parse(void* instance, const JsonType& parent) const override {
        auto* obj = static_cast<Owner*>(instance);
        if (parent.Contains(this->key_)) {
            obj->*member_ = this->factory_(parent[this->key_]);
        } else {
            CHECK_ARGUMENT(not this->required_,
                           fmt::format("missing required sub-parameter '{}'", this->key_));
            obj->*member_ = nullptr;
        }
    }

    void
    Serialize(const void* instance, JsonType& parent) const override {
        const auto* obj = static_cast<const Owner*>(instance);
        if (obj->*member_) {
            parent[this->key_].SetJson((obj->*member_)->ToJson());
        }
    }

    bool
    Equal(const void* lhs, const void* rhs) const override {
        const auto* l = static_cast<const Owner*>(lhs);
        const auto* r = static_cast<const Owner*>(rhs);
        const bool l_null = ((l->*member_) == nullptr);
        const bool r_null = ((r->*member_) == nullptr);
        if (l_null != r_null) {
            return false;
        }
        if (l_null) {
            return true;
        }
        return (l->*member_)->CheckCompatibility(r->*member_);
    }

private:
    std::string key_;
    MemberPtr member_;
    SubFactory factory_;
    bool required_;
};

// Field descriptor for a constant string tag (typically the ``type`` key
// used to identify a concrete parameter subclass). The constant is always
// emitted by Serialize, and on Parse it is verified to match if present.
class ConstantStringField : public FieldBase {
public:
    ConstantStringField(std::string key, std::string value);

    const std::string&
    Key() const override {
        return this->key_;
    }

    void
    Parse(void* instance, const JsonType& parent) const override;

    void
    Serialize(const void* instance, JsonType& parent) const override;

    bool
    Equal(const void* lhs, const void* rhs) const override {
        return true;
    }

private:
    std::string key_;
    std::string value_;
};

// Fluent builder used by the ``VSAG_PARAM_SCHEMA`` macros to assemble a
// schema declaratively. Owns a ``Schema`` and returns ``*this`` so that
// every ``Scalar`` / ``SubParam`` / ``TypeTag`` call can be chained.
template <typename Owner>
class SchemaBuilder {
public:
    template <typename T>
    SchemaBuilder&
    Scalar(const std::string& key,
           T Owner::*member,
           T default_value,
           ValidatorFn<T> validator = {}) {
        this->schema_.AddField(std::make_shared<ScalarField<T, Owner>>(
            key, member, std::move(default_value), std::move(validator)));
        return *this;
    }

    template <typename Sub>
    SchemaBuilder&
    SubParam(const std::string& key,
             std::shared_ptr<Sub> Owner::*member,
             std::function<std::shared_ptr<Sub>(const JsonType&)> factory,
             bool required = false) {
        this->schema_.AddField(
            std::make_shared<SubParamField<Sub, Owner>>(key, member, std::move(factory), required));
        return *this;
    }

    SchemaBuilder&
    TypeTag(const std::string& key, const std::string& value) {
        this->schema_.AddField(std::make_shared<ConstantStringField>(key, value));
        return *this;
    }

    Schema
    Build() {
        return std::move(this->schema_);
    }

private:
    Schema schema_;
};

}  // namespace vsag::param

// ---------------------------------------------------------------------------
// Macro DSL
//
// The declaration goes inside the class body in the header:
//
//   class BruteForceParameter : public InnerIndexParameter {
//   public:
//       VSAG_PARAM_SCHEMA_DECL();
//       // ... other members ...
//   };
//
// The definition lives in the corresponding .cpp file so that the schema
// can reference helper types (factories, sub-parameter types) that should
// not be transitively imported by every consumer of the header:
//
//   VSAG_PARAM_SCHEMA(BruteForceParameter)
//       VSAG_PARAM_TYPE_TAG(TYPE_KEY, INDEX_TYPE_BRUTE_FORCE)
//       VSAG_PARAM_SUBPARAM_REQUIRED(FlattenInterfaceParameter, base_codes_param,
//                                    BASE_CODES_KEY, CreateFlattenParam)
//   VSAG_PARAM_SCHEMA_END()
//
// The expansion produces a ``const Schema& ClassName::schema()`` definition
// whose function-local static is constructed once on first call.
// ---------------------------------------------------------------------------

#define VSAG_PARAM_SCHEMA_DECL() static const ::vsag::param::Schema& schema()

#define VSAG_PARAM_SCHEMA(ClassName)                   \
    const ::vsag::param::Schema& ClassName::schema() { \
        using SelfType = ClassName;                    \
        static const ::vsag::param::Schema instance = ::vsag::param::SchemaBuilder<ClassName>()

#define VSAG_PARAM(Type, Member, Key, Default, ...) \
    .template Scalar<Type>(Key, &SelfType::Member, static_cast<Type>(Default), ##__VA_ARGS__)

#define VSAG_PARAM_SUBPARAM(SubType, Member, Key, Factory) \
    .template SubParam<SubType>(Key, &SelfType::Member, Factory, false)

#define VSAG_PARAM_SUBPARAM_REQUIRED(SubType, Member, Key, Factory) \
    .template SubParam<SubType>(Key, &SelfType::Member, Factory, true)

#define VSAG_PARAM_TYPE_TAG(Key, Value) .TypeTag(Key, Value)

#define VSAG_PARAM_SCHEMA_END() \
    .Build();                   \
    return instance;            \
    }
