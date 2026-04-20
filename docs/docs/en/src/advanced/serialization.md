# Serialization

VSAG indexes can be serialized and deserialized through several interfaces, supporting
persistence, cross-process sharing, and distributed deployment.

## Three Interfaces

### 1. `BinarySet` / `ReaderSet`

The most flexible option. The index is split into named binary segments, and the caller owns the
storage medium (object store, KV, sharded uploads, etc.).

```cpp
// Save
vsag::BinarySet bs = index->Serialize().value();
for (const auto& key : bs.GetKeys()) {
    auto binary = bs.Get(key);
    // Write to storage
}

// Load
vsag::BinarySet bs_loaded;
// Populate bs_loaded by reading each key from storage.
auto empty = vsag::Factory::CreateIndex("hnsw", build_params).value();
empty->Deserialize(bs_loaded);
```

`ReaderSet` is similar to `BinarySet` but uses a user-supplied `Reader` to read on demand, which
avoids loading everything at once. This is useful for memory-constrained or partial-deserialization
scenarios (for example, the on-disk portion of DiskANN).

### 2. File Streams (`std::ostream` / `std::istream`)

The simplest option — serialize the whole index to a file or memory stream:

```cpp
std::ofstream out("index.bin", std::ios::binary);
index->Serialize(out);

std::ifstream in("index.bin", std::ios::binary);
empty->Deserialize(in);
```

### 3. Custom Write Function (`WriteFuncType`)

For streaming or chunked backends, supply a write callback:

```cpp
index->Serialize([&](const void* buf, uint64_t offset, uint64_t size) {
    // Write [buf, buf+size) at offset
});
```

## Notes

- `Deserialize` requires an **empty** target index whose configuration (`dim`, `metric_type`, etc.)
  matches the one used at serialization time.
- When upgrading across major versions, check the compatibility notes in the
  [release notes](../resources/release_notes.md).
- DiskANN's disk files are managed independently; `Serialize` returns the in-memory metadata side.
- References:
  `examples/cpp/318_feature_tune.cpp`, `examples/cpp/401_persistent_kv.cpp`,
  `examples/cpp/402_persistent_streaming.cpp`.
