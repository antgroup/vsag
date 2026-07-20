# Serialization Types

VSAG can persist an index in two shapes: an in-memory [`BinarySet`](#binaryset) (a named collection
of byte blobs) or, for on-disk / streaming scenarios, a [`ReaderSet`](#readerset) of lazy
[`Reader`](#reader) objects. These types are the payloads passed to
[`Index::Serialize` / `Index::Deserialize`](index_class.md#serialization).

For end-to-end workflows and stream-based serialization, see
[Serialization](../advanced/serialization.md) and `examples/cpp/401_persistent_kv.cpp` /
`402_persistent_streaming.cpp`.

## `Binary`

Declared in `vsag/binaryset.h`. A single named byte buffer with its length.

```cpp
struct Binary {
    std::shared_ptr<int8_t[]> data;  // the bytes
    uint64_t size;                   // number of bytes
};
```

The `shared_ptr` owns the buffer, so a `Binary` can be copied and stored freely without worrying
about lifetime.

## `BinarySet`

Declared in `vsag/binaryset.h`. A string-keyed map of `Binary` blobs — the standard in-memory
serialization container. An index serializes itself into several named parts (graph, vectors,
quantizer, etc.), all gathered in one `BinarySet`.

```cpp
class BinarySet {
public:
    void Set(const std::string& name, Binary binary);   // store a blob
    Binary Get(const std::string& name) const;          // {nullptr, 0} if absent
    std::vector<std::string> GetKeys() const;            // all stored names
    bool Contains(const std::string& key) const;
};
```

| Method | Description |
|--------|-------------|
| `Set(name, binary)` | Stores `binary` under `name`, overwriting any existing entry. |
| `Get(name)` | Returns the blob, or an empty `Binary{nullptr, 0}` if the name is absent. |
| `GetKeys()` | Returns every stored name. |
| `Contains(key)` | Whether a blob is stored under `key`. |

```cpp
// Serialize to a BinarySet, then persist each part however you like.
auto serialized = index->Serialize();
if (serialized.has_value()) {
    vsag::BinarySet bs = serialized.value();
    for (const auto& key : bs.GetKeys()) {
        vsag::Binary part = bs.Get(key);
        // write part.data[0 .. part.size) to your store, keyed by `key`
    }
}
```

To restore, rebuild the `BinarySet` from your store and call
[`Deserialize(const BinarySet&)`](index_class.md#serialization) on a fresh (empty) index.

## `Reader`

Declared in `vsag/readerset.h`. An abstract source of bytes that the index reads on demand — the
basis for deserializing large, disk-resident indexes without loading everything into memory. Obtain a
local-file reader from [`Factory::CreateLocalFileReader`](factory_engine.md#createlocalfilereader),
or implement `Reader` for a custom backend (object storage, mmap, etc.). Hold it through `ReaderPtr`
(`std::shared_ptr<Reader>`).

```cpp
class Reader {
public:
    virtual void Read(uint64_t offset, uint64_t len, void* dest) = 0;                  // sync
    virtual void AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack cb) = 0; // async
    virtual bool MultiRead(uint8_t* dests, const uint64_t* lens,
                           const uint64_t* offsets, uint64_t count);                    // batched
    virtual uint64_t Size() const = 0;
};
```

| Method | Description |
|--------|-------------|
| `Read(offset, len, dest)` | Synchronously copy `len` bytes from `offset` into `dest`. Thread-safe. |
| `AsyncRead(offset, len, dest, callback)` | Asynchronous read; `callback` is invoked with an [`IOErrorCode`](#ioerrorcode) and message on completion. |
| `MultiRead(dests, lens, offsets, count)` | Perform `count` synchronous reads in one call; returns `false` on any failure. |
| `Size()` | Total size of the underlying source in bytes. |

### `IOErrorCode`

```cpp
enum class IOErrorCode {
    IO_SUCCESS = 0,  // operation succeeded
    IO_ERROR = 1,    // general I/O error
    IO_TIMEOUT = 2,  // operation timed out
};
```

### `CallBack`

```cpp
using CallBack = std::function<void(IOErrorCode code, const std::string& message)>;
```

The completion handler for `AsyncRead`.

## `ReaderSet`

Declared in `vsag/readerset.h`. A string-keyed map of `Reader` objects — the streaming analogue of
`BinarySet`. Each named part of a serialized index maps to a `Reader` that fetches that part on
demand. Pass a fully populated `ReaderSet` to
[`Deserialize(const ReaderSet&)`](index_class.md#serialization).

```cpp
class ReaderSet {
public:
    void Set(const std::string& name, ReaderPtr reader);
    ReaderPtr Get(const std::string& name) const;   // nullptr if absent
    std::vector<std::string> GetKeys() const;
    bool Contains(const std::string& key) const;
};
```

The method semantics mirror [`BinarySet`](#binaryset), except values are `ReaderPtr` instead of
`Binary`.

```cpp
vsag::ReaderSet readers;
readers.Set("graph", vsag::Factory::CreateLocalFileReader("index.graph", 0, graph_size));
readers.Set("vectors", vsag::Factory::CreateLocalFileReader("index.vectors", 0, vec_size));

auto fresh = vsag::Factory::CreateIndex("hgraph", params).value();
fresh->Deserialize(readers);
```

## See also

- [Index](index_class.md#serialization) — the `Serialize` / `Deserialize` method family.
- [Factory & Engine](factory_engine.md#createlocalfilereader) — creating file-backed readers.
- [Serialization](../advanced/serialization.md) — full persistence guide.
