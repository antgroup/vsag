# Chunked Serialization

Chunked serialization writes an index body as a sequence of independently readable frames and
records their physical placement in the footer. Because every frame can be located and restored on
its own, the resulting file can be loaded concurrently — see
[Parallel Deserialization](parallel_deserialization.md).

This page describes the chunked form produced by `Serialize(SerializeWriter&)`. It is a variant of
the footer-based format described in [Serialization](serialization.md); the header-first format in
[New Serialization](new_serialization.md) is a separate mechanism and is not interchangeable with
it.

The chunked form is currently implemented for HGraph; other index types return an
unsupported-operation error.

## SerializeWriter

`SerializeWriter` is a byte sink that optionally understands frame boundaries. Implementations
decide the codec: a plain writer stores the body verbatim, while a compressing writer (for example
zstd) emits one independent compressed frame per chunk.

```cpp
class MyWriter : public vsag::SerializeWriter {
public:
    void Write(const char* data, uint64_t size) override { /* append bytes */ }

    // Compression is optional. A writer that does not compress reports an empty
    // name and leaves the frame hooks as no-ops.
    std::string GetCompressorName() const override { return "zstd"; }
    void BeginCompressedFrame() override { /* reset the codec session */ }
    uint64_t EndCompressedFrame() override { /* flush; return bytes written */ }
};
```

Each frame is self-contained: `BeginCompressedFrame` resets the codec session, so a reader can
decompress any single frame without touching its neighbours.

## Writing a Chunked File

```cpp
MyWriter writer(out);
index->Serialize(writer).value();                    // chunk_size defaults to 128 MiB
index->Serialize(writer, 64 * 1024 * 1024).value();  // or pick a frame size
```

The body is split into frames of at most `chunk_size` logical bytes. `chunk_size` is a property of
the written file, not a runtime knob: it is recorded in the footer, and readers always follow the
value stored in the file. Changing it therefore affects new files only and never invalidates
existing ones.

Each component is written with one of two granularities:

- **whole** — the component occupies a single frame. Small components, and components that expose
  no parallel fill hooks, are written this way.
- **chunked** — the component is a head / io-data / tail triple. Only the io data is split into
  frames; the head and tail stay plaintext.

## Footer Layout

The footer carries a `chunked_layout` key that describes the physical placement of every
component:

```json
{
  "chunked_layout": {
    "version": 1,
    "codec": "zstd",
    "chunk_size": 33554432,
    "components": [
      { "name": "label_table", "type": "whole",
        "offset": 0, "csize": 489597, "lsize": 2002832 },
      { "name": "base_codes", "type": "chunked", "io_size": 102760448,
        "head": { "offset": 489597, "size": 20 },
        "chunks": [ { "offset": 489617, "csize": 33555209 } ],
        "tail": { "offset": 102892065, "size": 8229 } }
    ]
  }
}
```

The layout is the single source of truth for readers: they parse the footer from the end of the
data source and drive every read from it, without prior knowledge of the component order. Two
validations run at parse time — the recorded chunks must cover the component extent exactly, and
unknown component types are rejected — so a reader older than a format extension fails fast
instead of misparsing the body.

## Compatibility

| File form | `Deserialize(istream)` | `ParallelDeserialize` |
| --- | --- | --- |
| existing all-in-one, uncompressed | supported | supported through the probe path |
| chunked, uncompressed | supported (same body order as before) | supported through the layout |
| chunked, compressed | rejected with a clear error | supported through the layout |

An uncompressed chunked file stays readable by the sequential path because its body bytes keep the
same order as the existing format and the extra `chunked_layout` key is ignored by older readers. A
compressed chunked body is a sequence of independent frames rather than one stream, so the
sequential reader rejects it instead of misparsing it.

In the other direction, an uncompressed file that carries no layout at all — including the
all-in-one output of `Serialize(ostream)` — can still be loaded concurrently; see
[Loading Files Without a Layout](parallel_deserialization.md#loading-files-without-a-layout).

## Notes

- Frame size trades footer size against load concurrency: smaller frames yield more parallel tasks
  but add roughly one record per frame to the footer.
- The codec name is recorded in the layout; a reader that cannot provide the matching decompressor
  fails with an explicit error.
