# VSAG TypeScript / Node.js Examples

Small Node.js / TypeScript programs that exercise `vsag`, VSAG's
official Node.js binding.

> **Note.** The TypeScript binding is still early — the package version
> on npm trails the C++ library, and not every C++ feature is wrapped
> yet. Treat these examples as a starting point and expect the surface
> to grow over time. Contributions welcome — see
> [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Prerequisites

- Node.js 16+ (matches the `engines.node` field in
  [`../../typescript/package.json`](../../typescript/package.json)).
- The `vsag` package. For a published version:
  ```bash
  npm install vsag
  ```
  For local development against an in-tree build, follow the build
  steps in [`../../typescript/`](../../typescript/) (`npm run build`
  there produces `dist/` and the native addon).

## Run

```bash
npx ts-node examples/typescript/101_index_hnsw.ts
```

…or compile first and run with plain `node`:

```bash
npx tsc examples/typescript/101_index_hnsw.ts \
    --target es2020 --module commonjs --esModuleInterop
node examples/typescript/101_index_hnsw.js
```

## File naming convention

Numbered examples mirror the numbering in
[`../cpp/`](../cpp/README.md) — a given prefix means the same topic
across languages.

## Examples

| File | Index | Notes |
| --- | --- | --- |
| [`101_index_hnsw.ts`](101_index_hnsw.ts) | HNSW | Shortest "build + KNN" round-trip in TypeScript. |

## Where to go next

- The C++ examples in [`../cpp/`](../cpp/README.md) and Python examples
  in [`../python/`](../python/README.md) cover index types and
  features not yet exposed through the Node binding.
- For deeper conceptual material, see the user guide at
  [https://vsag.io/docs](https://vsag.io/docs).
