<!-- agent-hints
canonical: docs/agents/coding-standards.md
purpose: C++ coding standards and common patterns enforced in VSAG
key-facts:
  - C++ files use 4-space indent and a 100-col limit; non-C++ text has no line-width limit
  - .cpp suffix and trailing newline required
  - Public APIs live in include/vsag/; implementation in src/
  - Prefer uint64_t over size_t for macOS compatibility
related:
  - ../../CONTRIBUTING.md
  - ../docs/en/src/development/code_structure.md
last-reviewed: 2026-08-13
-->

# Coding Standards

Follow the Google C++ Style Guide with the project-specific rules below.

## Formatting

- Use 4-space indentation in C++ source and header files.
- Use `.cpp` instead of `.cc`
- Keep C++ source and header files within a 100-character line limit.
- Do not impose the C++ line limit on Markdown or other non-C++ text files.
- Ensure committed text files end with a trailing newline

## Code layout

- Keep public APIs in `include/vsag/`.
- Keep implementation in `src/`.
- Place tests near the code they validate when appropriate.
- Add or update Doxygen comments for public APIs.
- Keep code in the `vsag` namespace unless the file clearly requires otherwise.

## What to optimize for

- Preserve API compatibility unless a breaking change is explicitly intended.
- Keep performance in mind for hot paths, memory layout, and parallel code.
- Follow existing project structure and naming before introducing new patterns.
- Prefer minimal, targeted changes over broad refactors.

## Common code patterns

- Builder-style chained APIs are common.
- `std::shared_ptr<T>` is used widely in public interfaces.
- Prefer existing error-handling/result patterns used in the surrounding code.
- Prefer `uint64_t` over `size_t` in code changes to avoid potential macOS
  compile issues.

## Public C API opaque tags

C++ struct definitions use CamelCase. The existing opaque C API tags
`vsag_error`, `vsag_dataset`, `vsag_engine`, `vsag_engine_options`, `vsag_filter`,
`vsag_index`, and `vsag_search_result` retain their public spelling. The struct-name checker permits
these exact tags only in `include/vsag/vsag_c_api.h` and `src/vsag_c_api.cpp`;
clang-tidy recognizes the same finite tag set. Other `vsag_*` struct names and
ordinary C++ implementation structures remain subject to the naming rule.

## Things to avoid

- Avoid changing files under `extern/` unless the task explicitly requires
  third-party dependency changes.
- Be careful with performance-sensitive code and cross-platform behavior.
- If changing build logic or dependencies, document the rationale clearly.
