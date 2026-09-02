# C++ Coding Guide

VSAG follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with
project-specific formatting, naming, and static-analysis rules. The configuration files at the
repository root are authoritative:

- [`.clang-format`](https://github.com/antgroup/vsag/blob/main/.clang-format) defines mechanical
  formatting.
- [`.clang-tidy`](https://github.com/antgroup/vsag/blob/main/.clang-tidy) defines naming and
  static-analysis checks.

VSAG requires **clang-format 15** and **clang-tidy 15** exactly. Other versions may produce
different formatting or diagnostics and are not accepted by CI.

## Code layout

- Put public APIs in `include/vsag/` and implementation code in `src/`.
- Use `.cpp` for C++ source files, not `.cc`.
- Keep code in the `vsag` namespace unless the file clearly requires otherwise.
- Preserve API compatibility unless a breaking change is explicitly planned.
- Add or update Doxygen comments when changing a public API.
- Prefer `uint64_t` over `size_t` in new code to avoid macOS compatibility problems.
- Do not modify `extern/` unless the change explicitly concerns a third-party dependency.
- End every committed text file with a trailing newline.

## Formatting rules

The formatter starts from the Google style and applies the following VSAG rules:

| Rule | VSAG style |
| --- | --- |
| Indentation | 4 spaces |
| C++ line length | At most 100 columns |
| Access specifiers | Outdented by one indentation level |
| Return types | Placed on a separate line from the function name |
| Short blocks, functions, and `if` statements | Never collapsed onto one line |
| Pointer and reference alignment | `*` and `&` bind to the type |
| Includes | Sorted automatically |
| Wrapped parameters and arguments | One per line instead of bin-packed |
| Trailing comments | Aligned when they form a consecutive group |
| Existing comment text | Not automatically reflowed |

For example, clang-format 15 formats a class declaration like this:

```cpp
class Example {
public:
    Result<DatasetPtr>
    Build(const DatasetPtr& base, const std::string& parameters);

private:
    void
    reset_state();

    DatasetPtr dataset_;
};
```

The 100-column limit applies to C++ source and header files. It does not impose a line-width limit
on Markdown or other non-C++ text.

## Naming rules

The `readability-identifier-naming` check enforces these names:

| Entity | Style | Example |
| --- | --- | --- |
| Namespace | `lower_case` | `vsag::storage` |
| Class | `CamelCase` | `HGraphIndex` |
| Struct | `CamelCase` | `SearchResult` |
| Type template parameter | `CamelCase` | `DataType` |
| Value template parameter | `lower_case` | `max_degree` |
| Free or C-style function | `lower_case` | `normalize_vector` |
| Public method | `CamelCase` | `Build` |
| Private method | `lower_case` | `reset_state` |
| Variable | `lower_case` | `search_result` |
| Private or protected data member | `lower_case_` | `search_result_` |
| Macro definition | `UPPER_CASE` | `CHECK_ARGUMENT` |
| Enum constant | `UPPER_CASE` | `ROUTING` |
| Global constant | `UPPER_CASE` | `DEFAULT_RESIZE_INCREASE_COUNT_BIT` |

`make lint` also runs `scripts/linters/check-struct-names.py`. It checks named struct definitions
across tracked project C/C++ files, including files outside the normal clang-tidy source scope.
Therefore, all new named structs must use `CamelCase`.

For entities without an explicit project rule, follow the Google C++ style and the convention in
the surrounding code.

## Choosing between `class` and `struct`

Use a `struct` for a passive, value-like data carrier:

- Its primary purpose is to group related values.
- Its data members are intentionally public and callers may modify them independently.
- It does not need to protect an invariant or hide a representation.
- It does not own a resource that requires controlled access or lifetime management.

Examples include `Binary`, `SparseVector`, and `MultiVector`. A `struct` may still have default
member initializers, constructors, or small helper functions. Having a method does not by itself
require using a `class`; the deciding factor is whether the type remains a transparent data value.

Use a `class` when the type represents behavior or must control its state:

- It maintains invariants through its public methods.
- It hides implementation details behind private or protected members.
- It owns or manages resources whose lifetime must be controlled.
- It provides a behavioral or polymorphic interface, especially one with virtual methods.

Examples include `BinarySet`, `Dataset`, `Filter`, and `Index`. Prefer `class` for new interfaces
and polymorphic base types.

When unsure, ask whether callers should be allowed to change every data member directly and in any
order. If yes, use `struct`; if the type must validate, coordinate, or restrict those changes, use
`class`. Do not use `struct` merely to avoid writing `public:`, and do not expose mutable
implementation details just to make a type aggregate-initializable.

Both classes and structs use `CamelCase`. Preserve the existing choice for a public type unless an
intentional API change requires otherwise; do not switch between `class` and `struct` as a cosmetic
cleanup.

## Static analysis

All enabled clang-tidy diagnostics are treated as errors. The configuration enables the following
check families:

- `bugprone-*` for likely logic defects and suspicious constructs.
- `clang-analyzer-*` for path-sensitive correctness and lifetime analysis.
- `clang-diagnostic-*` for compiler diagnostics exposed through clang-tidy.
- `modernize-*` for supported modern C++ practices.
- `openmp-*` for OpenMP correctness.
- `performance-*` for avoidable copies and inefficient operations.
- `readability-*` for maintainability and naming consistency.

Some checks are deliberately disabled because they conflict with VSAG's low-level and
performance-sensitive code or are too noisy for the current codebase:

| Disabled check | Reason |
| --- | --- |
| `clang-analyzer-deadcode.DeadStores` | Stored-but-unread values produce too much noise |
| `clang-diagnostic-deprecated-builtins` | Some deprecated builtins are required for toolchain or platform compatibility |
| `bugprone-easily-swappable-parameters` | Produces too many false positives |
| `modernize-use-trailing-return-type` | Conflicts with the project return-type style |
| `modernize-avoid-c-arrays` | C arrays are intentional in SIMD and other low-level code |
| `readability-identifier-length` | Short names are acceptable when clear in context |
| `readability-magic-numbers` | Numeric-heavy vector-index code makes the check excessively noisy |
| `readability-function-cognitive-complexity` | Deferred until existing violations can be addressed |
| `readability-redundant-access-specifiers` | Repeated access specifiers may separate method groups visually |

A disabled check is not a recommendation to ignore the underlying concern. Use meaningful
parameter names and named constants whenever they make the code clearer.

### Suppressing a diagnostic

Fix the underlying issue whenever practical. If a diagnostic is a false positive or the construct
is required for compatibility or performance, use the narrowest suppression and explain why it is
safe:

```cpp
// NOLINTNEXTLINE(performance-no-int-to-ptr): the platform API stores this handle as an integer.
auto* handle = reinterpret_cast<Handle*>(raw_handle);
```

Name the exact check in `NOLINT`, `NOLINTNEXTLINE`, or a tightly scoped
`NOLINTBEGIN`/`NOLINTEND` pair. Avoid an unqualified `NOLINT` or a file-wide suppression unless
there is no narrower practical option.

## Running the checks

Install the required tools:

```bash
# Ubuntu or Debian
sudo apt-get install clang-format-15 clang-tidy-15 clang-tools-15

# macOS
brew install llvm@15
```

Format all project C++ files and inspect the resulting diff:

```bash
make fmt
git diff --check
```

clang-tidy requires a compilation database in `build-release/`, so create a release build first:

```bash
make release
make lint
```

The default lint target uses production `.cpp` files under `src/` as clang-tidy entry points and
excludes `*_test.cpp`. It also runs the repository-wide struct-name checker. Tests and other
directories are not separate clang-tidy entry points in this command.

`make fix-lint` applies suggested replacements in place. It may make broad edits, so use it on a
clean or safely backed-up working tree, review the complete diff, and rerun the checks and relevant
tests:

```bash
make fix-lint
make fmt
make lint
make test
```

## Before submitting a change

- Format all changed C++ files with clang-format 15.
- Run clang-tidy 15 for changed production sources.
- Ensure names follow the rules above, including `CamelCase` struct names.
- Keep every `NOLINT` check-specific and document its rationale.
- Add tests for new features and regression tests for bug fixes.
- Keep unit-test coverage for the C++ library at or above 90%.
- Run `git diff --check` and review the complete diff.
