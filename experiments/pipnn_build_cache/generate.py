# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Generate isolated, instrumented copies; never rewrite production sources."""
import argparse
from pathlib import Path
import re


def before_main(source):
    match = re.search(r"\bint\s+main\s*\(", source)
    if match is None:
        raise ValueError("experiment main not found")
    return source[:match.start()]


def generate(repo, output):
    here = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    source = (repo / "src/impl/pipnn/pipnn_graph_builder.cpp").read_text()
    source = source.replace("private:", "public:")
    marker = "    // Preserve source/target emission order and the existing bidirectional reservoir policy."
    if source.count(marker) != 1:
        raise ValueError("unsupported builder revision: expected one small-leaf emission marker")
    capture = """    if (::capture_sink != nullptr) {
        Snapshot saved;
        saved.k = neighbor_count;
        saved.members.assign(leaf.begin(), leaf.end());
        saved.edges.reserve(count * neighbor_count);
        for (uint64_t source = 0; source < count; ++source) {
            for (uint64_t j = 0; j < neighbor_count; ++j) {
                const auto [distance, target] = nearest[source * neighbor_count + j];
                saved.edges.push_back({distance, leaf[target]});
            }
        }
        ::capture_sink->push_back(std::move(saved));
    }
    if (::capture_only) {
        return;
    }
"""
    source = source.replace(marker, capture + marker)
    (output / "builder_exposed.cpp").write_text(source)
    pair_marker = "PiPNNPipeline::pair_distance(uint32_t lhs, uint32_t rhs) const {"
    if source.count(pair_marker) != 1:
        raise ValueError("unsupported pair_distance definition")
    (output / "final_reuse_builder.cpp").write_text(
        source.replace(pair_marker, pair_marker + "\n    ++::prune_distance_calls;")
    )
    base = before_main((here / "bench.cpp").read_text())
    derived = before_main((here / "derived.cpp").read_text())
    (output / "derived_support.cpp").write_text(base)
    (output / "final_reuse_base_support.cpp").write_text(
        base.replace("builder_exposed.cpp", "final_reuse_builder.cpp")
    )
    (output / "final_reuse_support.cpp").write_text(
        derived.replace("derived_support.cpp", "final_reuse_base_support.cpp")
    )
    search = (here / "warm_final.cpp").read_text()
    search = search.replace('#include "quality512.cpp"', '#include "quality512.cpp"\n#include "search_qps_helpers.cpp"')
    search, count = re.subn(r'timing\["query"\]\s*=\s*probe512\([^;]+;', '', search)
    if count != 1:
        raise ValueError("quality call not found")
    search, count = re.subn(
        r'std::cout\s*<<\s*"keep=".*?std::endl;',
        'std::cout << "keep=" << keep << " path=" << (warm ? "warm" : "reference") << " core=" << timing["total_core_s"] << std::endl;',
        search,
        flags=re.S,
    )
    if count != 1:
        raise ValueError("benchmark progress output not found")
    search, count = re.subn(
        r'r\["different_adjacency_rows"\]\s*=\s*changed;',
        'r["different_adjacency_rows"] = changed;\n r["search"] = benchmark_pair(baseline, warmed, v, queries, live, out, keep);',
        search,
    )
    if count != 1:
        raise ValueError("paired comparison not found")
    (output / "search_qps_driver.cpp").write_text(search)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args.repo.resolve(), args.out.resolve())
