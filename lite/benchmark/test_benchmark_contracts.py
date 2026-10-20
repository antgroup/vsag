"""Compile actual benchmark helper bodies and verify edge cases without index timing."""
from pathlib import Path
import hashlib
import io
import json
import tarfile
import shutil
import struct
import subprocess
import tempfile
import unittest


def function_body(source, marker):
    start = source.index(marker)
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


class BenchmarkWorkflowTests(unittest.TestCase):
    def test_preparation_dependencies_are_required_before_discovery(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / '.github/workflows/lite.yml').read_text()
        begin = workflow.index('      - name: Install benchmark preparation dependencies')
        end = workflow.index('      - name: Lite CI contracts, collector fixtures', begin)
        benchmark_steps = workflow[begin:end]
        self.assertIn("if: matrix.mode == 'rabitq-on'", benchmark_steps)
        self.assertIn('numpy==1.26.4 h5py==3.11.0 pyarrow==17.0.0', benchmark_steps)
        imports = "python3 -c 'import numpy, h5py, pyarrow'"
        discovery = "python3 -m unittest discover -s lite/benchmark -p 'test_*.py'"
        self.assertIn(imports, benchmark_steps)
        self.assertIn(discovery, benchmark_steps)
        self.assertLess(benchmark_steps.index('pip install'), benchmark_steps.index(imports))
        self.assertLess(benchmark_steps.index(imports), benchmark_steps.index(discovery))
        self.assertIn('VSAG_MIXED_PROBE: ${{ github.workspace }}/build-lite/lite_acceptance_mixed_cpu_acceptance', benchmark_steps)
        self.assertIn('test -x "$VSAG_MIXED_PROBE"', benchmark_steps)
        self.assertLess(benchmark_steps.index('test -x'), benchmark_steps.index(discovery))
        self.assertNotIn('continue-on-error', benchmark_steps)

    def test_benchmark_build_and_lite_ci_contracts_coexist(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / '.github/workflows/lite.yml').read_text()
        mode = workflow.index('if [[ "$MODE" == rabitq-on ]]; then')
        block = workflow[mode:workflow.index('fi', mode)]
        self.assertIn('flags+=(-DENABLE_BENCHMARKS=ON)', block)
        self.assertIn('Lite CI contracts, collector fixtures and 90 percent gate', workflow)
        self.assertIn("python3 -m unittest discover -s lite/tools -p 'test_*.py'", workflow)
        self.assertIn('--fail-under 90', workflow)


class ReproductionDocumentationTests(unittest.TestCase):
    def test_quickstart_requires_dependencies_and_compiled_probe(self):
        root = Path(__file__).parent
        guide = (root / 'ACCEPTANCE.md').read_text()
        block = guide.split('## Reproduce and verify\n\n', 1)[1].split('For a new run,', 1)[0]
        self.assertIn('numpy==1.26.4 h5py==3.11.0 pyarrow==17.0.0', block)
        self.assertIn('-DENABLE_BENCHMARKS=ON', block)
        self.assertIn('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', block)
        self.assertIn('export VSAG_MIXED_PROBE="$BENCH_BUILD/lite_acceptance_mixed_cpu_acceptance"', block)
        for first, second in [('cmake --build', 'export VSAG_MIXED_PROBE'),
                              ('export VSAG_MIXED_PROBE', 'test -x'),
                              ('test -x', 'python3 -m unittest'),
                              ('python3 -c', 'python3 -m unittest')]:
            self.assertLess(block.index(first), block.index(second))

    def test_resident_protocol_separates_new_cohort_and_quality_verdict(self):
        guide = (Path(__file__).parent / 'ACCEPTANCE.md').read_text()
        block = guide.split('## Current-version resident protocol\n', 1)[1].split(
            '## Community review and remaining acceptance', 1)[0]
        for token in ['SIFT128', 'GIST960', 'Cohere768', '1000 query rows',
                      '--resident --trials 3 --rounds 20 --cycles 250',
                      'initial/final current-state FP64 exact truth',
                      'not invented', 'fresh current Full', 'not labeled hours-long']:
            self.assertIn(token, block)

    def test_current_and_historical_recipes_remain_distinct(self):
        recipes = (Path(__file__).parent / 'README.md').read_text()
        build = 'cmake --build "$BENCH_FEATURE_BUILD" --target vsag_lite_static -j2'
        install = 'cmake --install "$BENCH_FEATURE_BUILD" --prefix "$BENCH_CURRENT_LITE_INSTALL"'
        self.assertLess(recipes.index(build), recipes.index(install))
        root = Path(__file__).parent
        index = (root / 'ACCEPTANCE.md').read_text()
        recipes = (root / 'README.md').read_text()
        self.assertIn('cba340660f2ea479b316d7b73a27f34186b8a1d9', index)
        self.assertIn('E9 measures `a3a76a8`', index)
        self.assertIn('historical measurement', index)
        self.assertIn('## Reproduction recipes', recipes)
        self.assertIn('### Guarded historical Full, package and quality replay', recipes)
        for tool in ['prepare_sift.py', 'prepare_gist.cpp', 'prepare_cohere.py',
                     'run_mixed_cpu_acceptance.py', 'verify_mixed_cpu_acceptance.py',
                     'pack_mixed_cpu_acceptance.py', 'build_fresh_full_reference.py',
                     'run_pinned_full_acceptance.py', 'verify_pinned_full_acceptance.py',
                     'measure_runtime_package.py', 'verify_runtime_package.py',
                     'run_quality_target.py', 'verify_quality_target.py']:
            self.assertTrue((root / tool).is_file())
            self.assertIn(tool, recipes)
        self.assertIn('unpack(Path(sys.argv[1]), Path(sys.argv[2]))', recipes)
        self.assertIn('not generic latest-source commands', recipes)
        self.assertIn('--prepare-build --feature-repo', recipes)
        self.assertIn('build-source-receipt.json', recipes)
        self.assertIn('not signed or tamper-proof provenance', recipes)


class ResidentPublishedEvidenceTests(unittest.TestCase):
    def test_report_rows_match_verified_compact_evidence(self):
        root = Path(__file__).parent
        result = root / 'results/resident-three-datasets-20261019'
        archive = result / 'raw.tar.gz'
        metadata = json.loads((result / 'archive.json').read_text())
        self.assertEqual(archive.stat().st_size, metadata['bytes'])
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), metadata['sha256'])
        with tarfile.open(archive, 'r:gz') as stream:
            summary = json.load(stream.extractfile('summary.json'))
            complete = json.load(stream.extractfile('completed.json'))
            protocol = json.load(stream.extractfile('frozen-protocol.json'))
        self.assertEqual(complete['processes'], 27)
        self.assertEqual(complete['mutation_calls'], 405000)
        self.assertEqual(complete['interleaved_queries'], 405000)
        self.assertEqual(protocol['query_rows'], 1000)
        self.assertFalse(complete['full_comparison'])
        main_report = (root / 'ACCEPTANCE_REPORT.md').read_text()
        report = (result / 'README.md').read_text()
        self.assertEqual(len(summary), 9)
        labels = {'fp32': 'FP32', 'fp16': 'FP16', 'rabitq8': 'RaBitQ8'}
        for cell in summary.values():
            query = cell['operations']['query']
            for operation in cell['operations'].values():
                for metric in operation.values():
                    self.assertEqual(len(metric['values']), 3)
                    self.assertEqual(metric['median'], sorted(metric['values'])[1])
            row = (f"| {cell['dataset']} | {labels[cell['storage']]} | "
                   f"{query['p50_us']['median']:.3f} | {query['p99_us']['median']:.3f} | "
                   f"{query['call_only_ops_per_second']['median']:.1f} | "
                   f"{cell['quality']['initial']['median']:.4f} → "
                   f"{cell['quality']['final']['median']:.4f} |")
            self.assertIn(row, report)
            self.assertIn(row, main_report)


class ReviewFixedEvidenceTests(unittest.TestCase):
    def test_current_report_metrics_and_source_binding(self):
        root = Path(__file__).parent
        result = root / 'results/review-fixed-resident-20261020'
        archive = result / 'raw.tar.gz'
        metadata = json.loads((result / 'archive.json').read_text())
        self.assertEqual(archive.stat().st_size, metadata['bytes'])
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), metadata['sha256'])
        with tarfile.open(archive, 'r:gz') as outer:
            manifest = json.load(outer.extractfile('manifest.json'))
            self.assertEqual(len(outer.getmembers()), metadata['members'])
            self.assertEqual(set(manifest) | {'manifest.json'}, set(outer.getnames()))
            for name, digest in manifest.items():
                self.assertEqual(hashlib.sha256(outer.extractfile(name).read()).hexdigest(), digest)
            summary = json.load(outer.extractfile('summary.json'))
            complete = json.load(outer.extractfile('completed.json'))
            protocol = json.load(outer.extractfile('frozen-protocol.json'))
            freeze = json.load(outer.extractfile('protocol-freeze-receipt.json'))
            reuse = json.load(outer.extractfile('input-reuse-note.json'))
            self.assertTrue(reuse['input_preparation_reused'])
            self.assertTrue(reuse['frozen_before_new_measurement'])
            self.assertEqual(freeze['sha256'], hashlib.sha256(
                outer.extractfile('frozen-protocol.json').read()).hexdigest())
            self.assertEqual(protocol['feature_sha'], 'cba340660f2ea479b316d7b73a27f34186b8a1d9')
            for key, expected in [('trials', 3), ('rounds', 20), ('cycles', 250),
                                  ('query_rows', 1000), ('base_count', 10000)]:
                self.assertEqual(protocol[key], expected)
            for dataset in ['SIFT', 'GIST', 'Cohere']:
                inner_bytes = outer.extractfile(dataset + '/raw.tar.gz').read()
                inner_metadata = json.load(outer.extractfile(dataset + '/archive.json'))
                self.assertEqual(len(inner_bytes), inner_metadata['bytes'])
                self.assertEqual(hashlib.sha256(inner_bytes).hexdigest(), inner_metadata['sha256'])
                with tarfile.open(fileobj=io.BytesIO(inner_bytes), mode='r:gz') as inner:
                    identity = json.load(inner.extractfile('identity.json'))
                    receipt_bytes = inner.extractfile('build-source-receipt.json').read()
                    receipt = json.loads(receipt_bytes)
                    self.assertEqual(identity['feature_sha'], protocol['feature_sha'])
                    self.assertEqual(identity['benchmark_parent'], protocol['benchmark_sha'])
                    self.assertEqual(identity['library_sha256'], protocol['library_sha256'])
                    self.assertEqual(receipt['library_sha256'], protocol['library_sha256'])
                    self.assertEqual(receipt['source']['feature_sha'], protocol['feature_sha'])
                    self.assertEqual(hashlib.sha256(receipt_bytes).hexdigest(),
                                     protocol['build_receipt_sha256'])
        self.assertEqual(complete['processes'], 27)
        self.assertEqual(complete['mutation_calls'], 405000)
        self.assertEqual(complete['interleaved_queries'], 405000)
        self.assertFalse(complete['quality_accepted'])
        self.assertFalse(complete['full_comparison'])
        self.assertEqual(len(summary), 9)
        report = (result / 'README.md').read_text()
        main_report = (root / 'ACCEPTANCE_REPORT.md').read_text()
        labels = {'fp32': 'FP32', 'fp16': 'FP16', 'rabitq8': 'RaBitQ8'}
        for cell in summary.values():
            prefix = f"| {cell['dataset']} | {labels[cell['storage']]} | "
            for operation in cell['operations'].values():
                for metric in operation.values():
                    self.assertEqual(len(metric['values']), 3)
                    self.assertEqual(metric['median'], sorted(metric['values'])[1])
            query = cell['operations']['query']
            row = (prefix + f"{query['p50_us']['median']:.3f} | {query['p99_us']['median']:.3f} | "
                   f"{query['call_only_ops_per_second']['median']:.1f} | "
                   f"{cell['quality']['initial']['median']:.4f} → "
                   f"{cell['quality']['final']['median']:.4f} |")
            self.assertIn(row, report)
            self.assertIn(row, main_report)
            for name in ['add', 'remove', 'update']:
                operation = cell['operations'][name]
                row = (prefix + f"{operation['p50_us']['median']:.3f} | "
                       f"{operation['p99_us']['median']:.3f} | "
                       f"{operation['call_only_ops_per_second']['median']:.1f} | "
                       f"{operation['user_sum_ms']['median']:.3f} | "
                       f"{operation['system_sum_ms']['median']:.3f} |")
                self.assertIn(row, report)
            for name in ['save', 'load']:
                operation = cell['operations'][name]
                row = (prefix + f"{operation['p50_us']['median']/1000:.3f} | "
                       f"{operation['p99_us']['median']/1000:.3f} | "
                       f"{operation['call_only_ops_per_second']['median']:.1f} |")
                self.assertIn(row, report)
            for name in ['query', 'save', 'load']:
                operation = cell['operations'][name]
                row = (prefix + f"{name} | {operation['count']['median']:.0f} | "
                       f"{operation['user_sum_ms']['median']:.3f} | "
                       f"{operation['system_sum_ms']['median']:.3f} |")
                self.assertIn(row, report)
            process = cell['process']
            memory = cell['memory']
            self.assertIn(prefix + f"{memory['snapshot_final_bytes']['median']/1048576:.3f} | "
                          f"{cell['operations']['build']['wall_sum_ms']['median']/1000:.3f} |", report)
            self.assertIn(prefix + f"{process['wall_seconds']['median']:.2f} | "
                          f"{process['user_seconds']['median']:.2f} | "
                          f"{process['system_seconds']['median']:.2f} | "
                          f"{process['cpu_core_utilization_percent']['median']:.1f} |", report)
            self.assertIn(prefix + f"{process['peak_kib']['median']/1024:.3f} | "
                          f"{memory['rss_initial_kib']['median']/1024:.3f} | "
                          f"{memory['rss_final_kib']['median']/1024:.3f} | "
                          f"{memory['rss_change_kib']['median']:.0f} |", report)
            self.assertIn(prefix + ' | '.join(f"{process[k]['median']:.0f}" for k in
                          ['minor_faults', 'major_faults', 'voluntary', 'involuntary']) + ' |', report)


@unittest.skipUnless(shutil.which('c++'), 'C++ compiler required for benchmark helper regression')
class BenchmarkContractTests(unittest.TestCase):
    def test_all_percentile_implementations(self):
        root = Path(__file__).parent
        paths = ['main.cpp', 'dataset_main.cpp', 'full_main.cpp',
                 'full_dataset_main.cpp', 'full_rabitq_dataset_main.cpp']
        code = '''#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
'''
        for i, name in enumerate(paths):
            body = function_body((root / name).read_text(), 'double\npercentile(')
            code += body.replace('percentile(', f'percentile_{i}(', 1) + '\n'
        code += '''int main() {
using Fn = double (*)(std::vector<double>, double);
std::vector<Fn> functions = {percentile_0, percentile_1, percentile_2, percentile_3, percentile_4};
for (auto fn : functions) {
    for (uint64_t n : {1, 2, 3, 100, 900}) {
        std::vector<double> values;
        for (uint64_t i = n; i > 0; --i) values.push_back(static_cast<double>(i));
        for (double f : {0.0, 0.01, 0.5, 0.99, 1.0}) {
            const auto rank = std::max<uint64_t>(1, static_cast<uint64_t>(std::ceil(f * n)));
            if (fn(values, f) != static_cast<double>(rank)) return 1;
            if (values.front() != static_cast<double>(n)) return 2;
        }
    }
    bool rejected = false;
    try { fn({}, 0.5); } catch (const std::exception&) { rejected = true; }
    if (!rejected) return 3;
    for (double invalid : {-1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        rejected = false;
        try { fn({1}, invalid); } catch (const std::exception&) { rejected = true; }
        if (!rejected) return 4;
    }
}
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'probe.cpp'
            binary = Path(directory) / 'probe'
            source.write_text(code)
            subprocess.run(['c++', '-std=c++17', '-O2', str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_record_reader_rejects_invalid_lengths_and_dimensions(self):
        root = Path(__file__).parent
        for name in ['dataset_main.cpp', 'full_dataset_main.cpp']:
            with self.subTest(source=name), tempfile.TemporaryDirectory() as directory:
                original = (root / name).read_text()
                prefix = '''#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
'''
                code = prefix + function_body(original, 'int32_t\nread_dim(') + '\n'
                code += function_body(original, 'template <typename T>\nstd::vector<std::vector<T>>\nread_records(')
                code += '''\nint main(int argc, char** argv) {
try { auto rows = read_records<float>(argv[1], std::stoi(argv[2])); return rows.size() == 2 ? 0 : 2; }
catch (const std::exception&) { return 3; }
}
'''
                source = Path(directory) / 'reader.cpp'
                binary = Path(directory) / 'reader'
                source.write_text(code)
                subprocess.run(['c++', '-std=c++17', str(source), '-o', str(binary)], check=True)
                data = Path(directory) / 'vectors.fvecs'
                valid = struct.pack('<iffiff', 2, 1.0, 2.0, 2, 3.0, 4.0)
                data.write_bytes(valid)
                subprocess.run([str(binary), str(data), '2'], check=True)
                for dimension in ['0', '-1']:
                    self.assertNotEqual(subprocess.run([str(binary), str(data), dimension]).returncode, 0)
                for malformed in [b'', valid[:-1], b'xx']:
                    data.write_bytes(malformed)
                    self.assertNotEqual(subprocess.run([str(binary), str(data), '2']).returncode, 0)


if __name__ == '__main__':
    unittest.main()
