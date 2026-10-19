"""Compile actual benchmark helper bodies and verify edge cases without index timing."""
from pathlib import Path
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
