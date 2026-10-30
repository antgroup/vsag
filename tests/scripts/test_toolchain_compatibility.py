#!/usr/bin/env python3
"""Behavioral regression checks for installed-package and standard audits."""
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import shlex
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ToolchainCompatibilityTest(unittest.TestCase):
    def test_client_records_the_imported_library_in_a_custom_libdir(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            library = prefix / 'lib64/libvsag.so'
            library.parent.mkdir()
            library.touch()
            stale = prefix / 'stale/libvsag.so'
            stale.parent.mkdir()
            stale.touch()
            config = prefix / 'lib64/cmake/vsag'
            config.mkdir(parents=True)
            (config / 'vsagConfig.cmake').write_text(
                'add_library(vsag::vsag SHARED IMPORTED)\n'
                'set_target_properties(vsag::vsag PROPERTIES\n'
                f'  IMPORTED_LOCATION "{library}"\n'
                '  INTERFACE_COMPILE_FEATURES cxx_std_17)\n')
            build = prefix / 'client'
            result = subprocess.run(
                ['cmake', '-S', str(ROOT / 'tests/compatibility'), '-B', str(build),
                 '-Dvsag_DIR=' + str(config)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((build / 'installed-library.txt').read_text().strip(),
                             str(library))

    def test_standard_audit_without_python39_path_api(self):
        with tempfile.TemporaryDirectory() as directory:
            entries = [{'file': str(ROOT / 'src/probe.cpp'), 'directory': str(ROOT),
                        'arguments': ['c++', '-std=c++20', '-c', 'probe.cpp']},
                       {'file': str(ROOT) + '-neighbor/src/probe.cpp',
                        'directory': str(ROOT), 'arguments': ['c++', '-std=c++11']},
                       {'file': str(ROOT / 'src-neighbor/probe.cpp'),
                        'directory': str(ROOT), 'arguments': ['c++', '-std=c++11']}]
            (Path(directory) / 'compile_commands.json').write_text(json.dumps(entries))
            script = '''
import runpy
import sys
from pathlib import Path
def unavailable(*args):
    raise AttributeError('is_relative_to is unavailable before Python 3.9')
Path.is_relative_to = unavailable
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
            result = subprocess.run(
                ['python3', '-c', script, str(ROOT / 'scripts/ci/check-cxx-standards.py'),
                 directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("'cpu': 1", result.stdout)

    def test_simd_commands_omit_unused_avx512_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / 'CMakeLists.txt').write_text('''
cmake_minimum_required(VERSION 3.18)
project(simd_flags LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(ENABLE_TESTS OFF)
set(DIST_CONTAINS_AVX512 ON)
set(DIST_CONTAINS_AVX512VPOPCNTDQ ON)
add_library(cpuinfo INTERFACE)
add_subdirectory("${VSAG_SOURCE_DIR}/src/simd" simd)
''')
            build = source / 'build'
            result = subprocess.run(
                ['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
                 '-DVSAG_SOURCE_DIR=' + str(ROOT)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            commands = json.loads((build / 'compile_commands.json').read_text())
            checked = set()
            for entry in commands:
                name = Path(entry['file']).name
                if name in {'avx512.cpp', 'avx512vpopcntdq.cpp'}:
                    arguments = shlex.split(entry['command'])
                    self.assertIn('-mavx512f', arguments)
                    self.assertNotIn('-mavx512pf', arguments)
                    self.assertNotIn('-mavx512er', arguments)
                    checked.add(name)
            self.assertEqual(checked, {'avx512.cpp', 'avx512vpopcntdq.cpp'})

    def test_empty_installed_headers_fail_before_configuring_a_client(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / 'include/vsag').mkdir(parents=True)
            result = subprocess.run(
                ['bash', str(ROOT / 'scripts/release/check_compatibility.sh'),
                 str(prefix), '1'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('No public headers found', result.stderr)
            self.assertNotIn('CMake', result.stderr)

    def test_elf_path_guard_accepts_origin_paths_and_rejects_absolute_entries(self):
        script = (ROOT / 'scripts/release/check_compatibility.sh').read_text()
        guard = next(line for line in script.splitlines() if line.startswith('if grep -E '))
        pattern = shlex.split(guard)[3]
        cases = [
            ('(RUNPATH) Library runpath: [$ORIGIN/../lib]', False),
            ('(RPATH) Library rpath: [$ORIGIN/lib:$ORIGIN/../lib]', False),
            ('(NEEDED) Shared library: [libvsag.so]', False),
            ('(RUNPATH) Library runpath: [/build/lib]', True),
            ('(RPATH) Library rpath: [$ORIGIN/lib:/build/lib]', True),
            ('(NEEDED) Shared library: [/build/libvsag.so]', True),
        ]
        for value, forbidden in cases:
            with self.subTest(value=value):
                result = subprocess.run(['grep', '-Eq', pattern], input=value,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0 if forbidden else 1)

    def test_make_lint_filters_escape_checkout_paths(self):
        from tests.scripts.test_build_benchmark_workflow import step

        with tempfile.TemporaryDirectory(prefix="vsag-work[1].+ '$-") as directory:
            checkout = Path(directory).resolve()
            shutil.copy2(ROOT / 'Makefile', checkout / 'Makefile')
            linters = checkout / 'scripts/linters'
            linters.mkdir(parents=True)
            checker = linters / 'check-struct-names.py'
            checker.write_text('#!/bin/sh\nexit 0\n')
            checker.chmod(0o755)
            runner = linters / 'run-clang-tidy-20.sh'
            runner.write_text(
                '#!/usr/bin/env python3\nimport json, os, sys\n'
                'from pathlib import Path\n'
                'Path(os.environ["TEST_LINT_TRACE"]).write_text(json.dumps(sys.argv[1:]))\n')
            runner.chmod(0o755)
            trace = checkout / 'arguments.json'
            invocations = [('lint', ['make', 'lint']), ('fix-lint', ['make', 'fix-lint'])]
            for workflow in ['pr-ci-community.yml', 'pr-ci-team.yml']:
                body = (ROOT / '.github/workflows' / workflow).read_text()
                name = 'Run Lint' if workflow == 'pr-ci-community.yml' else 'Lint changed sources'
                script = textwrap.dedent(step(body, name).split('run: |\n', 1)[1])
                invocations.append((workflow, ['bash', '-e', '-c', script]))
            changed_pattern = "src/it's $(touch injected).cpp$"
            for target, command in invocations:
                result = subprocess.run(
                    command, cwd=checkout,
                    env=dict(os.environ, TEST_LINT_TRACE=str(trace),
                             LINT_PATTERN=changed_pattern),
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = json.loads(trace.read_text())
                pattern = re.compile(arguments[arguments.index('-source-filter') + 1])
                self.assertIsNotNone(pattern.search(str(checkout / 'src/probe.cpp')))
                self.assertIsNone(pattern.search(str(checkout / 'src/probe_test.cpp')))
                self.assertIsNone(pattern.search(str(checkout / 'extern/probe.cpp')))
                neighbor = str(checkout).replace('[1]', '1') + '/src/probe.cpp'
                self.assertIsNone(pattern.search(neighbor))
                self.assertEqual('-fix' in arguments, target == 'fix-lint')
                if target.endswith('.yml'):
                    self.assertEqual(arguments[-1], changed_pattern)
                    self.assertFalse((checkout / 'injected').exists())

    def test_source_baseline_propagates_both_abis_to_library_and_client(self):
        from tests.scripts.test_build_benchmark_workflow import step
        from tests.scripts.test_dependency_workflows import job_text

        body = job_text(ROOT / '.github/workflows/toolchain-compatibility.yml',
                        'source-baseline')
        self.assertIn('abi: [0, 1]', body)
        build = step(body, 'Build C++20 library with the older compiler')
        client = step(body, 'Validate older C++17 client')
        for value in [build, client]:
            self.assertIn('ABI: ${{ matrix.abi }}', value)
        scripts = [textwrap.dedent(value.split('run: |\n', 1)[1])
                   for value in [build, client]]
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            binaries = checkout / 'bin'
            binaries.mkdir()
            trace = checkout / 'trace.jsonl'
            runner = binaries / 'trace-command'
            runner.write_text(
                '#!/usr/bin/env python3\nimport json, os, sys\n'
                'with open(os.environ["TEST_TRACE"], "a") as output:\n'
                '    output.write(json.dumps([sys.argv, os.environ.get("CXX")]) + "\\n")\n')
            runner.chmod(0o755)
            for name in ['make', 'cmake', 'g++-9', 'clang++-13']:
                (binaries / name).symlink_to(runner.name)
            helper = checkout / 'scripts/release/check_compatibility.sh'
            helper.parent.mkdir(parents=True)
            helper.write_text('#!/bin/sh\nexec trace-command "$@"\n')
            for compiler, cxx in [('gcc9', 'g++-9'), ('clang13', 'clang++-13')]:
                for abi in ['0', '1']:
                    trace.write_text('')
                    env = dict(os.environ, COMPILER=compiler, ABI=abi,
                               TEST_TRACE=str(trace),
                               PATH=str(binaries) + os.pathsep + os.environ['PATH'])
                    for script in scripts:
                        result = subprocess.run(['bash', '-e', '-c', script],
                                                cwd=checkout, env=env,
                                                capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                    calls = [json.loads(line) for line in trace.read_text().splitlines()]
                    make = next(args for args, _ in calls if Path(args[0]).name == 'make')
                    self.assertIn('-DENABLE_CXX11_ABI=' + abi, make[-1])
                    self.assertIn('-DCMAKE_CXX_COMPILER=' + cxx, make[-1])
                    self.assertEqual(calls[-1][0][1:], ['/tmp/vsag-baseline', abi])
                    self.assertEqual(calls[-1][1], cxx)

    def test_missing_compile_arguments_fail_with_a_standard_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            for fields in [{}, {'arguments': []}, {'command': ''}, {'command': None}]:
                entry = {'file': str(ROOT / 'src/probe.cpp'), 'directory': str(ROOT)}
                entry.update(fields)
                (Path(directory) / 'compile_commands.json').write_text(json.dumps([entry]))
                result = subprocess.run(
                    ['python3', str(ROOT / 'scripts/ci/check-cxx-standards.py'), directory],
                    capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('expected C++20', result.stderr)
                self.assertNotIn('Traceback', result.stderr)

    def audit(self, cpu_standard, node_standard, cuda_standard):
        with tempfile.TemporaryDirectory() as directory:
            entries = []
            for source, standard in [('src/probe.cpp', cpu_standard),
                                     ('node_bindings/probe.cpp', node_standard),
                                     ('src/gpu/probe.cu', cuda_standard)]:
                entries.append({'file': str(ROOT / source), 'directory': str(ROOT),
                                'arguments': ['c++', '-std=' + standard, '-c', source]})
            (Path(directory) / 'compile_commands.json').write_text(json.dumps(entries))
            return subprocess.run(
                ['python3', str(ROOT / 'scripts/ci/check-cxx-standards.py'), directory],
                capture_output=True, text=True)

    def test_historical_standard_aliases_are_accepted(self):
        for cpu in ['c++20', 'gnu++20', 'c++2a', 'gnu++2a']:
            for client in ['c++17', 'gnu++17', 'c++1z', 'gnu++1z']:
                with self.subTest(cpu=cpu, client=client):
                    result = self.audit(cpu, client, client)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrong_cpu_and_client_standards_are_rejected(self):
        for standards in [('c++17', 'c++17', 'c++17'),
                          ('c++20', 'c++20', 'c++17'),
                          ('c++20', 'c++17', 'c++20')]:
            with self.subTest(standards=standards):
                self.assertNotEqual(self.audit(*standards).returncode, 0)


if __name__ == '__main__':
    unittest.main()
