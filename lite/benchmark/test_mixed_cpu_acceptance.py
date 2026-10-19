# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import csv
import json
import shutil
import sys
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import run_mixed_cpu_acceptance as runner
from run_mixed_cpu_acceptance import percentile, sha, verify_run
from verify_mixed_cpu_acceptance import verify



class BuildSourceTests(unittest.TestCase):
    def fixture(self, root):
        feature = root / 'feature'
        build = root / 'build'
        (feature / 'lite').mkdir(parents=True)
        (feature / 'src/lite').mkdir(parents=True)
        (feature / 'lite/CMakeLists.txt').write_text('fixture\n')
        names = ['index.cpp', 'bruteforce_backend.cpp', 'graph_backend.cpp', 'rabitq_backend.cpp']
        for name in names:
            (feature / 'src/lite' / name).write_text('// fixture\n')
        subprocess.run(['git', 'init', '-q', str(feature)], check=True)
        subprocess.run(['git', 'add', '.'], cwd=feature, check=True)
        subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture'], cwd=feature, check=True)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=feature, text=True).strip()
        build.mkdir()
        (build / 'libvsag-lite.so').write_bytes(b'library fixture, not executable')
        (build / 'CMakeCache.txt').write_text(
            'CMAKE_PROJECT_NAME:STATIC=vsag_lite\nCMAKE_BUILD_TYPE:STRING=Release\n'
            'ENABLE_RABITQ_LITE_BACKEND:BOOL=ON\nCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON\n'
            'CMAKE_HOME_DIRECTORY:INTERNAL=' + str(feature / 'lite') + '\n'
            'CMAKE_CACHEFILE_DIR:INTERNAL=' + str(build) + '\n')
        commands = [dict(directory=str(build), file=str(feature / 'src/lite' / name),
                         command='c++ -o CMakeFiles/vsag_lite_objects.dir/' + name + '.o') for name in names]
        runner.write_json(build / 'compile_commands.json', commands)
        receipt = dict(schema=1, source=runner.feature_identity(feature, head), build=str(build),
                       library_sha256=sha(build / 'libvsag-lite.so'),
                       metadata=runner.build_metadata(feature, build))
        runner.write_json(build / runner.BUILD_RECEIPT, receipt)
        return feature, build, head

    def test_matching_source_build_and_library(self):
        with tempfile.TemporaryDirectory() as directory:
            feature, build, head = self.fixture(Path(directory))
            self.assertEqual(runner.validate_build(feature, head, build)['source']['feature_sha'], head)

    def test_unreceipted_or_replaced_library_rejected(self):
        for failure in ['missing receipt', 'replaced library', 'wrong schema']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                feature, build, head = self.fixture(Path(directory))
                if failure == 'missing receipt':
                    (build / runner.BUILD_RECEIPT).unlink()
                elif failure == 'replaced library':
                    (build / 'libvsag-lite.so').write_bytes(b'old unrelated library')
                else:
                    receipt = json.loads((build / runner.BUILD_RECEIPT).read_text())
                    receipt['schema'] = 2
                    runner.write_json(build / runner.BUILD_RECEIPT, receipt)
                with self.assertRaises(ValueError):
                    runner.validate_build(feature, head, build)

    def test_source_checkout_or_inputs_changed_rejected(self):
        for failure in ['wrong sha', 'dirty source', 'untracked source']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                feature, build, head = self.fixture(Path(directory))
                if failure == 'wrong sha':
                    head = '0' * 40
                elif failure == 'dirty source':
                    (feature / 'src/lite/index.cpp').write_text('// changed\n')
                else:
                    (feature / 'src/lite/untracked.h').write_text('// overlay\n')
                with self.assertRaises(ValueError):
                    runner.validate_build(feature, head, build)

    def test_foreign_cache_or_translation_unit_rejected(self):
        for failure in ['source directory', 'build directory', 'disabled rabitq', 'foreign source', 'missing unit']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                feature, build, head = self.fixture(Path(directory))
                cache = build / 'CMakeCache.txt'
                commands = build / 'compile_commands.json'
                if failure == 'source directory':
                    cache.write_text(cache.read_text().replace(str(feature / 'lite'), '/other/lite'))
                elif failure == 'build directory':
                    cache.write_text(cache.read_text().replace(str(build), '/other/build'))
                elif failure == 'disabled rabitq':
                    cache.write_text(cache.read_text().replace('BOOL=ON', 'BOOL=OFF'))
                else:
                    entries = json.loads(commands.read_text())
                    if failure == 'foreign source':
                        entries[0]['file'] = '/other/src/lite/index.cpp'
                    else:
                        entries.pop()
                    runner.write_json(commands, entries)
                with self.assertRaises(ValueError):
                    runner.build_metadata(feature, build)

    def test_modified_metadata_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            feature, build, head = self.fixture(Path(directory))
            path = build / 'CMakeCache.txt'
            path.write_text(path.read_text() + '// changed flags\n')
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                runner.validate_build(feature, head, build)

    def test_existing_build_never_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            feature, build, head = self.fixture(Path(directory))
            before = sha(build / 'libvsag-lite.so')
            with self.assertRaises(FileExistsError):
                runner.prepare_build(feature, head, build)
            self.assertEqual(sha(build / 'libvsag-lite.so'), before)

    def test_relocated_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature, build, head = self.fixture(root)
            copy = root / 'copied-build'
            shutil.copytree(build, copy)
            with self.assertRaisesRegex(ValueError, 'copied'):
                runner.validate_build(feature, head, copy)

    def test_runtime_library_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature, build, head = self.fixture(root)
            library = build / 'libvsag-lite.so'
            digest = sha(library)
            runner.validate_loaded_library('libvsag-lite.so => ' + str(library) + ' (0x123)', library, digest)
            for text in ['libvsag-lite.so => /other/libvsag-lite.so (0x123)',
                         'libvsag-lite.so => not found', '']:
                with self.subTest(text=text), self.assertRaises(ValueError):
                    runner.validate_loaded_library(text, library, digest)

    def test_offline_receipt_cross_binding_rejects_inconsistent_identity(self):
        for failure in ['feature', 'library', 'build', 'cache', 'source hash']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                feature, build, head = self.fixture(root)
                output = root / 'output'
                output.mkdir()
                for name in ['CMakeCache.txt', 'compile_commands.json', runner.BUILD_RECEIPT]:
                    shutil.copyfile(build / name, output / name)
                (output / 'source.cpp').write_text('// benchmark fixture\n')
                receipt = json.loads((build / runner.BUILD_RECEIPT).read_text())
                identity = dict(build=str(build), feature_sha=head,
                    library_sha256=receipt['library_sha256'],
                    benchmark_source_sha256=sha(output / 'source.cpp'),
                    build_source_receipt_sha256=sha(output / runner.BUILD_RECEIPT),
                    feature_sources=dict(receipt['source']['source_sha256']))
                if failure == 'feature':
                    identity['feature_sha'] = '0' * 40
                elif failure == 'library':
                    identity['library_sha256'] = '0' * 64
                elif failure == 'build':
                    identity['build'] = '/other/build'
                elif failure == 'cache':
                    (output / 'CMakeCache.txt').write_text('// altered cache\n')
                else:
                    identity['feature_sources']['src/lite/index.cpp'] = '0' * 64
                runner.write_json(output / 'identity.json', identity)
                # Recompute the outer manifest: the receipt cross-check must catch the mismatch.
                runner.write_json(output / 'manifest.json', {
                    p.name: sha(p) for p in output.iterdir() if p.is_file()})
                with self.assertRaises(AssertionError):
                    verify(output)

    def test_invalid_build_fails_before_result_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature, build, head = self.fixture(root)
            (build / runner.BUILD_RECEIPT).unlink()
            output = root / 'new-parent/result'
            script = Path(runner.__file__)
            cpu = str(min(os.sched_getaffinity(0)))
            result = subprocess.run([sys.executable, str(script), '--feature-repo', str(feature),
                '--feature-sha', head, '--build', str(build), '--dataset', 'Data=' + str(root),
                '--cpu', cpu, '--output', str(output)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--prepare-build', result.stderr)
            self.assertFalse(output.parent.exists())


class AcceptanceTests(unittest.TestCase):
    def test_nearest_rank_and_invalid_values(self):
        self.assertEqual(percentile(list(range(1, 101)), .99), 99)
        self.assertEqual(percentile([3, 1, 2], .5), 2)
        for values in [[], [-1], [float('nan')], [float('inf')]]:
            with self.assertRaises(ValueError):
                percentile(values, .5)

    def fixture(self, folder):
        with (folder / 'samples.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['round', 'id', 'operation', 'wall_us', 'user_us',
                             'system_us', 'voluntary', 'involuntary', 'peak_kib'])
            for operation, count in dict(build=1, add=1, remove=1, update=1,
                                         query=3, save=1, load=1).items():
                for _ in range(count):
                    writer.writerow([1, 0, operation, 2, 1, 0, 0, 0, 100])
        (folder / 'quality.csv').write_text('round,query,hits,k\n0,0,10,10\n1,0,9,10\n')
        (folder / 'states.csv').write_text(
            'round,count,rss_kib,peak_kib,snapshot_bytes\n0,100,90,100,0\n1,100,95,100,20\n')

    def test_valid_raw_and_missing_call(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder)
            result = verify_run(folder, 1, 1, 1)
            self.assertEqual(result['recall'], {'0': 1, '1': .9})
            self.assertEqual(result['query']['call_only_ops_per_second'], 500000)
            text = (folder / 'samples.csv').read_text().splitlines(True)
            (folder / 'samples.csv').write_text(''.join(text[:-1]))
            with self.assertRaises(AssertionError):
                verify_run(folder, 1, 1, 1)

    def test_count_drift_and_duplicate_quality_query(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder)
            (folder / 'states.csv').write_text(
                (folder / 'states.csv').read_text().replace('1,100,', '1,99,'))
            with self.assertRaises(AssertionError):
                verify_run(folder, 1, 1, 1)
            self.fixture(folder)
            (folder / 'quality.csv').write_text('round,query,hits,k\n0,0,10,10\n1,1,9,10\n')
            with self.assertRaises(AssertionError):
                verify_run(folder, 1, 1, 1)

    @unittest.skipUnless(os.environ.get('VSAG_MIXED_PROBE'), 'Compiled C++ probe required')
    def test_resident_and_reloaded_lifecycle_parity(self):
        binary = os.environ['VSAG_MIXED_PROBE']
        with tempfile.TemporaryDirectory(dir='/dev/shm') as directory:
            root = Path(directory)
            dataset = root / 'input'
            dataset.mkdir()
            for name, count in [('base.fvecs', 96), ('queries.fvecs', 5)]:
                with (dataset / name).open('wb') as stream:
                    for row in range(count):
                        stream.write(struct.pack('<i17f', 17,
                            *[math.sin((row + 1) * (d + 3) * .17) for d in range(17)]))
            for storage in ['fp32', 'fp16', 'rabitq8']:
                receipts = []
                for resident in [False, True]:
                    tag = storage + ('-resident' if resident else '-reload')
                    output = root / tag
                    snapshots = root / (tag + '-snapshots')
                    snapshots.mkdir()
                    command = [binary, str(dataset), str(output), storage, '3', '7', str(snapshots)]
                    if resident:
                        command.append('resident')
                    result = subprocess.run(command, capture_output=True, text=True, check=True)
                    self.assertIn('resident=' + str(int(resident)), result.stdout)
                    raw = verify_run(output, 3, 7, 5)
                    self.assertTrue(all(int(s['threads']) == 1 for s in raw['states']))
                    receipts.append([(p.name, sha(p)) for p in sorted(snapshots.iterdir())])
                self.assertEqual(receipts[0], receipts[1])

    @unittest.skipUnless(os.environ.get('VSAG_MIXED_PROBE'), 'Compiled C++ probe required')
    def test_bad_lifecycle_rejected(self):
        result = subprocess.run([os.environ['VSAG_MIXED_PROBE'], 'missing', 'missing-output',
            'fp32', '1', '1', 'missing-snapshot', 'invalid'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid lifecycle mode', result.stderr)


if __name__ == '__main__':
    unittest.main()
