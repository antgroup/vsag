# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import csv
import io
import json
import shutil
import sys
import math
import os
from pathlib import Path
import struct
import statistics
import subprocess
import tarfile
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


class SnapshotReleaseTests(unittest.TestCase):
    def fixture(self, root):
        snapshots = root / 'run'
        snapshots.mkdir()
        for name in ['round-1', 'round-2']:
            (snapshots / name).write_bytes(name.encode())
        records = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                   for p in snapshots.iterdir()}
        return snapshots, records

    def test_verified_temporary_snapshots_released_after_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots, records = self.fixture(root)
            (root / 'raw.csv').write_bytes(b'original evidence')
            runner.release_verified_snapshots(root, snapshots, records)
            self.assertFalse(snapshots.exists())
            self.assertEqual((root / 'raw.csv').read_bytes(), b'original evidence')
            self.assertEqual(set(records), {'round-1', 'round-2'})

    def test_full_set_checked_before_any_removal(self):
        for failure in ['changed bytes', 'extra entry', 'missing entry', 'symlink entry']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                snapshots, records = self.fixture(root)
                if failure == 'changed bytes':
                    (snapshots / 'round-2').write_bytes(b'changed')
                elif failure == 'extra entry':
                    (snapshots / 'unverified').write_bytes(b'not a snapshot')
                elif failure == 'missing entry':
                    (snapshots / 'round-2').unlink()
                else:
                    (snapshots / 'round-2').unlink()
                    (snapshots / 'round-2').symlink_to(snapshots / 'round-1')
                before = {p.name for p in snapshots.iterdir()}
                with self.assertRaises(ValueError):
                    runner.release_verified_snapshots(root, snapshots, records)
                self.assertEqual(before, {p.name for p in snapshots.iterdir()})
                self.assertEqual((snapshots / 'round-1').read_bytes(), b'round-1')

    def test_outside_or_symlink_directory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scratch = root / 'scratch'
            scratch.mkdir()
            snapshots, records = self.fixture(root)
            alias = scratch / 'alias'
            alias.symlink_to(snapshots, target_is_directory=True)
            for target in [snapshots, alias, scratch]:
                with self.subTest(target=str(target)), self.assertRaises(ValueError):
                    runner.release_verified_snapshots(scratch, target, records)
            self.assertEqual(len(list(snapshots.iterdir())), 2)


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



class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_schedule(self):
        self.assertEqual(runner.checkpoint_rounds(5, 2), [0, 2, 4, 5])
        self.assertEqual(runner.checkpoint_rounds(5, 8), [0, 5])
        self.assertEqual(runner.checkpoint_rounds(5, 0), [])

    @unittest.skipUnless(os.environ.get('VSAG_MIXED_PROBE'), 'Compiled C++ probe required')
    def test_checkpoint_parity_rebuild_and_corrupt_evidence(self):
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
                quality = []
                for every in [0, 2]:
                    output = root / f'{storage}-{every}'
                    snapshots = root / f'{storage}-{every}-snapshots'
                    snapshots.mkdir()
                    command = [binary, str(dataset), str(output), storage, '3', '7',
                               str(snapshots), 'resident']
                    if every:
                        command.append(str(every))
                    subprocess.run(command, capture_output=True, text=True, check=True)
                    raw = verify_run(output, 3, 7, 5)
                    quality.append((output / 'quality.csv').read_bytes())
                    receipts.append([(p.name, sha(p)) for p in sorted(snapshots.iterdir())])
                    if every:
                        curve = runner.verify_checkpoints(output, 3, 7, 5, every)
                        self.assertEqual(set(curve), {'0', '2', '3'})
                        for n in ['0', '3']:
                            self.assertEqual(curve[n]['maintained_recall'], raw['recall'][n])
                        self.assertEqual(curve['0']['fresh_recall'], curve['0']['maintained_recall'])
                        target = output / 'checkpoint-2-queries.csv'
                        saved = target.read_text()
                        lines = saved.splitlines()
                        values = lines[1].split(',')
                        values[2] = str((int(values[2]) + 1) % 11)
                        lines[1] = ','.join(values)
                        target.write_text('\n'.join(lines) + '\n')
                        with self.assertRaises(AssertionError):
                            runner.verify_checkpoints(output, 3, 7, 5, every)
                        target.write_text(saved)
                        with self.assertRaises((AssertionError, FileNotFoundError)):
                            runner.verify_checkpoints(output, 3, 7, 5, 1)
                        runner.verify_checkpoints(output, 3, 7, 5, every)
                self.assertEqual(quality[0], quality[1])
                self.assertEqual(receipts[0], receipts[1])

    @unittest.skipUnless(os.environ.get('VSAG_MIXED_PROBE'), 'Compiled C++ probe required')
    def test_invalid_checkpoint_interval_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'must-not-exist'
            for every in ['0', '1000001']:
                result = subprocess.run([os.environ['VSAG_MIXED_PROBE'], 'missing', str(target),
                    'fp32', '1', '1', 'missing-snapshot', 'resident', every], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('invalid checkpoint interval', result.stderr)
                self.assertFalse(target.exists())



class CurvePublishedTests(unittest.TestCase):
    def test_curve_archive_raw_and_all_report_cells(self):
        root = Path(__file__).parent / 'results/crud-degradation-curve-20261022'
        receipt = json.loads((root / 'archive.json').read_text())
        self.assertEqual(sha(root / 'raw.tar.gz'), receipt['sha256'])
        self.assertEqual((root / 'raw.tar.gz').stat().st_size, receipt['bytes'])
        from pack_mixed_cpu_acceptance import unpack
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / 'evidence'
            unpack(root / 'raw.tar.gz', evidence)
            manifest = json.loads((evidence / 'manifest.json').read_text())
            self.assertEqual(len(manifest)+1, receipt['members'])
            for name, item in manifest.items():
                self.assertEqual((evidence / name).stat().st_size, item['bytes'])
                self.assertEqual(sha(evidence / name), item['sha256'])
            unpack(evidence / 'measurements/raw.tar.gz', evidence / 'raw')
            raw = evidence / 'raw'
            self.assertEqual(verify(raw), receipt['audit'])
            self.assertEqual(receipt['audit']['mutation_calls'], 225000)
            self.assertEqual(receipt['audit']['interleaved_queries'], 225000)
            identity = json.loads((raw / 'identity.json').read_text())
            self.assertEqual(identity['configuration']['checkpoint_every'], 10)
            self.assertTrue(identity['configuration']['resident'])
            audit = json.loads((evidence / 'independent-audit.json').read_text())
            self.assertTrue(audit['all_300_snapshot_hashes_and_sizes_match_E12'])
            self.assertEqual(audit['independent_numpy_FP64_query_states'], 110)
            report = (root / 'README.md').read_text()
            rows = json.loads((raw / 'rows.json').read_text())
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual(row['trial'], 0)
                self.assertEqual(set(row['checkpoints']), {str(n) for n in range(0, 101, 10)})
                for n in range(0, 101, 10):
                    cell = row['checkpoints'][str(n)]
                    expected = f"| {n} | {n*750} | {cell['maintained_recall']:.4f} | {cell['fresh_recall']:.4f} |"
                    self.assertIn(expected, report)
            self.assertIn('不是新算法优化', report)
            self.assertIn('5.59个百分点', report)
            self.assertIn('不是加载旧快照导致', report)


class ExtendedPublishedTests(unittest.TestCase):
    def test_gist_extension_raw_metrics_and_report(self):
        root = Path(__file__).parent
        result = root / 'results/extended-resident-gist-20261020'
        receipt = json.loads((result / 'archive.json').read_text())
        archive = result / 'raw.tar.gz'
        self.assertEqual(sha(archive), receipt['sha256'])
        self.assertEqual(archive.stat().st_size, receipt['bytes'])
        from pack_mixed_cpu_acceptance import unpack
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / 'evidence'
            unpack(archive, evidence)
            manifest = json.loads((evidence / 'manifest.json').read_text())
            self.assertEqual(len(manifest) + 1, receipt['members'])
            for name, digest in manifest.items():
                self.assertEqual(sha(evidence / name), digest)
            unpack(evidence / 'measurements/raw.tar.gz', evidence / 'measurements/raw')
            raw = evidence / 'measurements/raw'
            self.assertEqual(verify(raw), receipt['audit'])
            rows = json.loads((raw / 'rows.json').read_text())
            self.assertEqual(len(rows), 9)
            self.assertEqual(receipt['audit']['mutation_calls'], 675000)
            self.assertEqual(receipt['audit']['interleaved_queries'], 675000)
            summary = json.loads((evidence / 'summary.json').read_text())
            self.assertEqual(set(summary), {'fp32', 'fp16', 'rabitq8'})
            report = (result / 'README.md').read_text()
            main = (root / 'ACCEPTANCE_REPORT.md').read_text()
            med = statistics.median
            labels = dict(fp32='FP32', fp16='FP16', rabitq8='RaBitQ8')
            for mode, cell in summary.items():
                selected = sorted([r for r in rows if r['storage'] == mode], key=lambda r: r['trial'])
                self.assertEqual([r['trial'] for r in selected], [0, 1, 2])
                for op, fields in cell['operations'].items():
                    for key, values in fields.items():
                        self.assertEqual(values, [r[op][key] for r in selected])
                for n, fields in cell['checkpoints'].items():
                    for key, values in fields.items():
                        self.assertEqual(values, [int(r['states'][int(n)][key]) for r in selected])
                for n, values in cell['recalls'].items():
                    self.assertEqual(values, [r['recall'][n] for r in selected])
                changes = [int(r['states'][100]['rss_before_persistence_kib']) -
                           int(r['states'][50]['rss_before_persistence_kib']) for r in selected]
                self.assertEqual(changes, cell['late_rss_delta_kib'])
                prefix = '| ' + labels[mode] + ' | '
                query = cell['operations']['query']
                line = prefix + ' | '.join([f"{med(query['p50_us']):.3f}", f"{med(query['p99_us']):.3f}",
                    f"{med(query['call_only_ops_per_second']):.1f}", f"{med(cell['recalls']['0']):.4f}",
                    f"{med(cell['recalls']['100']):.4f}"]) + ' |'
                self.assertIn(line, report)
                self.assertIn(line, main)
                rss_line = prefix + ' | '.join(f"{med(cell['checkpoints'][str(n)]['rss_before_persistence_kib'])/1024:.3f}"
                    for n in [0, 20, 50, 80, 100]) + ' | ' + str(med(changes)) + ' |'
                self.assertIn(rss_line, report)
                for offset, row in enumerate(selected):
                    tag = f"GIST-{mode}-{row['trial']}"
                    fields = {}
                    for raw_line in (raw / (tag + '.resources.txt')).read_text().splitlines():
                        if ': ' in raw_line:
                            key, value = raw_line.strip().split(': ', 1)
                            fields[key] = value
                    elapsed = 0
                    for part in fields['Elapsed (wall clock) time (h:mm:ss or m:ss)'].split(':'):
                        elapsed = elapsed * 60 + float(part)
                    process = cell['processes'][offset]
                    self.assertEqual(process['wall_s'], elapsed)
                    self.assertEqual(process['user_s'], float(fields['User time (seconds)']))
                    self.assertEqual(process['system_s'], float(fields['System time (seconds)']))
                    self.assertEqual(process['cpu_core_percent'], (process['user_s'] + process['system_s']) / elapsed * 100)
                    operations = ['build', 'add', 'remove', 'update', 'query', 'save', 'load']
                    api_wall = sum(row[op]['wall_sum_ms'] for op in operations) / 1000
                    self.assertEqual(process['timed_api_wall_s'], api_wall)
                    self.assertEqual(process['other_wall_s'], elapsed - api_wall)
                    self.assertEqual(process['timed_api_cpu_s'], sum(row[op]['user_sum_ms'] + row[op]['system_sum_ms'] for op in operations) / 1000)
                    early, late, ids = [], [], set()
                    with (raw / tag / 'samples.csv').open() as samples:
                        for sample in csv.DictReader(samples):
                            n = int(sample['round'])
                            if sample['operation'] == 'update':
                                ids.add(int(sample['id']))
                            if sample['operation'] == 'query':
                                if 1 <= n <= 20:
                                    early.append(float(sample['wall_us']))
                                if 81 <= n <= 100:
                                    late.append(float(sample['wall_us']))
                    self.assertEqual(cell['distinct_update_ids'][offset], len(ids))
                    self.assertEqual(cell['windows'][offset], dict(first_20_query_p50_us=percentile(early, .5), last_20_query_p50_us=percentile(late, .5)))
                    for key, source in [('peak_kib', 'Maximum resident set size (kbytes)'),
                            ('minor_faults', 'Minor (reclaiming a frame) page faults'),
                            ('major_faults', 'Major (requiring I/O) page faults'),
                            ('voluntary', 'Voluntary context switches'), ('involuntary', 'Involuntary context switches')]:
                        self.assertEqual(process[key], int(fields[source]))
                cpu_line = prefix + ' | '.join(f"{med([p[key] for p in cell['processes']]):.{digits}f}"
                    for key, digits in [('wall_s', 2), ('user_s', 2), ('system_s', 2), ('cpu_core_percent', 1)])
                cpu_line += f" | {med([p['peak_kib'] for p in cell['processes']])/1024:.3f} |"
                self.assertIn(cpu_line, report)
                phase_line = prefix + ' | '.join(f"{med([p[key] for p in cell['processes']]):.3f}"
                    for key in ['timed_api_wall_s', 'other_wall_s', 'timed_api_cpu_s']) + ' |'
                self.assertIn(phase_line, report)
                faults_line = prefix + ' | '.join(str(med([p[key] for p in cell['processes']]))
                    for key in ['minor_faults', 'major_faults', 'voluntary', 'involuntary']) + ' |'
                self.assertIn(faults_line, report)
                windows_line = prefix + ' | '.join([
                    f"{med([p['first_20_query_p50_us'] for p in cell['windows']]):.3f}",
                    f"{med([p['last_20_query_p50_us'] for p in cell['windows']]):.3f}",
                    str(med(cell['distinct_update_ids']))]) + ' |'
                self.assertIn(windows_line, report)
                save = cell['operations']['save']
                load = cell['operations']['load']
                io_line = (prefix + f"{med(save['p50_us'])/1000:.3f} / {med(save['p99_us'])/1000:.3f} | "
                    f"{med(load['p50_us'])/1000:.3f} / {med(load['p99_us'])/1000:.3f} | "
                    f"{med(cell['checkpoints']['100']['snapshot_bytes'])/1048576:.3f} |")
                self.assertIn(io_line, report)
                for op in ['add', 'remove', 'update']:
                    values = cell['operations'][op]
                    line = prefix + ' | '.join(f'{med(values[key]):.{digits}f}' for key, digits in
                        [('p50_us', 3), ('p99_us', 3), ('call_only_ops_per_second', 1),
                         ('user_sum_ms', 3), ('system_sum_ms', 3)]) + ' |'
                    self.assertIn(line, report)
            with tarfile.open(root / 'results/review-fixed-resident-20261020/raw.tar.gz', 'r:gz') as old:
                inner = old.extractfile('GIST/raw.tar.gz').read()
            with tarfile.open(fileobj=io.BytesIO(inner), mode='r:gz') as old:
                previous = json.load(old.extractfile('rows.json'))
            for row in rows:
                old_row = next(r for r in previous if (r['storage'], r['trial']) == (row['storage'], row['trial']))
                for name, digest in old_row['snapshots'].items():
                    self.assertEqual(row['snapshots'][name], digest)




class IsolationTests(unittest.TestCase):
    def test_mutation_mode_validation(self):
        self.assertEqual([runner.mutation_factor(m) for m in ['all', 'update', 'replace', 'reinsert']], [3, 1, 2, 2])
        with self.assertRaises(ValueError):
            runner.mutation_factor('invalid')

    @unittest.skipUnless(os.environ.get('VSAG_MIXED_PROBE'), 'Compiled C++ probe required')
    def test_bad_mutation_mode_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'must-not-exist'
            result = subprocess.run([os.environ['VSAG_MIXED_PROBE'], 'missing', str(target),
                'fp32', '1', '1', 'missing-snapshot', 'resident', '1', 'invalid'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('invalid mutation mode', result.stderr)
            self.assertFalse(target.exists())

    @unittest.skipUnless(os.environ.get('VSAG_MIXED_PROBE'), 'Compiled C++ probe required')
    def test_isolated_geometry_stages_truth_and_ceiling(self):
        binary = os.environ['VSAG_MIXED_PROBE']
        f32 = lambda v: struct.unpack('<f', struct.pack('<f', v))[0]
        with tempfile.TemporaryDirectory(dir='/dev/shm') as directory:
            root = Path(directory)
            dataset = root / 'input'
            dataset.mkdir()
            vectors = {}
            for name, count in [('base.fvecs', 96), ('queries.fvecs', 5)]:
                vectors[name] = [[f32(math.sin((row + 1) * (d + 3) * .17)) for d in range(17)] for row in range(count)]
                with (dataset / name).open('wb') as stream:
                    for row in vectors[name]:
                        stream.write(struct.pack('<i17f', 17, *row))
            truth_by_workload = {}
            for storage in ['fp32', 'fp16', 'rabitq8']:
                for mode in ['update', 'replace', 'reinsert']:
                    output = root / (storage + '-' + mode)
                    snapshots = root / (storage + '-' + mode + '-snapshots')
                    snapshots.mkdir()
                    subprocess.run([binary, str(dataset), str(output), storage, '3', '7', str(snapshots),
                                    'resident', '2', mode], capture_output=True, text=True, check=True)
                    raw = verify_run(output, 3, 7, 5, mode)
                    curve = runner.verify_checkpoints(output, 3, 7, 5, 2, mode)
                    ceiling = runner.verify_ceiling(output, 3, 5)
                    self.assertEqual(ceiling['ef_search'], 96)
                    self.assertEqual(curve['3']['mutation_calls'], 21 * runner.mutation_factor(mode))
                    self.assertEqual(raw['query']['count'], 21 * runner.mutation_factor(mode))
                    self.assertEqual('update' in raw, mode == 'update')
                    self.assertEqual('remove' in raw, mode != 'update')
                    self.assertEqual('add' in raw, mode != 'update')
                    if storage == 'fp32':
                        self.assertEqual(ceiling['maintained_recall'], 1)
                        self.assertEqual(ceiling['fresh_recall'], 1)
                    base = [row[:] for row in vectors['base.fvecs']]
                    truth_by_workload[storage, mode] = []
                    for number in range(4):
                        if number and mode != 'reinsert':
                            for cycle in range(7):
                                identifier = ((number - 1) * 65537 + cycle * 8191) % 96
                                donor = (identifier + number * 97 + cycle + 1) % 96
                                base[identifier] = [f32(f32(.875 * a) + f32(.125 * b)) for a, b in
                                                    zip(base[identifier], vectors['base.fvecs'][donor])]
                        if number not in [0, 2, 3]:
                            continue
                        with (output / f'checkpoint-{number}-neighbors.csv').open() as stream:
                            rows = list(csv.DictReader(stream))
                        truths = []
                        for q, query in enumerate(vectors['queries.fvecs']):
                            expected = sorted(range(96), key=lambda i:
                                (sum((float(a) - b) ** 2 for a, b in zip(base[i], query)), i))[:10]
                            actual = [int(r['truth_id']) for r in rows[q * 10:(q + 1) * 10]]
                            self.assertEqual(actual, expected)
                            truths.append(actual)
                        truth_by_workload[storage, mode].append(truths)
                    if mode == 'reinsert':
                        self.assertTrue(all(t == truth_by_workload[storage, mode][0] for t in truth_by_workload[storage, mode]))
                    target = output / 'ceiling.csv'
                    original = target.read_text()
                    target.write_text(original.replace('3,0,0,96,', '3,0,0,95,', 1))
                    with self.assertRaises(AssertionError):
                        runner.verify_ceiling(output, 3, 5)
                    target.write_text(original)
                self.assertEqual(truth_by_workload[storage, 'update'], truth_by_workload[storage, 'replace'])
            self.assertEqual(truth_by_workload['fp32', 'update'], truth_by_workload['fp16', 'update'])
            self.assertEqual(truth_by_workload['fp32', 'update'], truth_by_workload['rabitq8', 'update'])


class IsolationPublishedTests(unittest.TestCase):
    def test_all_isolation_archives_and_report_cells(self):
        root = Path(__file__).parent / 'results/crud-isolation-20261022'
        receipt = json.loads((root / 'archive.json').read_text())
        archive = root / 'raw.tar.gz'
        self.assertEqual(sha(archive), receipt['sha256'])
        self.assertEqual(archive.stat().st_size, receipt['bytes'])
        self.assertEqual(receipt['mutation_calls'], 375000)
        self.assertEqual(receipt['interleaved_queries'], 375000)
        self.assertEqual(receipt['processes'], 9)
        self.assertEqual(receipt['checkpoint_states'], 45)
        report = (root / 'README.md').read_text()
        from pack_mixed_cpu_acceptance import unpack
        counts = 0
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / 'evidence'
            unpack(archive, evidence)
            manifest = json.loads((evidence / 'manifest.json').read_text())
            self.assertEqual(len(manifest) + 1, receipt['members'])
            for name, item in manifest.items():
                self.assertEqual((evidence / name).stat().st_size, item['bytes'])
                self.assertEqual(sha(evidence / name), item['sha256'])
            independent = json.loads((evidence / 'analysis/independent-audit.json').read_text())
            self.assertEqual(independent['status'], 'PASS')
            self.assertEqual(independent['independent_numpy_FP64_query_states'], 100)
            self.assertTrue(independent['same_data_fresh_rebuild_results_exact'])
            self.assertTrue(independent['reinsert_fresh_results_and_truth_constant'])
            slot_audit = json.loads((evidence / 'slot-controls/audit.json').read_text())
            self.assertEqual(slot_audit['status'], 'PASS')
            self.assertEqual(len(slot_audit['cells']), 2)
            self.assertEqual(sha(evidence / 'slot-controls/slot-control.cpp'), slot_audit['source_sha256'])
            curve = json.loads((evidence / 'analysis/curve.json').read_text())
            self.assertEqual(len(curve), 45)
            for mode in ['update', 'replace', 'reinsert']:
                raw = evidence / ('raw-' + mode)
                nested = evidence / ('measurements/' + mode + '.tar.gz')
                nested_receipt = json.loads((evidence / ('measurements/' + mode + '.json')).read_text())
                self.assertEqual(sha(nested), nested_receipt['sha256'])
                self.assertEqual(nested.stat().st_size, nested_receipt['bytes'])
                unpack(nested, raw)
                audit = verify(raw)
                self.assertEqual(audit, nested_receipt['unpacked_audit'])
                self.assertEqual(audit['processes'], 3)
                self.assertEqual(audit['mutation_calls'], 75000 if mode == 'update' else 150000)
                counts += audit['mutation_calls']
                identity = json.loads((raw / 'identity.json').read_text())
                self.assertEqual(identity['configuration']['mutation_mode'], mode)
                self.assertEqual(identity['configuration']['checkpoint_every'], 25)
                self.assertEqual(sha(raw / 'source.cpp'), sha(evidence / 'tooling/mixed_cpu_acceptance.cpp'))
                # Historical E14 must remain bound to its measured source, not a later CSV extension.
                self.assertEqual(identity['benchmark_source_sha256'],
                                 '4a2d5b163caac56c6cf8f85808725fe954672969f12dbbad40b74b7d5de93554')
                self.assertEqual(sha(raw / 'source.cpp'), identity['benchmark_source_sha256'])
                rows = json.loads((raw / 'rows.json').read_text())
                if mode in ['replace', 'reinsert']:
                    cell = next(c for c in slot_audit['cells'] if c['mode'] == mode)
                    self.assertTrue(cell['replay_snapshot_matches_timed'])
                    self.assertTrue(cell['actual_snapshot_ids_match_simulated_order'])
                    self.assertTrue(cell['all_stored_fp32_vectors_match_independent_replay'])
                    fp32 = next(r for r in rows if r['storage'] == 'fp32')
                    self.assertEqual(cell['snapshots']['maintained.snapshot'], fp32['snapshots']['round-100'])
                    with (evidence / ('slot-controls/' + mode + '/neighbors.csv')).open() as stream:
                        control_rows = list(csv.DictReader(stream))
                    self.assertEqual(len(control_rows), 20000)
                    for ef in [512, 10000]:
                        selected = [r for r in control_rows if int(r['ef']) == ef]
                        self.assertEqual(len(selected), 10000)
                        for side in ['maintained', 'canonical', 'same_order']:
                            hits = 0
                            for q in range(1000):
                                group = selected[q * 10:(q + 1) * 10]
                                self.assertEqual([int(r['rank']) for r in group], list(range(10)))
                                self.assertTrue(all(int(r['query']) == q for r in group))
                                truth = {int(r['truth_id']) for r in group}
                                values = [(float(r[side + '_distance']), int(r[side + '_id'])) for r in group]
                                self.assertEqual(values, sorted(values))
                                self.assertEqual(len({i for _, i in values}), 10)
                                self.assertTrue(all(math.isfinite(d) and 0 <= i < 10000 for d, i in values))
                                hits += len(truth & {i for _, i in values})
                            self.assertEqual(hits / 10000, cell['quality'][str(ef)][side])
                for number in [0, 25, 50, 75, 100]:
                    values = [next(r for r in rows if r['storage'] == storage)['checkpoints'][str(number)]['maintained_recall']
                              for storage in ['fp32', 'fp16', 'rabitq8']]
                    self.assertIn('| ' + str(number) + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |', report)
                for row in rows:
                    cell = row['checkpoints']['100']
                    values = [cell['maintained_recall'], cell['fresh_recall'], row['ceiling']['maintained_recall'], row['ceiling']['fresh_recall']]
                    self.assertIn(' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |', report)
            self.assertEqual(counts, 375000)


class RepairPilotPublishedTests(unittest.TestCase):
    def test_all_repair_pilot_cells_and_rejected_candidate_preserved(self):
        root = Path(__file__).parent / 'results/repair-reselection-pilot-20261023'
        receipt = json.loads((root / 'archive.json').read_text())
        archive = root / 'raw.tar.gz'
        self.assertEqual(sha(archive), receipt['sha256'])
        self.assertEqual(archive.stat().st_size, receipt['bytes'])
        self.assertEqual((receipt['cells'], receipt['processes'], receipt['mutation_calls']), (8, 16, 700000))
        report = (root / 'README.md').read_text()
        from pack_mixed_cpu_acceptance import unpack
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / 'evidence'
            unpack(archive, evidence)
            manifest = json.loads((evidence / 'manifest.json').read_text())
            self.assertEqual(len(manifest) + 1, receipt['members'])
            for path, item in manifest.items():
                self.assertEqual(sha(evidence / path), item['sha256'])
                self.assertEqual((evidence / path).stat().st_size, item['bytes'])
            rejected = json.loads((evidence / 'r1/rejected.json').read_text())
            self.assertEqual(rejected['commands'][-1]['exit'], 8)
            self.assertIn('1580', (evidence / 'r1/candidate-ctest.log').read_text())
            self.assertIn('3072', (evidence / 'r1/candidate-ctest.log').read_text())
            identity = json.loads((evidence / 'r2/identity.json').read_text())
            self.assertEqual(sha(evidence / 'r2/graph_backend.cpp'), identity['candidate_source_sha256'])
            self.assertEqual(sha(evidence / 'r2/baseline.cpp'), identity['production_source_sha256'])
            self.assertIn('100% tests passed', (evidence / 'followup/san-ctest.log').read_text())
            audit = json.loads((evidence / 'audit.json').read_text())
            self.assertEqual(audit['status'], 'PASS')
            self.assertTrue(audit['no_production_change'])
            rows = json.loads((evidence / 'followup/rows.json').read_text())
            self.assertEqual(rows, audit['rows'])
            self.assertEqual(len(rows), 7)
            plan = json.loads((evidence / 'followup/plan.json').read_text())
            self.assertEqual([[r['dataset'], r['storage'], r['workload']] for r in rows], plan['cells'])
            released = json.loads((evidence / 'followup/released-snapshot-receipts.json').read_text())
            self.assertEqual(len(released), 14)
            for row in rows:
                self.assertTrue(row['initial_snapshot_equal'])
                self.assertTrue(row['final_geometry_layout_equal'])
                tag = row['dataset'] + '-' + row['storage'] + '-' + row['workload']
                for side in ['baseline', 'candidate']:
                    folder = evidence / ('followup/' + tag + '-' + side)
                    with (folder / 'neighbors.csv').open() as stream:
                        records = list(csv.DictReader(stream))
                    self.assertEqual(len(records), 20000)
                    for ef in [512, 10000]:
                        selected = [r for r in records if int(r['ef']) == ef]
                        self.assertEqual(len(selected), 10000)
                        for view in ['maintained', 'canonical', 'same_order']:
                            hits = 0
                            for q in range(1000):
                                group = selected[q * 10:(q + 1) * 10]
                                self.assertEqual([(int(r['query']), int(r['rank'])) for r in group], [(q, rank) for rank in range(10)])
                                truth = {int(r['truth_id']) for r in group}
                                values = [(float(r[view + '_distance']), int(r[view + '_id'])) for r in group]
                                self.assertEqual(values, sorted(values))
                                self.assertEqual(len({i for _, i in values}), 10)
                                self.assertTrue(all(math.isfinite(d) and 0 <= i < 10000 for d, i in values))
                                hits += len(truth & {i for _, i in values})
                            self.assertEqual(hits / 10000, row['quality'][side][str(ef)][view])
                    with (folder / 'mutation-time.csv').open() as stream:
                        timing = list(csv.DictReader(stream))
                    self.assertEqual(timing, [row['loop_time'][side]])
                    self.assertEqual(int(timing[0]['calls']), 25000 if row['workload'] == 'update' else 50000)
                    self.assertTrue(math.isfinite(float(timing[0]['wall_ms'])) and float(timing[0]['wall_ms']) >= 0)
                    release = next(r for r in released if r['cell'] == tag + '-' + side)
                    self.assertEqual(next(i['sha256'] for i in release['snapshots'] if i['path'].endswith('/maintained.snapshot')), row[side + '_sha256'])
                base = row['quality']['baseline']['512']['maintained']
                candidate = row['quality']['candidate']['512']['maintained']
                self.assertIn(f' | {base:.4f} | {candidate:.4f} | {(candidate-base)*100:+.2f} | ', report)


class RepairMixedPublishedTests(unittest.TestCase):
    def test_mixed_curve_costs_counts_and_rejected_cache_are_recomputable(self):
        root = Path(__file__).parent / 'results/repair-mixed-cost-20261023'
        receipt = json.loads((root / 'archive.json').read_text())
        archive = root / 'raw.tar.gz'
        self.assertEqual(sha(archive), receipt['sha256'])
        self.assertEqual(archive.stat().st_size, receipt['bytes'])
        from pack_mixed_cpu_acceptance import unpack
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / 'evidence'
            unpack(archive, evidence)
            manifest = json.loads((evidence / 'manifest.json').read_text())
            self.assertEqual(len(manifest) + 1, receipt['members'])
            for name, item in manifest.items():
                self.assertEqual(sha(evidence / name), item['sha256'])
                self.assertEqual((evidence / name).stat().st_size, item['bytes'])
            complete = json.loads((evidence / 'mixed/completed.json').read_text())
            self.assertEqual(complete['status'], 'PASS')
            self.assertEqual((complete['mutation_calls'], complete['interleaved_queries'], complete['save_load_pairs']),
                             (225000, 225000, 300))
            identity = json.loads((evidence / 'mixed/identity.json').read_text())
            self.assertEqual(identity['schema'], 'private-experimental-mixed-v1')
            self.assertEqual(sha(evidence / 'mixed/source.cpp'), identity['source_sha256'])
            rows = json.loads((evidence / 'mixed/rows.json').read_text())
            self.assertEqual([r['side'] for r in rows], ['baseline', 'r2', 'cache'])
            for row in rows:
                folder = evidence / 'mixed' / row['side']
                actual = verify_run(folder, 100, 250, 1000)
                actual['checkpoints'] = runner.verify_checkpoints(folder, 100, 250, 1000, 25)
                self.assertEqual(actual, {k:v for k,v in row.items() if k not in ['side', 'snapshots']})
                self.assertEqual(set(row['snapshots']), {'round-'+str(n) for n in range(1,101)})
                self.assertTrue(all(v['bytes'] > 0 and len(v['sha256']) == 64 for v in row['snapshots'].values()))
                with (folder / 'samples.csv').open() as stream:
                    sequence = iter(csv.DictReader(stream))
                    def consume(number, identifier, operation):
                        item = next(sequence)
                        self.assertEqual((int(item['round']), int(item['id']), item['operation']),
                                         (number, identifier, operation))
                    consume(0, -1, 'build')
                    query = 0
                    for number in range(1, 101):
                        for cycle in range(250):
                            identifier = ((number-1)*65537 + cycle*8191) % 10000
                            for operation in ['update', 'remove', 'add']:
                                consume(number, identifier, operation)
                                consume(number, query % 1000, 'query')
                                query += 1
                        consume(number, -1, 'save')
                        consume(number, -1, 'load')
                    self.assertIsNone(next(sequence, None))
            self.assertEqual(rows[1]['snapshots'], rows[2]['snapshots'])
            for name in ['quality.csv'] + [f'checkpoint-{n}-neighbors.csv' for n in [0,25,50,75,100]]:
                self.assertEqual((evidence / 'mixed/r2' / name).read_bytes(), (evidence / 'mixed/cache' / name).read_bytes())
            for number in [0,25,50,75,100]:
                control = None
                for side in ['baseline','r2','cache']:
                    with (evidence / 'mixed' / side / f'checkpoint-{number}-neighbors.csv').open() as stream:
                        current = [tuple(row[k] for k in ['round','query','rank','truth_id','fresh_id','fresh_distance'])
                                   for row in csv.DictReader(stream)]
                    if control is None:
                        control = current
                    self.assertEqual(current, control)
            cached = json.loads((evidence / 'cache/rows.json').read_text())
            cache_identity = json.loads((evidence / 'cache/identity.json').read_text())
            self.assertEqual(sha(evidence / 'cache/probe.cpp'), cache_identity['probe_sha256'])
            self.assertEqual(sha(evidence / 'cache/graph_backend.cpp'), cache_identity['cache_source_sha256'])
            self.assertEqual(sha(evidence / 'reproduction/graph_backend.r2.cpp'), cache_identity['r2_source_sha256'])
            self.assertEqual(identity['libraries']['cache']['sha256'], cache_identity['cache_library_sha256'])
            self.assertEqual(identity['libraries']['r2']['sha256'], cache_identity['r2_library_sha256'])
            self.assertEqual(len(cached), 18)
            self.assertEqual(sum(row['calls'] for row in cached), 750000)
            for mode in ['update', 'replace', 'reinsert']:
                for trial in range(3):
                    pair = [r for r in cached if r['workload'] == mode and r['trial'] == trial]
                    self.assertEqual({r['side'] for r in pair}, {'r2', 'cache'})
                    self.assertEqual(pair[0]['snapshots'], pair[1]['snapshots'])
                    self.assertTrue(all(r['loop_wall_ms'] > 0 and r['user_s'] >= 0 and r['system_s'] >= 0 for r in pair))
            self.assertIn('duplicate include', (evidence / 'cache/first-attempt-probe-tidy.log').read_text())
            for config in ['release', 'san']:
                self.assertIn('100% tests passed', (evidence / 'cache' / (config+'-ctest.log')).read_text())
            counts = json.loads((evidence / 'counts/rows.json').read_text())
            self.assertEqual({(r['workload'],r['side']) for r in counts},
                             {(m,s) for m in ['update','replace','reinsert'] for s in ['baseline','r2']})
            for row in counts:
                self.assertTrue(row['exact'])
                log = (evidence / 'counts' / (row['workload']+'-'+row['side']+'.log')).read_text()
                self.assertEqual(json.loads(next(line[7:] for line in log.splitlines() if line.startswith('COUNTS '))), row['counts'])
                self.assertLessEqual(row['counts']['repeated_distance'], row['counts']['repair_distance'])
                self.assertEqual(row['counts']['repairs'], 25000)
            report = (root / 'README.md').read_text()
            for row in rows:
                for operation in ['update','remove','add','query','save','load']:
                    item = row[operation]
                    self.assertIn(f"| {row['side']} | {operation} | {item['count']} | {item['p50_us']:.3f} | {item['p99_us']:.3f} | {item['call_only_ops_per_second']:.1f} | {item['wall_sum_ms']/1000:.3f} | {(item['user_sum_ms']+item['system_sum_ms'])/1000:.3f} |", report)
            for number in [0,25,50,75,100]:
                values = [r['checkpoints'][str(number)]['maintained_recall'] for r in rows]
                self.assertIn('| '+str(number)+' | '+' | '.join(f'{v:.4f}' for v in values)+' |', report)


class ProcessHighWaterTests(unittest.TestCase):
    def test_optional_current_process_high_water_and_legacy_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            AcceptanceTests().fixture(folder)
            verify_run(folder,1,1,1)
            path = folder / 'states.csv'
            with path.open() as stream:
                states = list(csv.DictReader(stream))
            def write(values):
                with path.open('w',newline='') as stream:
                    writer = csv.DictWriter(stream,fieldnames=list(states[0])+['proc_hwm_kib'])
                    writer.writeheader()
                    for row,value in zip(states,values):
                        writer.writerow(dict(row,proc_hwm_kib=value))
            write([110,120])
            verify_run(folder,1,1,1)
            for invalid in [[0,120],[110,0],[120,110],[80,120]]:
                write(invalid)
                with self.assertRaises(AssertionError):
                    verify_run(folder,1,1,1)

    @unittest.skipUnless(os.environ.get('VSAG_MIXED_PROBE'), 'Compiled C++ probe required')
    def test_current_process_high_water_excludes_large_launcher_history(self):
        binary = str(Path(os.environ['VSAG_MIXED_PROBE']).resolve())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'data'
            dataset.mkdir()
            for filename, count in [('base.fvecs',64),('queries.fvecs',5)]:
                with (dataset / filename).open('wb') as stream:
                    for identifier in range(count):
                        stream.write(struct.pack('<i17f',17,*[(identifier+1)*(d+1)*0.001 for d in range(17)]))
            snapshots = root / 'snapshots'
            snapshots.mkdir()
            output = root / 'output'
            launcher = '''import subprocess,sys
ballast=bytearray(96*1024*1024)
for offset in range(0,len(ballast),4096): ballast[offset]=1
subprocess.run(sys.argv[1:],check=True)
'''
            subprocess.run([sys.executable,'-c',launcher,binary,str(dataset),str(output),
                            'fp32','1','2',str(snapshots),'resident'],check=True,capture_output=True)
            verify_run(output,1,2,5)
            with (output / 'states.csv').open() as stream:
                states = list(csv.DictReader(stream))
            self.assertEqual(len(states),2)
            for state in states:
                self.assertGreaterEqual(int(state['peak_kib']),96*1024)
                self.assertGreater(int(state['proc_hwm_kib']),0)
                self.assertLess(int(state['proc_hwm_kib']),int(state['peak_kib'])/2)
                self.assertGreaterEqual(int(state['proc_hwm_kib'])+64,int(state['rss_kib']))


if __name__ == '__main__':
    unittest.main()
