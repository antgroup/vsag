# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import json
import hashlib
import io
import shutil
import statistics
import tarfile
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_pinned_full_acceptance
import build_fresh_full_reference

from verify_pinned_full_acceptance import verify
from run_pinned_full_acceptance import validate_fresh_receipt


class FullAuditTests(unittest.TestCase):

    def preflight_fixture(self, root):
        feature = root / 'feature'
        build = root / 'build'
        full = root / 'full'
        dataset = root / 'dataset'
        raw = root / 'raw'
        for path in [feature / 'include', build / 'include', full / 'include',
                     full / 'lib', build / 'lib', dataset, raw]:
            path.mkdir(parents=True, exist_ok=True)
        for path in [feature / 'LICENSE', build / 'libvsag-lite.so',
                     build / 'lib/libvsag-lite.so', full / 'lib/libvsag.so', full / 'include/vsag.h',
                     *[dataset / name for name in ['base.fvecs', 'queries.fvecs', 'groundtruth.ivecs']],
                     *[raw / name for name in ['base-100k.fbin', 'queries.fbin']]]:
            path.write_bytes(b'fixture')
        receipt = root / 'receipt.json'
        receipt.write_text(json.dumps(dict(
            full_source_head='dbb2fe0bec685ae7a343b1d2060d0553a98e20d2', fresh_core=True,
            ccache=False, prior_object_count=0, core_object_count=100, install=str(full),
            library_sha256='a' * 64, full_headers_sha256={'vsag.h': 'b' * 64},
            full_build_boundary='fixture', full_suite='not rerun')))
        return feature, build, full, dataset, raw, receipt

    def invoke_preflight(self, root, failure=None, existing=False):
        feature, build, full, dataset, raw, receipt = self.preflight_fixture(root)
        output = root / 'new-parent/output'
        if existing:
            output.mkdir(parents=True)
            (output / 'preserved.txt').write_bytes(b'original result')
        if failure == 'malformed receipt':
            receipt.write_text('{invalid')
        elif failure == 'invalid receipt':
            receipt.write_text('{}')
        elif failure == 'missing dataset':
            (dataset / 'base.fvecs').unlink()
        elif failure == 'empty dataset':
            (dataset / 'base.fvecs').write_bytes(b'')
        elif failure == 'missing raw':
            (raw / 'queries.fbin').unlink()
        elif failure == 'missing lite library':
            (build / 'libvsag-lite.so').unlink()
            (build / 'lib/libvsag-lite.so').unlink()
        elif failure == 'missing library':
            (full / 'lib/libvsag.so').unlink()
        elif failure == 'missing headers':
            (full / 'include/vsag.h').unlink()
            (full / 'include').rmdir()
        archive = root / 'source.tar.gz'
        archive.write_bytes(b'fixture archive')
        argv = ['runner', '--feature', str(feature), '--feature-build', str(build), '--source-repo', str(feature), '--full-install', str(full), '--dataset', 'Fixture=' + str(dataset), '--fresh-full-receipt', str(receipt), '--output', str(output)]

        if failure == 'duplicate dataset':
            argv += ['--dataset', 'Fixture=' + str(dataset)]
        elif failure == 'unsafe dataset label':
            argv[argv.index('--dataset') + 1] = '../escape=' + str(dataset)
        elif failure == 'legacy replay':
            offset = argv.index('--fresh-full-receipt')
            del argv[offset:offset + 2]
            argv += ['--reference-provenance', str(receipt)]

        def digest(path):
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if failure == 'wrong library' and path.name == 'libvsag.so':
                return '0' * 64
            if path.name == 'libvsag.so':
                return '9aeaa550703f116bef2959769d4f2f03da254b07c906ba941d8186224cc70c72' if failure == 'legacy replay' else 'a' * 64
            if path.name == 'vsag.h':
                return 'b' * 64
            if path.name == 'libvsag-lite.so':
                return '3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
            if path == archive:
                return '0' * 64 if failure == 'wrong archive' else '0c6697115a21dc210ab63cc0dfd659c53614d39ffc89115292765a1565ca9b54'
            return 'c' * 64

        def command(argv, **kwargs):
            if argv[:2] == ['git', 'rev-parse']:
                return 'wrong' if failure == 'wrong feature' else 'dbb2fe0bec685ae7a343b1d2060d0553a98e20d2'
            if argv[:2] == ['git', 'status']:
                return b' M tracked' if failure == 'dirty feature' else b''
            if argv[:2] == ['git', 'show']:
                if failure == 'missing helper':
                    raise subprocess.CalledProcessError(128, argv)
                if argv[2].endswith('/provenance.json'):
                    return receipt.read_bytes()
                return b'fixture historical source'
            return 'fixture tool version'

        # Stop before compilation or measurement; this fixture tests ordering only.
        with mock.patch.object(run_pinned_full_acceptance, 'sha', side_effect=digest), \
             mock.patch.object(run_pinned_full_acceptance.subprocess, 'check_output', side_effect=command), \
             mock.patch.object(run_pinned_full_acceptance.subprocess, 'run') as run, \
             mock.patch.object(run_pinned_full_acceptance.shutil, 'which', return_value=None if failure == 'missing tool' else '/tool'), \
             mock.patch.object(run_pinned_full_acceptance.tempfile, 'TemporaryDirectory', side_effect=RuntimeError('measurement boundary')), \
             mock.patch.object(sys, 'argv', argv):
            try:
                run_pinned_full_acceptance.main()
            finally:
                run.assert_not_called()
        return output

    def test_preflight_failures_do_not_create_output(self):
        for failure in ['wrong feature', 'dirty feature', 'missing dataset', 'empty dataset', 'missing lite library', 'malformed receipt', 'invalid receipt', 'missing library', 'wrong library', 'missing headers', 'missing helper', 'missing tool', 'duplicate dataset', 'unsafe dataset label']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises((AssertionError, FileNotFoundError, ValueError, KeyError,
                                        subprocess.CalledProcessError)):
                    self.invoke_preflight(root, failure)
                self.assertFalse((root / 'new-parent').exists())

    def test_legacy_replay_preflight_reaches_measurement_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, '^measurement boundary$'):
                self.invoke_preflight(root, 'legacy replay')
            self.assertEqual(list((root / 'new-parent/output').iterdir()), [])

    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                self.invoke_preflight(root, existing=True)
            output = root / 'new-parent/output'
            self.assertEqual([p.name for p in output.iterdir()], ['preserved.txt'])
            self.assertEqual((output / 'preserved.txt').read_bytes(), b'original result')

    def test_valid_preflight_reaches_measurement_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, '^measurement boundary$'):
                self.invoke_preflight(root)
            self.assertTrue((root / 'new-parent/output').is_dir())

    def test_fresh_receipt_guards(self):
        # A current source needs both fresh receipts; a historical library hash
        # must never be relabeled by changing only the command-line commit.
        receipt = dict(full_source_head='feature', fresh_core=True, ccache=False,
            prior_object_count=0, core_object_count=100, install='/tmp/full-reference',
            library_sha256='a' * 64, full_headers_sha256={'vsag.h': 'b' * 64})
        validate_fresh_receipt(receipt, 'feature', Path('/tmp/full-reference'))
        for field, invalid in [('full_source_head', 'historical'), ('fresh_core', False),
                ('ccache', True), ('prior_object_count', 1), ('core_object_count', 1),
                ('install', '/tmp/another-reference'), ('library_sha256', 'short'),
                ('full_headers_sha256', {})]:
            with self.subTest(field=field):
                altered = dict(receipt, **{field: invalid})
                with self.assertRaises(AssertionError):
                    validate_fresh_receipt(altered, 'feature', Path('/tmp/full-reference'))

    def test_nonhex_receipt_digest_rejected(self):
        with self.assertRaises(ValueError):
            validate_fresh_receipt(dict(full_source_head='feature', fresh_core=True,
                ccache=False, prior_object_count=0, core_object_count=100,
                install='/tmp/full-reference', library_sha256='z' * 64), 'feature', Path('/tmp/full-reference'))

    def fixture(self, directory, ef=512, digest='full'):
        root = Path(directory)
        (root / 'measurements').mkdir()
        (root / 'provenance.json').write_text(json.dumps(
            dict(full_library_sha256='full', lite_library_sha256='lite')))
        (root / 'manifest.json').write_text('{}')
        (root / 'measurements/identity.json').write_text(json.dumps(dict(
            arguments=dict(ef=ef, full_ef=512), reference_library_sha256=digest,
            lite_library_sha256='lite')))
        return root

    def test_unequal_query_budget_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, ef=128)
            with self.assertRaises(AssertionError):
                verify(root)

    def test_wrong_reference_library_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, digest='old-lite')
            with self.assertRaises(AssertionError):
                verify(root)

    def test_manifest_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            (root / 'manifest.json').write_text(json.dumps({'provenance.json': 'incorrect'}))
            with self.assertRaises(AssertionError):
                verify(root)


class CurrentPublishedFullTests(unittest.TestCase):
    def test_complete_current_evidence_and_report_cells(self):
        root = Path(__file__).parent
        result = root / 'results/full-comparison'
        archive = result / 'raw.tar.gz'
        receipt = json.loads((result / 'archive.json').read_text())
        self.assertEqual(archive.stat().st_size, receipt['bytes'])
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), receipt['sha256'])
        with tarfile.open(archive, 'r:gz') as outer:
            manifest = json.load(outer.extractfile('manifest.json'))
            self.assertEqual(set(outer.getnames()), set(manifest) | {'manifest.json'})
            self.assertEqual(len(outer.getmembers()), receipt['members'])
            for name, digest in manifest.items():
                self.assertEqual(hashlib.sha256(outer.extractfile(name).read()).hexdigest(), digest)
            summary = json.load(outer.extractfile('full-comparison-summary.json'))
            freeze = json.load(outer.extractfile('full-matrix-freeze.json'))
            full_receipt = json.load(outer.extractfile('full-build-record/receipt.json'))
            audits = json.load(outer.extractfile('audits.json'))
            package = json.load(outer.extractfile('package/report.json'))
            inner_bytes = outer.extractfile('full/raw.tar.gz').read()
            inner_receipt = json.load(outer.extractfile('full/archive.json'))
            self.assertEqual(len(inner_bytes), inner_receipt['bytes'])
            self.assertEqual(hashlib.sha256(inner_bytes).hexdigest(), inner_receipt['sha256'])
            with tarfile.open(fileobj=io.BytesIO(inner_bytes), mode='r:gz') as inner:
                provenance = json.load(inner.extractfile('provenance.json'))
                rows = json.load(inner.extractfile('measurements/rows.json'))
                lite_receipt = json.load(inner.extractfile('lite-build-source-receipt.json'))
            self.assertEqual(full_receipt['full_source_head'], freeze['feature_sha'])
            self.assertEqual(full_receipt['compiler_profile'], 'aligned-o3-overlay')
            self.assertTrue(full_receipt['fresh_core'])
            self.assertFalse(full_receipt['ccache'])
            self.assertEqual(full_receipt['prior_object_count'], 0)
            self.assertGreater(full_receipt['core_object_count'], 50)
            self.assertEqual(freeze['full_library'], full_receipt['library_sha256'])
            self.assertEqual(provenance['full_library_sha256'], full_receipt['library_sha256'])
            self.assertEqual(provenance['feature_sha'], lite_receipt['source']['feature_sha'])
            self.assertEqual(provenance['lite_library_sha256'], lite_receipt['library_sha256'])
            for name in ['dispatch-runtime-lite.stdout', 'dispatch-runtime-full.stdout']:
                dispatch = outer.extractfile(name).read().decode()
                self.assertIn('avx512', dispatch)
                for dimension in [128, 960, 768, 129]:
                    self.assertIn('fixture\t' + str(dimension) + '\t', dispatch)
        self.assertEqual(len(rows), 54)
        self.assertEqual(sum(len(row['loads']) for row in rows), 162)
        self.assertEqual(audits['full']['query_results'], 54000)
        self.assertEqual(audits['package']['relocated_smokes'], 6)
        self.assertEqual(len(summary), 9)
        text = (result / 'README.md').read_text()
        main = (root / 'ACCEPTANCE_REPORT.md').read_text()
        labels = dict(fp32='FP32', fp16='FP16', rabitq8='RaBitQ8')
        for cell in summary.values():
            for side, values in cell['sides'].items():
                selected = [r for r in rows if (r['dataset'], r['storage'], r['name']) ==
                            (cell['dataset'], cell['storage'], side)]
                self.assertEqual(len(selected), 3)
                for key, samples in values['values'].items():
                    loader_metric = key in {'load_ms', 'loaded_rss_kib', 'process_peak_rss_kib',
                                            'before_create_rss_kib', 'before_load_rss_kib'}
                    actual = ([float(s[key]) for r in selected for s in r['loads']] if loader_metric
                              else [float(r['build'][key]) for r in selected])
                    self.assertEqual(samples, actual)
                    self.assertEqual(values['metrics'][key], statistics.median(actual))
                increments = [float(s['loaded_rss_kib']) - float(s['before_load_rss_kib'])
                              for r in selected for s in r['loads']]
                self.assertEqual(increments, values['incremental_rss_values'])
                self.assertEqual(statistics.median(increments), values['metrics']['incremental_rss_kib'])
            lite = cell['sides']['lite']['metrics']
            full = cell['sides']['full']['metrics']
            prefix = f"| {cell['dataset']} | {labels[cell['storage']]} | "
            recall_pp = (lite['recall_at_k'] - full['recall_at_k']) * 100
            p50_percent = (lite['search_p50_us'] / full['search_p50_us'] - 1) * 100
            self.assertEqual(cell['recall_difference_pp'], recall_pp)
            self.assertEqual(cell['p50_change_percent'], p50_percent)
            query = (prefix + f"{lite['search_p50_us']:.3f} | {full['search_p50_us']:.3f} | "
                     f"{lite['search_p99_us']:.3f} | {full['search_p99_us']:.3f} | "
                     f"{lite['recall_at_k']:.4f} | {full['recall_at_k']:.4f} | {recall_pp:.2f} |")
            self.assertIn(query, text)
            self.assertIn(query, main)
            self.assertIn(prefix + f"{p50_percent:.2f}% | {lite['build_ms']:.3f} | {full['build_ms']:.3f} |", text)
            for keys, scale in [(['load_ms', 'save_ms'], 1),
                                 (['snapshot_bytes'], 1048576),
                                 (['loaded_rss_kib', 'incremental_rss_kib'], 1024)]:
                values = []
                for key in keys:
                    values += [lite[key]/scale, full[key]/scale]
                if keys == ['snapshot_bytes']:
                    values += [lite['external_peak_rss_kib']/1024, full['external_peak_rss_kib']/1024]
                self.assertIn(prefix + ' | '.join(f'{value:.3f}' for value in values) + ' |', text)
        for title, field, percent in [('未压缩载荷', 'payload_bytes', 'payload_reduction_percent'),
                                      ('gzip归档', 'tar_gzip_bytes', 'compressed_reduction_percent')]:
            lite = package['packages']['lite'][field]
            full = package['packages']['full'][field]
            reduction = (1 - lite/full)*100
            self.assertAlmostEqual(package[percent], reduction)
            self.assertIn(f'| {title} | {lite} | {full} | {reduction:.2f}% |', text)


class CurrentSourceTests(unittest.TestCase):
    def test_actual_cmake_soname_alias_chain_survives_sdk_copy(self):
        with tempfile.TemporaryDirectory(dir='/dev/shm') as directory:
            root = Path(directory)
            source, build = root / 'source', root / 'build'
            source.mkdir()
            (source / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.18)\nproject(alias_fixture LANGUAGES CXX)\n'
                'add_library(vsag SHARED library.cpp)\n'
                'set_target_properties(vsag PROPERTIES VERSION 0.18.0 SOVERSION 0.18)\n')
            (source / 'library.cpp').write_text('extern "C" int marker(){return 7;}\n')
            subprocess.run(['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja'],
                           check=True, capture_output=True, text=True)
            subprocess.run(['cmake', '--build', str(build)], check=True, capture_output=True, text=True)
            library = build / 'libvsag.so.0.18.0'
            original = library.read_bytes()
            consumer = source / 'consumer.cpp'
            consumer.write_text('extern "C" int marker(); int main(){return marker()==7?0:1;}\n')
            broken, fixed = root / 'broken', root / 'fixed'
            for destination in [broken, fixed]:
                destination.mkdir()
                shutil.copyfile(library, destination / library.name)
            for alias in ['libvsag.so', 'libvsag.so.0.0']:
                (broken / alias).symlink_to(library.name)
            def consume(destination):
                binary = root / (destination.name + '-consumer')
                subprocess.run(['c++', str(consumer), '-L' + str(destination),
                                '-Wl,-rpath,' + str(destination), '-lvsag', '-o', str(binary)],
                               check=True, capture_output=True, text=True)
                env = {key:value for key,value in __import__('os').environ.items()
                       if key not in ['LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'LD_DEBUG']}
                return subprocess.run([str(binary)], env=env, capture_output=True, text=True)
            control = consume(broken)
            self.assertNotEqual(control.returncode, 0)
            self.assertIn('libvsag.so.0.18', control.stderr)
            aliases = build_fresh_full_reference.preserve_library_aliases(library, fixed)
            self.assertEqual(aliases['libvsag.so.0.18'], 'libvsag.so.0.18.0')
            self.assertTrue((fixed / 'libvsag.so').is_symlink())
            self.assertEqual(consume(fixed).returncode, 0)
            self.assertEqual(library.read_bytes(), original)

    def test_full_alias_validation_precedes_any_alias_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / 'source', root / 'install'
            source.mkdir()
            destination.mkdir()
            library = source / 'libvsag.so.0.18.0'
            library.write_bytes(b'owned library fixture')
            (source / 'libvsag.so').symlink_to(library.name)
            (root / 'foreign').write_bytes(b'foreign')
            (source / 'libvsag.so.bad').symlink_to('../foreign')
            with self.assertRaises(ValueError):
                build_fresh_full_reference.preserve_library_aliases(library, destination)
            self.assertEqual(list(destination.iterdir()), [])

    def test_strict_fp_supports_historical_and_current_compiler_profiles(self):
        transform = build_fresh_full_reference.strict_fp_config
        legacy = 'vsag_add_compile_flag (-g)\nvsag_add_compile_flag (-Ofast)\n'
        self.assertEqual(transform(legacy),
                         'vsag_add_compile_flag (-g)\nvsag_add_compile_flag (-O3)\n')
        with self.assertRaises(ValueError):
            transform('unknown compiler profile\n')
        source = Path(__file__).resolve().parents[2] / 'cmake/VSAGCompilerConfig.cmake'
        original = source.read_text()
        modified = transform(original)
        self.assertEqual(source.read_text(), original)
        self.assertIn('set (CMAKE_CXX_STANDARD 20)', modified)
        begin = modified.index('if (CMAKE_BUILD_TYPE STREQUAL "Debug")')
        end = modified.index('if (ENABLE_THIN_LTO)', begin)
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'flags.cmake'
            for compiler in ['GNU', 'Clang']:
                with self.subTest(compiler=compiler):
                    script.write_text(
                        'cmake_minimum_required(VERSION 3.18)\n'
                        'function(vsag_add_compile_flag flag)\n'
                        '  set(flags ${flags} ${flag} PARENT_SCOPE)\n'
                        'endfunction()\n'
                        'function(vsag_add_c_compile_flag flag)\nendfunction()\n'
                        'set(CMAKE_BUILD_TYPE Release)\n'
                        'set(CMAKE_CXX_COMPILER_ID ' + compiler + ')\n'
                        'set(APPLE FALSE)\n' + modified[begin:end] +
                        '\nif(NOT "-O3" IN_LIST flags OR "-Ofast" IN_LIST flags OR "-ffast-math" IN_LIST flags)\n'
                        '  message(FATAL_ERROR "Unexpected strict flags: ${flags}")\n'
                        'endif()\n')
                    subprocess.run(['cmake', '-P', str(script)], check=True,
                                   capture_output=True, text=True)


    def test_fresh_full_ignores_inherited_compiler_launchers(self):
        from contextlib import nullcontext
        import os
        with tempfile.TemporaryDirectory(dir='/dev/shm') as directory:
            root = Path(directory)
            source, deps = root / 'source', root / 'deps'
            source.mkdir()
            for folder, sentinel in [('cpuinfo-src', 'CMakeLists.txt'),
                    ('nlohmann_json-src', 'CMakeLists.txt'), ('roaringbitmap-src', 'CMakeLists.txt'),
                    ('thread_pool-src', 'ThreadPool.h'), ('tsl-src', 'CMakeLists.txt')]:
                (deps / folder).mkdir(parents=True)
                (deps / folder / sentinel).write_text('fixture')
            archive = root / 'antlr.tar.gz'
            archive.write_bytes(b'fixture archive')
            launcher = root / 'cache-launcher'
            launcher.write_text('#!/bin/sh\n: > "$0.used"\nexit 91\n')
            launcher.chmod(0o755)
            captured = {}
            def inspect_configure(command, **kwargs):
                captured.update(command=command, env=kwargs['env'])
                raise RuntimeError('configure boundary')
            args = ['builder', '--source', str(source), '--source-sha', 'a' * 40,
                    '--deps', str(deps), '--antlr-archive', str(archive),
                    '--install', str(root / 'install'), '--output', str(root / 'output')]
            with mock.patch.dict(os.environ, {'CMAKE_C_COMPILER_LAUNCHER': str(launcher),
                    'CMAKE_CXX_COMPILER_LAUNCHER': str(launcher), 'CCACHE_DISABLE': '0'}), \
                 mock.patch.object(sys, 'argv', args), \
                 mock.patch.object(build_fresh_full_reference, 'validate_source_checkout', return_value={}), \
                 mock.patch.object(build_fresh_full_reference.hashlib, 'md5') as digest, \
                 mock.patch.object(build_fresh_full_reference, 'cached_archive_url', return_value=nullcontext('http://127.0.0.1/fixture')), \
                 mock.patch.object(build_fresh_full_reference.subprocess, 'run', side_effect=inspect_configure):
                digest.return_value.hexdigest.return_value = '3b75610fc8a827119258cba09a068be5'
                with self.assertRaisesRegex(RuntimeError, '^configure boundary$'):
                    build_fresh_full_reference.main()
            # Use the actual tool's captured environment/launcher settings in a real C/C++ build.
            (source / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.18)\nproject(cache_fixture LANGUAGES C CXX)\nadd_executable(probe main.cpp helper.c)\n')
            (source / 'main.cpp').write_text('extern "C" int marker(); int main(){return marker()==7?0:1;}\n')
            (source / 'helper.c').write_text('int marker(void){return 7;}\n')
            flags = [arg for arg in captured['command'] if arg.startswith(('-DCMAKE_C_COMPILER_LAUNCHER=', '-DCMAKE_CXX_COMPILER_LAUNCHER='))]
            build = root / 'actual-build'
            subprocess.run(['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja', *flags],
                           env=captured['env'], check=True, capture_output=True, text=True)
            subprocess.run(['cmake', '--build', str(build)], env=captured['env'],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(build / 'probe')], check=True)
            self.assertFalse(Path(str(launcher) + '.used').exists())
            self.assertEqual(captured['env']['CCACHE_DISABLE'], '1')
            self.assertEqual(set(flags), {'-DCMAKE_C_COMPILER_LAUNCHER=', '-DCMAKE_CXX_COMPILER_LAUNCHER='})

    def test_full_source_identity_checks_clean_tracked_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/current.cpp').write_text('// source\n')
            (root / 'CMakeLists.txt').write_text('# fixture\n')
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', 'add', '.'], cwd=root, check=True)
            subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                            '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture'], cwd=root, check=True)
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
            identity = build_fresh_full_reference.validate_source_checkout(root, head)
            self.assertEqual(set(identity), {'src/current.cpp', 'CMakeLists.txt'})
            for invalid in ['0' * 40, 'main', head.upper()]:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    build_fresh_full_reference.validate_source_checkout(root, invalid)
            (root / 'untracked.h').write_text('// overlay\n')
            with self.assertRaises(ValueError):
                build_fresh_full_reference.validate_source_checkout(root, head)
            (root / 'untracked.h').unlink()
            (root / 'src/current.cpp').write_text('// changed\n')
            with self.assertRaises(ValueError):
                build_fresh_full_reference.validate_source_checkout(root, head)

    def test_current_feature_cannot_use_unreceipted_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = FullAuditTests()
            feature, build, full, dataset, raw, receipt = helper.preflight_fixture(root)
            output = root / 'new/output'
            args = ['runner', '--feature', str(feature), '--feature-sha', 'c' * 40,
                    '--feature-build', str(build), '--source-repo', str(feature),
                    '--full-install', str(full), '--dataset', 'Fixture=' + str(dataset),
                    '--output', str(output)]
            with mock.patch.object(sys, 'argv', args), self.assertRaisesRegex(ValueError, 'fresh Full'):
                run_pinned_full_acceptance.main()
            self.assertFalse(output.parent.exists())

    def test_current_lite_receipt_is_checked_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = FullAuditTests()
            feature, build, full, dataset, raw, receipt = helper.preflight_fixture(root)
            output = root / 'new/output'
            args = ['runner', '--feature', str(feature), '--feature-sha', 'c' * 40,
                    '--feature-build', str(build), '--source-repo', str(feature),
                    '--full-install', str(full), '--dataset', 'Fixture=' + str(dataset),
                    '--fresh-full-receipt', str(receipt), '--output', str(output)]
            with mock.patch.object(sys, 'argv', args), self.assertRaisesRegex(ValueError, 'build-source receipt'):
                run_pinned_full_acceptance.main()
            self.assertFalse(output.parent.exists())


if __name__ == '__main__':
    unittest.main()
