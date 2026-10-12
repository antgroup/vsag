#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Fresh consumers, current Lite versus pinned installed native Full, equal query ef."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

from run_mixed_cpu_acceptance import sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=Path, required=True)
    parser.add_argument('--feature-build', type=Path, required=True)
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--source-ref', default='a62e62e08db3f0d7da2e4439785feed6fcf90257')
    parser.add_argument('--full-install', type=Path, required=True)
    parser.add_argument('--dataset', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    expected_feature = 'dbb2fe0bec685ae7a343b1d2060d0553a98e20d2'
    expected_full = '9aeaa550703f116bef2959769d4f2f03da254b07c906ba941d8186224cc70c72'
    feature_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.feature, text=True).strip()
    assert feature_sha == expected_feature
    assert not subprocess.check_output(['git', 'status', '--porcelain', '-uno'], cwd=args.feature)
    full_library = args.full_install / 'lib/libvsag.so'
    assert sha(full_library) == expected_full
    old_provenance = json.loads(subprocess.check_output(['git', 'show', args.source_ref +
        ':lite/benchmark/results/current-full-10k600-20261011/provenance.json'], cwd=args.source_repo))
    for name, digest in old_provenance['full_headers_sha256'].items():
        assert sha(args.full_install / 'include' / name) == digest, name
    meta = dict(feature_sha=feature_sha, lite_library_sha256=sha(args.feature_build / 'libvsag-lite.so'),
        full_source_sha=old_provenance['full_source_head'], full_library_sha256=expected_full,
        full_boundary=old_provenance['full_build_boundary'], full_headers=old_provenance['full_headers_sha256'],
        full_suite='not rerun', query_ef_both=512, trials=3, loads_per_builder=3,
        compiler=subprocess.check_output(['c++', '--version'], text=True),
        helpers_source_ref=args.source_ref, helper_source_hashes={}, compile_commands=[])
    env = {k: v for k, v in os.environ.items() if not k.startswith('VSAG_')}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='vsag-equal-full-', dir='/dev/shm') as directory:
        scratch = Path(directory)
        lite = scratch / 'lite'
        lite.mkdir()
        (lite / 'libvsag-lite.so').symlink_to((args.feature_build / 'libvsag-lite.so').resolve())
        names = ['graph_crud_quality.cpp', 'load_memory.cpp', 'full_rabitq_dataset_main.cpp',
                 'run_aligned_comparison.py', 'measure_elf_closure.py']
        for name in names:
            source = subprocess.check_output(['git', 'show', args.source_ref + ':lite/benchmark/' + name],
                                             cwd=args.source_repo)
            path = scratch / name
            path.write_bytes(source)
            meta['helper_source_hashes'][name] = sha(path)
            (args.output / ('source-' + name)).write_bytes(source)
        targets = [(scratch / 'graph_crud_quality.cpp', lite / 'lite_graph_crud_quality', False, False),
                   (scratch / 'load_memory.cpp', lite / 'lite_load_memory', False, False),
                   (scratch / 'full_rabitq_dataset_main.cpp', scratch / 'full_builder', True, False),
                   (scratch / 'load_memory.cpp', scratch / 'full_loader', True, True)]
        for source, target, full, loader in targets:
            command = ['c++', '-std=c++17', '-O3', '-DNDEBUG', '-Wall', '-Wextra', '-Wpedantic']
            if loader:
                command.append('-DVSAG_BENCH_FULL')
            prefix = args.full_install if full else args.feature
            library_dir = args.full_install / 'lib' if full else args.feature_build
            command += ['-I' + str(prefix / 'include'), str(source), '-L' + str(library_dir),
                        '-Wl,-rpath,' + str(library_dir), '-lvsag' if full else '-lvsag-lite', '-o', str(target)]
            with (args.output / (target.name + '.compile.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=log, env=env, check=True)
            meta['compile_commands'].append(dict(argv=command, binary_sha256=sha(target)))
        command = ['python3', str(scratch / 'run_aligned_comparison.py'), '--lite-build', str(lite),
            '--full-library', str(full_library), '--full-builder', str(scratch / 'full_builder'),
            '--full-loader', str(scratch / 'full_loader'), '--output', str(args.output / 'measurements'),
            '--scratch', str(scratch), '--trials', '3', '--loads', '3', '--ef', '512', '--full-ef', '512', '--cpu', '0']
        for value in args.dataset:
            command += ['--dataset', value]
        meta['measurement_command'] = command
        write_json(args.output / 'provenance.json', meta)
        subprocess.run(command, env=env, check=True)
        assert sha(full_library) == expected_full
        assert sha(args.feature_build / 'libvsag-lite.so') == meta['lite_library_sha256']
        subprocess.run(['python3', str(scratch / 'measure_elf_closure.py'), '--lite',
            str(args.feature_build / 'libvsag-lite.so'), '--full', str(full_library),
            '--output', str(args.output / 'elf'), '--scratch', str(scratch)], env=env, check=True)
    manifest = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    write_json(args.output / 'manifest.json', manifest)
    with tarfile.open(args.output / 'raw.tar.gz', 'w:gz') as stream:
        for name in sorted(list(manifest) + ['manifest.json']):
            stream.add(args.output / name, arcname=name, recursive=False)
    write_json(args.output / 'archive.json', dict(sha256=sha(args.output / 'raw.tar.gz'),
        bytes=(args.output / 'raw.tar.gz').stat().st_size, members=len(manifest) + 1, originals_preserved=True))
    print('Complete same-config initial-index comparison; not same-quality or fresh Full library')


if __name__ == '__main__':
    main()
