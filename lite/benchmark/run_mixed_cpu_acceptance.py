#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Measure an explicitly pinned #2904 library from the #2926 evidence branch."""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def percentile(values, fraction):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Invalid latency samples')
    return sorted(values)[max(1, math.ceil(len(values) * fraction)) - 1]


def verify_run(output, rounds, cycles, queries):
    with (output / 'samples.csv').open() as stream:
        samples = list(csv.DictReader(stream))
    result = {}
    expected = dict(build=1, add=rounds * cycles, remove=rounds * cycles,
                    update=rounds * cycles, query=rounds * cycles * 3,
                    save=rounds, load=rounds)
    assert len(samples) == sum(expected.values())
    for operation, count in expected.items():
        selected = [row for row in samples if row['operation'] == operation]
        assert len(selected) == count, operation
        walls = [float(row['wall_us']) for row in selected]
        total = sum(walls)
        result[operation] = dict(count=count, p50_us=percentile(walls, .5),
            p99_us=percentile(walls, .99), wall_sum_ms=total / 1000,
            call_only_ops_per_second=count * 1e6 / total,
            user_sum_ms=sum(float(row['user_us']) for row in selected) / 1000,
            system_sum_ms=sum(float(row['system_us']) for row in selected) / 1000)
        for row in selected:
            assert all(float(row[key]) >= 0 for key in
                       ['user_us', 'system_us', 'voluntary', 'involuntary', 'peak_kib'])
    with (output / 'quality.csv').open() as stream:
        quality = list(csv.DictReader(stream))
    assert len(quality) == queries * 2
    result['recall'] = {}
    for round_number in [0, rounds]:
        selected = [row for row in quality if int(row['round']) == round_number]
        assert len(selected) == queries
        assert {int(row['query']) for row in selected} == set(range(queries))
        assert all(0 <= int(row['hits']) <= int(row['k']) == 10 for row in selected)
        result['recall'][str(round_number)] = sum(int(row['hits']) for row in selected) / (queries * 10)
    with (output / 'states.csv').open() as stream:
        states = list(csv.DictReader(stream))
    assert len(states) == rounds + 1
    count = int(states[0]['count'])
    assert all(int(row['count']) == count for row in states)
    assert [int(row['round']) for row in states] == list(range(rounds + 1))
    result['states'] = states
    return result



BUILD_RECEIPT = 'build-source-receipt.json'


def feature_identity(feature, expected):
    feature = feature.resolve(strict=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=feature, text=True).strip()
    if head != expected or subprocess.check_output(['git', 'status', '--porcelain'], cwd=feature):
        raise ValueError('Feature checkout must be clean and match --feature-sha')
    names = subprocess.check_output(['git', 'ls-files', '-z', '--', 'src/lite', 'src/simd',
        'include/vsag', 'lite/CMakeLists.txt'], cwd=feature).decode().split('\0')
    fingerprints = {name: sha(feature / name) for name in names if name}
    if 'lite/CMakeLists.txt' not in fingerprints or 'src/lite/index.cpp' not in fingerprints:
        raise ValueError('Missing tracked Lite build inputs')
    return dict(feature_repo=str(feature), feature_sha=head, source_sha256=fingerprints)


def build_metadata(feature, build):
    cache = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line and ':' in line.split('=', 1)[0]:
            key, value = line.split('=', 1)
            cache[key.split(':', 1)[0]] = value
    expected = {'CMAKE_PROJECT_NAME': 'vsag_lite', 'CMAKE_BUILD_TYPE': 'Release',
                'ENABLE_RABITQ_LITE_BACKEND': 'ON', 'CMAKE_EXPORT_COMPILE_COMMANDS': 'ON'}
    for name, value in expected.items():
        if cache.get(name) != value:
            raise ValueError('Unsupported Lite build configuration: ' + name)
    for name, path in [('CMAKE_HOME_DIRECTORY', feature / 'lite'), ('CMAKE_CACHEFILE_DIR', build)]:
        if name not in cache or Path(cache[name]).resolve() != path.resolve():
            raise ValueError('Build/source directory mismatch: ' + name)
    commands = json.loads((build / 'compile_commands.json').read_text())
    compiled = set()
    for entry in commands:
        command = entry.get('command', ' '.join(entry.get('arguments', [])))
        if 'CMakeFiles/vsag_lite_objects.dir/' not in command:
            continue
        directory = Path(entry['directory']).resolve()
        path = (directory / entry['file']).resolve()
        if directory != build.resolve() or (feature / 'src/lite').resolve() not in path.parents:
            raise ValueError('Foreign Lite translation unit in compile_commands.json')
        compiled.add(str(path.relative_to(feature.resolve())))
    required = {'src/lite/index.cpp', 'src/lite/bruteforce_backend.cpp',
                'src/lite/graph_backend.cpp', 'src/lite/rabitq_backend.cpp'}
    if not required <= compiled:
        raise ValueError('Missing Lite/RaBitQ production compilation entries')
    return dict(cache_sha256=sha(build / 'CMakeCache.txt'),
                compile_commands_sha256=sha(build / 'compile_commands.json'),
                compiled_sources=sorted(compiled))


def clean_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith('VSAG_') and k not in
           ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT', 'LD_DEBUG']}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def prepare_build(feature, expected, build):
    source = feature_identity(feature, expected)
    if build.exists():
        raise FileExistsError('Use a new build directory; existing builds are never cleaned: ' + str(build))
    commands = [
        ['cmake', '-S', str(feature / 'lite'), '-B', str(build), '-G', 'Ninja',
         '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
         '-DENABLE_RABITQ_LITE_BACKEND=ON', '-DENABLE_TESTS=OFF', '-DENABLE_BENCHMARKS=OFF',
         '-DCMAKE_CXX_COMPILER_LAUNCHER='],
        ['cmake', '--build', str(build), '--target', 'vsag_lite', '-j2']]
    env = clean_environment()
    env['CCACHE_DISABLE'] = '1'
    build.mkdir(parents=True, exist_ok=False)
    with (build / 'prepare-build.log').open('w') as log:
        for command in commands:
            subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
    if feature_identity(feature, expected) != source:
        raise ValueError('Feature sources changed during build')
    receipt = dict(schema=1, source=source, build=str(build),
                   library_sha256=sha(build / 'libvsag-lite.so'),
                   metadata=build_metadata(feature, build), commands=commands,
                   boundary='Fresh standalone Lite build; local provenance, not signed attestation')
    write_json(build / BUILD_RECEIPT, receipt)
    return receipt


def validate_build(feature, expected, build):
    path = build / BUILD_RECEIPT
    if not path.is_file():
        raise ValueError('Missing build-source receipt; use --prepare-build with a new --build directory')
    receipt = json.loads(path.read_text())
    if receipt.get('schema') != 1 or receipt.get('build') != str(build):
        raise ValueError('Unsupported or copied build-source receipt')
    if receipt['source'] != feature_identity(feature, expected):
        raise ValueError('Build-source receipt does not match the feature checkout')
    if receipt['metadata'] != build_metadata(feature, build):
        raise ValueError('Build metadata changed after preparation')
    if receipt['library_sha256'] != sha(build / 'libvsag-lite.so'):
        raise ValueError('Lite library changed after preparation')
    return receipt


def validate_loaded_library(text, library, expected):
    matches = [line.split('=>', 1)[1].rsplit(' (', 1)[0].strip()
               for line in text.splitlines() if line.strip().startswith('libvsag-lite.so =>')]
    if len(matches) != 1 or Path(matches[0]).resolve() != library.resolve() or sha(library) != expected:
        raise ValueError('Probe is not linked to the prepared Lite library')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-repo', type=Path, required=True)
    parser.add_argument('--feature-sha', required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--prepare-build', action='store_true',
                        help='Create a new standalone Lite build and source/library receipt; no measurement')
    parser.add_argument('--dataset', action='append', help='name=path')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--cycles', type=int, default=100)
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--resident', action='store_true',
                        help='Keep the original index between rounds; reload only checks persistence')
    args = parser.parse_args()
    args.feature_repo = args.feature_repo.resolve(strict=True)
    args.build = args.build.resolve()
    if args.prepare_build:
        if args.dataset or args.output:
            parser.error('--prepare-build cannot be combined with --dataset or --output')
        prepare_build(args.feature_repo, args.feature_sha, args.build)
        print('Prepared source-bound Lite build:', args.build, flush=True)
        return
    if not args.dataset or args.output is None:
        parser.error('Measurement requires --dataset and --output')
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.trials, args.rounds, args.cycles) <= 0 or args.cpu not in os.sched_getaffinity(0):
        parser.error('Positive trial/round/cycle counts and an available --cpu are required')
    receipt = validate_build(args.feature_repo, args.feature_sha, args.build)
    for tool in ['c++', 'ldd', 'taskset']:
        if not shutil.which(tool):
            parser.error('Missing required tool: ' + tool)
    if not os.access('/usr/bin/time', os.X_OK):
        parser.error('Missing executable /usr/bin/time')
    source = Path(__file__).with_name('mixed_cpu_acceptance.cpp')
    env = clean_environment()
    env['LD_LIBRARY_PATH'] = str(args.build)
    library = args.build / 'libvsag-lite.so'
    identity = dict(feature_sha=args.feature_sha, library_sha256=sha(library),
        build_source_receipt_sha256=sha(args.build / BUILD_RECEIPT),
        build=str(args.build.resolve()), benchmark_source_sha256=sha(source),
        benchmark_branch=subprocess.check_output(['git', 'branch', '--show-current'],
            cwd=source.parent, text=True).strip(),
        benchmark_parent=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=source.parent, text=True).strip(),
        configuration=dict(rounds=args.rounds, cycles=args.cycles, trials=args.trials,
            degree=16, construction_ef=128, query_ef=512, k=10, cpu=args.cpu,
            read_to_mutation='1:1', externally_serialized=True,
            resident=args.resident, measurement_buffer='pre-sized/touched then cleared'),
        feature_sources={str(p.relative_to(args.feature_repo)): sha(p)
            for p in sorted((args.feature_repo / 'src/lite').glob('*')) if p.is_file()},
        platform=subprocess.check_output(['uname', '-a'], text=True).strip(),
        cpu=subprocess.check_output(['lscpu'], text=True),
        compiler=subprocess.check_output(['c++', '--version'], text=True),
        limits=['Fixed-round real-change interleaved regression, not hours-long endurance',
            'Repeated use of the supplied query cohort, not new independent queries',
            'FP64 oracle at initial/final state, mixed queries checked for live IDs/order/uniqueness',
            'Per-call getrusage instrumentation adds overhead and has microsecond CPU resolution',
            'Call-only throughput excludes validation, persistence and bookkeeping',
            'Process peak includes input/oracle buffers, old+loaded index and truth computation',
            'Save/load are same-process uncontrolled-cache, not fresh or cold I/O',
            'Not a Full comparison or a complete package measurement'])
    identity['inputs'] = {}
    datasets = []
    for value in args.dataset:
        name, directory = value.split('=', 1)
        if not name.isalnum() or name in identity['inputs']:
            raise ValueError('Dataset names must be unique and alphanumeric')
        path = Path(directory).resolve()
        for filename in ['base.fvecs', 'queries.fvecs']:
            item = path / filename
            if not item.is_file() or item.stat().st_size == 0:
                raise ValueError('Missing or empty dataset input: ' + str(item))
        identity['inputs'][name] = {filename: dict(path=str(path / filename), sha256=sha(path / filename))
                                    for filename in ['base.fvecs', 'queries.fvecs']}
        datasets.append((name, path))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'source.cpp').write_bytes(source.read_bytes())
    for name in ['CMakeCache.txt', 'compile_commands.json', BUILD_RECEIPT]:
        (args.output / name).write_bytes((args.build / name).read_bytes())
    write_json(args.output / 'identity.json', identity)
    runs = []
    with tempfile.TemporaryDirectory(prefix='vsag-mixed2926-', dir='/dev/shm') as temporary:
        scratch = Path(temporary)
        binary = scratch / 'probe'
        command = ['c++', '-std=c++17', '-O3', '-DNDEBUG', '-Wall', '-Wextra', '-Wpedantic',
            '-I' + str(args.feature_repo / 'include'), str(source), '-L' + str(args.build),
            '-Wl,-rpath,' + str(args.build), '-lvsag-lite', '-o', str(binary)]
        write_json(args.output / 'compile.json', command)
        with (args.output / 'compile.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
        identity['binary_sha256'] = sha(binary)
        identity['ldd'] = subprocess.check_output(['ldd', str(binary)], env=env, text=True)
        validate_loaded_library(identity['ldd'], library, receipt['library_sha256'])
        write_json(args.output / 'identity.json', identity)
        modes = ['fp32', 'fp16', 'rabitq8']
        for trial in range(args.trials):
            for name, dataset in datasets:
                for mode in modes[trial % 3:] + modes[:trial % 3]:
                    tag = f'{name}-{mode}-{trial}'
                    selected = args.output / tag
                    snapshots = scratch / tag
                    snapshots.mkdir()
                    command = ['/usr/bin/time', '-v', '-o', str(args.output / (tag + '.resources.txt')),
                        'taskset', '-c', str(args.cpu), str(binary), str(dataset), str(selected),
                        mode, str(args.rounds), str(args.cycles), str(snapshots)]
                    if args.resident:
                        command.append('resident')
                    write_json(args.output / (tag + '.command.json'), dict(argv=command,
                        environment={key: env[key] for key in ['OMP_NUM_THREADS',
                            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LD_LIBRARY_PATH']}))
                    with (args.output / (tag + '.stdout')).open('w') as stdout, \
                         (args.output / (tag + '.stderr')).open('w') as stderr:
                        subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=True)
                    query_file = dataset / 'queries.fvecs'
                    with query_file.open('rb') as stream:
                        dim = int.from_bytes(stream.read(4), 'little', signed=True)
                    queries = query_file.stat().st_size // (4 + 4 * dim)
                    result = verify_run(selected, args.rounds, args.cycles, queries)
                    result.update(dataset=name, storage=mode, trial=trial,
                        snapshots={p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                                   for p in sorted(snapshots.iterdir())})
                    write_json(selected / 'audit.json', result)
                    runs.append(result)
                    write_json(args.output / 'rows.json', runs)
                    print(tag, 'PASS', 'recall', result['recall'], flush=True)
        assert sha(library) == identity['library_sha256'], 'Library changed during measurement'
        validate_build(args.feature_repo, args.feature_sha, args.build)
        assert sha(args.build / BUILD_RECEIPT) == identity['build_source_receipt_sha256']
    write_json(args.output / 'manifest.json', {str(p.relative_to(args.output)): sha(p)
        for p in sorted(args.output.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
    print('COMPLETE', len(runs), 'fresh processes; not Full or long endurance', flush=True)


if __name__ == '__main__':
    main()
