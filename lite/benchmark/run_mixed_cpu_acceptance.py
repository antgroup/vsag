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


def mutation_factor(mode):
    if mode not in ['all', 'update', 'replace', 'reinsert']:
        raise ValueError('Invalid mutation mode')
    return 3 if mode == 'all' else 1 if mode == 'update' else 2


def verify_run(output, rounds, cycles, queries, mutation_mode='all'):
    with (output / 'samples.csv').open() as stream:
        samples = list(csv.DictReader(stream))
    result = {}
    expected = dict(build=1, add=rounds * cycles, remove=rounds * cycles,
                    update=rounds * cycles, query=rounds * cycles * 3,
                    save=rounds, load=rounds)
    factor = mutation_factor(mutation_mode)
    expected['query'] = rounds * cycles * factor
    if mutation_mode == 'update':
        expected.update(add=0, remove=0)
    elif mutation_mode in ['replace', 'reinsert']:
        expected['update'] = 0
    assert len(samples) == sum(expected.values())
    for operation, count in expected.items():
        selected = [row for row in samples if row['operation'] == operation]
        assert len(selected) == count, operation
        if count == 0:
            continue
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
    if any('proc_hwm_kib' in row for row in states):
        assert all('proc_hwm_kib' in row for row in states)
        high_water = [int(row['proc_hwm_kib']) for row in states]
        assert high_water == sorted(high_water)
        assert all(value > 0 and value >= int(row['rss_kib']) for value, row in zip(high_water, states))
    result['states'] = states
    if mutation_mode != 'all':
        # Verify stage purity and the exact deterministic ID/query schedule, not only totals.
        position = 0
        query = 0
        def consume(number, identifier, operation):
            nonlocal position
            row = samples[position]
            assert (int(row['round']), int(row['id']), row['operation']) == (number, identifier, operation)
            position += 1
        consume(0, -1, 'build')
        stages = ['update'] if mutation_mode == 'update' else ['remove', 'add']
        for number in range(1, rounds + 1):
            for cycle in range(cycles):
                identifier = ((number - 1) * 65537 + cycle * 8191) % count
                for operation in stages:
                    consume(number, identifier, operation)
                    consume(number, query % queries, 'query')
                    query += 1
            consume(number, -1, 'save')
            consume(number, -1, 'load')
        assert position == len(samples)
    return result




def checkpoint_rounds(rounds, every):
    if every <= 0:
        return []
    return sorted({0, rounds, *range(every, rounds + 1, every)})


def verify_checkpoints(output, rounds, cycles, queries, every, mutation_mode='all'):
    result = {}
    for number in checkpoint_rounds(rounds, every):
        prefix = output / f'checkpoint-{number}'
        with Path(str(prefix) + '-state.csv').open() as stream:
            states = list(csv.DictReader(stream))
        assert len(states) == 1
        state = states[0]
        assert int(state['round']) == number
        assert int(state['mutation_calls']) == int(state['query_calls']) == number * cycles * mutation_factor(mutation_mode)
        count = int(state['count'])
        assert count > 10 and math.isfinite(float(state['rebuild_ms'])) and float(state['rebuild_ms']) >= 0
        with Path(str(prefix) + '-queries.csv').open() as stream:
            samples = list(csv.DictReader(stream))
        with Path(str(prefix) + '-neighbors.csv').open() as stream:
            neighbors = list(csv.DictReader(stream))
        assert len(samples) == queries and len(neighbors) == queries * 10
        for q, sample in enumerate(samples):
            assert int(sample['round']) == number and int(sample['query']) == q and int(sample['k']) == 10
            selected = neighbors[q * 10:(q + 1) * 10]
            assert [int(row['rank']) for row in selected] == list(range(10))
            assert all(int(row['round']) == number and int(row['query']) == q for row in selected)
            truth = {int(row['truth_id']) for row in selected}
            assert len(truth) == 10 and all(0 <= value < count for value in truth)
            for side in ['maintained', 'fresh']:
                values = [(float(row[side + '_distance']), int(row[side + '_id'])) for row in selected]
                assert len({value[1] for value in values}) == 10 and values == sorted(values)
                assert all(math.isfinite(distance) and 0 <= identifier < count for distance, identifier in values)
                assert int(sample[side + '_hits']) == len(truth & {v[1] for v in values})
                assert math.isfinite(float(sample[side + '_us'])) and float(sample[side + '_us']) >= 0
        result[str(number)] = dict(mutation_calls=number * cycles * mutation_factor(mutation_mode), count=count,
            maintained_recall=sum(int(row['maintained_hits']) for row in samples) / (queries * 10),
            fresh_recall=sum(int(row['fresh_hits']) for row in samples) / (queries * 10),
            maintained_p50_us=percentile([float(row['maintained_us']) for row in samples], .5),
            fresh_p50_us=percentile([float(row['fresh_us']) for row in samples], .5),
            rebuild_ms=float(state['rebuild_ms']))
    expected = {f'checkpoint-{n}-{suffix}.csv' for n in checkpoint_rounds(rounds, every)
                for suffix in ['neighbors', 'queries', 'state']}
    assert {p.name for p in output.glob('checkpoint-*.csv')} == expected
    with (output / 'quality.csv').open() as stream:
        endpoints = list(csv.DictReader(stream))
    for number in [0, rounds]:
        selected = [row for row in endpoints if int(row['round']) == number]
        assert len(selected) == queries
        assert result[str(number)]['maintained_recall'] == sum(int(row['hits']) for row in selected) / (queries * 10)
    return result


def verify_ceiling(output, rounds, queries):
    with (output / 'ceiling.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with (output / f'checkpoint-{rounds}-neighbors.csv').open() as stream:
        formal = list(csv.DictReader(stream))
    with (output / f'checkpoint-{rounds}-state.csv').open() as stream:
        count = int(next(csv.DictReader(stream))['count'])
    assert len(rows) == len(formal) == queries * 10
    hits = dict(maintained=0, fresh=0)
    for q in range(queries):
        group = rows[q * 10:(q + 1) * 10]
        original = formal[q * 10:(q + 1) * 10]
        assert [int(r['rank']) for r in group] == list(range(10))
        assert all(int(r['round']) == rounds and int(r['query']) == q and
                   int(r['ef_search']) == count for r in group)
        assert [r['truth_id'] for r in group] == [r['truth_id'] for r in original]
        truth = {int(r['truth_id']) for r in group}
        for side in hits:
            values = [(float(r[side + '_distance']), int(r[side + '_id'])) for r in group]
            assert values == sorted(values) and len({i for _, i in values}) == 10
            assert all(math.isfinite(d) and 0 <= i < count for d, i in values)
            hits[side] += len(truth & {i for _, i in values})
    return dict(ef_search=count, **{side + '_recall': hits[side] / (queries * 10) for side in hits})


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


def release_verified_snapshots(scratch, snapshots, expected):
    # These are exclusively tool-created RAM snapshots, normally removed when
    # TemporaryDirectory exits. Verify the entire set before releasing a run.
    root = scratch.resolve(strict=True)
    if snapshots.is_symlink() or snapshots.resolve(strict=True).parent != root:
        raise ValueError('Snapshot directory must be a direct child of scratch')
    entries = list(snapshots.iterdir())
    if {path.name for path in entries} != set(expected):
        raise ValueError('Unexpected temporary snapshot set')
    for path in entries:
        if path.is_symlink() or not path.is_file() or path.resolve().parent != snapshots.resolve():
            raise ValueError('Unsafe temporary snapshot')
        record = expected[path.name]
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Temporary snapshot changed after verification')
    for path in entries:
        path.unlink()
    snapshots.rmdir()


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
    parser.add_argument('--checkpoint-every', type=int, default=0,
                        help='Opt-in resident quality/rebuild diagnosis every N rounds; not production tuning')
    parser.add_argument('--mutation-mode', choices=['all', 'update', 'replace', 'reinsert'],
                        default='all', help='Opt-in isolated mutation workload; requires resident checkpoints')
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
    if args.checkpoint_every < 0 or args.checkpoint_every > 1000000 or (args.checkpoint_every and not args.resident):
        parser.error('--checkpoint-every requires --resident and an interval in [1,1000000]')
    if args.mutation_mode != 'all' and not (args.resident and args.checkpoint_every):
        parser.error('Isolated mutation modes require --resident and --checkpoint-every')
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
    if args.checkpoint_every:
        identity['configuration']['checkpoint_every'] = args.checkpoint_every
        identity['limits'] += [
            'Checkpoint FP64 and fresh rebuild run in a forked single-threaded child; no parent allocator mutations',
            'Fork/wait and cache warming perturb timing; checkpoint latencies are diagnostic, not a formal speed comparison',
            'Fresh rebuild changes graph/insertion history; RaBitQ additionally retrains its model',
            'Parent RSS/peak excludes child allocations; process time/resources may include children',
            'Save/Load equality is explicit snapshot consistency, not per-operation durability']
    if args.mutation_mode != 'all':
        identity['limits'][0] = 'Fixed-round isolated mutation diagnostic, not hours-long endurance'
        identity['configuration']['mutation_mode'] = args.mutation_mode
        identity['configuration']['calls_per_cycle'] = mutation_factor(args.mutation_mode)
        identity['configuration']['final_diagnostic_ef'] = 'N'
        identity['limits'] += [
            'Isolated workloads have unequal API counts per cycle; compare quality at the same data-change cycles',
            'Reinsert keeps vector values unchanged; update/replace use the same changed-value recurrence',
            'Final ef=N is a separate routing/encoding diagnosis, not the formal query budget or speed result']
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
                    if args.checkpoint_every:
                        command.append(str(args.checkpoint_every))
                    if args.mutation_mode != 'all':
                        command.append(args.mutation_mode)
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
                    result = verify_run(selected, args.rounds, args.cycles, queries, args.mutation_mode)
                    if args.checkpoint_every:
                        result['checkpoints'] = verify_checkpoints(selected, args.rounds, args.cycles,
                                                                  queries, args.checkpoint_every, args.mutation_mode)
                    if args.mutation_mode != 'all':
                        result['ceiling'] = verify_ceiling(selected, args.rounds, queries)
                    result.update(dataset=name, storage=mode, trial=trial,
                        snapshots={p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                                   for p in sorted(snapshots.iterdir())})
                    write_json(selected / 'audit.json', result)
                    runs.append(result)
                    write_json(args.output / 'rows.json', runs)
                    release_verified_snapshots(scratch, snapshots, result['snapshots'])
                    print(tag, 'PASS', 'recall', result['recall'], flush=True)
        assert sha(library) == identity['library_sha256'], 'Library changed during measurement'
        validate_build(args.feature_repo, args.feature_sha, args.build)
        assert sha(args.build / BUILD_RECEIPT) == identity['build_source_receipt_sha256']
    write_json(args.output / 'manifest.json', {str(p.relative_to(args.output)): sha(p)
        for p in sorted(args.output.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
    print('COMPLETE', len(runs), 'fresh processes; not Full or long endurance', flush=True)


if __name__ == '__main__':
    main()
