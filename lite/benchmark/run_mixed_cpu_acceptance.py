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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-repo', type=Path, required=True)
    parser.add_argument('--feature-sha', required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--dataset', action='append', required=True, help='name=path')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--cycles', type=int, default=100)
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--resident', action='store_true',
                        help='Keep the original index between rounds; reload only checks persistence')
    args = parser.parse_args()
    assert min(args.trials, args.rounds, args.cycles) > 0
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.feature_repo,
                                   text=True).strip() == args.feature_sha
    assert not subprocess.check_output(['git', 'status', '--porcelain', '-uno'], cwd=args.feature_repo)
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('mixed_cpu_acceptance.cpp')
    env = {k: v for k, v in os.environ.items() if not k.startswith('VSAG_')}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               LD_LIBRARY_PATH=str(args.build.resolve()))
    library = args.build / 'libvsag-lite.so'
    identity = dict(feature_sha=args.feature_sha, library_sha256=sha(library),
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
    (args.output / 'source.cpp').write_bytes(source.read_bytes())
    (args.output / 'CMakeCache.txt').write_bytes((args.build / 'CMakeCache.txt').read_bytes())
    (args.output / 'compile_commands.json').write_bytes((args.build / 'compile_commands.json').read_bytes())
    identity['inputs'] = {}
    datasets = []
    for value in args.dataset:
        name, directory = value.split('=', 1)
        assert name and name.isalnum(), 'Dataset name must be alphanumeric'
        path = Path(directory).resolve()
        identity['inputs'][name] = {filename: dict(path=str(path / filename), sha256=sha(path / filename))
                                    for filename in ['base.fvecs', 'queries.fvecs']}
        datasets.append((name, path))
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
    write_json(args.output / 'manifest.json', {str(p.relative_to(args.output)): sha(p)
        for p in sorted(args.output.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
    print('COMPLETE', len(runs), 'fresh processes; not Full or long endurance', flush=True)


if __name__ == '__main__':
    main()
