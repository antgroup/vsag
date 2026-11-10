#!/usr/bin/env python3
"""Stream Linux build resource evidence without changing the command's result."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


CGROUP_FILES = (
    'memory.current', 'memory.peak', 'memory.max', 'memory.events',
    'memory.events.local', 'memory.pressure', 'cpu.stat',
    'memory.usage_in_bytes', 'memory.max_usage_in_bytes',
    'memory.limit_in_bytes', 'memory.failcnt', 'memory.oom_control',
)
COMPILERS = {'cc1', 'cc1plus', 'gcc', 'g++', 'clang', 'clang++', 'lto1', 'ld', 'ld.lld'}


def read(path):
    try:
        return path.read_text(errors='replace').strip()
    except OSError:
        return None


def cgroup_directories(proc=Path('/proc'), root=Path('/sys/fs/cgroup')):
    """Resolve both host and cgroup-namespaced mount layouts, including v1."""
    groups = read(proc / 'self/cgroup') or ''
    mounts = read(proc / 'self/mountinfo') or ''
    found = set()
    for line in groups.splitlines():
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        _, controllers, group = parts
        for mount in mounts.splitlines():
            fields = mount.split()
            if '-' not in fields or len(fields) < 7:
                continue
            split = fields.index('-')
            if len(fields) <= split + 3:
                continue
            kind, options = fields[split + 1], fields[split + 3].split(',')
            if not ((kind == 'cgroup2' and not controllers) or
                    (kind == 'cgroup' and 'memory' in controllers.split(',') and 'memory' in options)):
                continue
            base, mountpoint = Path(fields[3]), Path(fields[4])
            try:
                candidate = mountpoint / Path(group).relative_to(base)
            except ValueError:
                # Membership is relative to the private cgroup namespace, while
                # mountinfo can retain the host-side root. Resolve it under the
                # matching controller mount, not the unified cgroup root.
                try:
                    candidate = mountpoint / Path(group).relative_to('/')
                except ValueError:
                    continue
                if '..' in candidate.parts or not candidate.is_relative_to(root):
                    continue
                if not any((candidate / name).is_file() for name in CGROUP_FILES):
                    continue
            if '..' not in candidate.parts and candidate.is_relative_to(root):
                found.add(candidate)
    # With a private cgroup namespace the container root can be mounted directly.
    if any((root / name).exists() for name in CGROUP_FILES):
        found.add(root)
    return sorted(found)


def processes(pgid, proc=Path('/proc')):
    result = []
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        stat = read(path / 'stat')
        if not stat or ')' not in stat:
            continue
        fields = stat.rsplit(')', 1)[1].split()
        if len(fields) < 22 or fields[2] != str(pgid):
            continue
        status = read(path / 'status') or ''
        rss = next((int(line.split()[1]) for line in status.splitlines()
                    if line.startswith('VmRSS:')), 0)
        comm = read(path / 'comm')
        item = {'pid': int(path.name), 'name': comm, 'state': fields[0], 'rssKiB': rss,
                'cpuTicks': int(fields[11]) + int(fields[12])}
        # Do not publish arbitrary command lines or environments. Only compiler
        # source operands help identify work still in flight (Ninja prints completions).
        if comm and (comm in COMPILERS or comm.startswith(('clang-', 'clang++-', 'g++-', 'gcc-'))):
            args = (read(path / 'cmdline') or '').split('\0')
            item['sources'] = [arg for arg in args if not arg.startswith('-') and
                               arg.endswith(('.cpp', '.cc', '.c', '.cxx', '.pch'))][:8]
            item['waitChannel'] = read(path / 'wchan')
        result.append(item)
    return sorted(result, key=lambda item: item['rssKiB'], reverse=True)


def snapshot(pgid):
    memory = read(Path('/proc/meminfo')) or ''
    memory = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
              for line in memory.splitlines() if line.startswith(
                  ('MemTotal:', 'MemAvailable:', 'SwapTotal:', 'SwapFree:'))}
    children = processes(pgid)
    return {'time': datetime.now(timezone.utc).isoformat(), 'memory': memory,
            'memoryPressure': read(Path('/proc/pressure/memory')),
            'loadAverage': read(Path('/proc/loadavg')),
            'clockTicksPerSecond': os.sysconf('SC_CLK_TCK'),
            'cgroups': {str(path): {name: read(path / name) for name in CGROUP_FILES
                                   if (path / name).exists()}
                        for path in cgroup_directories()},
            'processCount': len(children), 'processRssKiB': sum(p['rssKiB'] for p in children),
            'largestProcesses': children[:12],
            'compilers': [p for p in children if 'sources' in p]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--interval', type=float, default=15)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command or not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('a command and positive sampling interval are required')
    output = Path(args.output_dir)
    started = time.monotonic()
    received = []
    proc = None
    signal_at = None

    def forward(signum, _frame):
        nonlocal signal_at
        if not received:
            received.append(signum)
            signal_at = time.monotonic()
        if proc is not None:
            try:
                os.killpg(proc.pid, signum)
            except ProcessLookupError:
                pass

    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    capture_error = None
    try:
        output.mkdir(parents=True, exist_ok=True)
        log_context = (output / 'samples.jsonl').open('wb', buffering=0)
    except OSError as error:
        capture_error = type(error).__name__
        log_context = nullcontext(None)
        print('[build-resource] file capture unavailable: ' + capture_error, flush=True)
    try:
        proc = subprocess.Popen(command, start_new_session=True)
        if received:
            forward(received[0], None)
        with log_context as log:
            next_sample = 0.0
            while True:
                now = time.monotonic()
                if now >= next_sample or proc.poll() is not None:
                    try:
                        sample = snapshot(proc.pid)
                        sample['elapsedSeconds'] = round(now - started, 3)
                        record = json.dumps(sample, sort_keys=True)
                        print('[build-resource] ' + record, flush=True)
                        if log is not None:
                            log.write((record + '\n').encode())
                    except (OSError, ValueError) as error:
                        capture_error = type(error).__name__
                        print('[build-resource] sample unavailable: ' + capture_error, flush=True)
                    next_sample = now + args.interval
                if proc.poll() is not None:
                    break
                if signal_at is not None and now - signal_at >= 10:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                time.sleep(min(0.2, args.interval))
        result = proc.wait()
        code = 128 + received[0] if received else (128 - result if result < 0 else result)
        summary = {'exitCode': code, 'commandReturnCode': result,
                   'receivedSignal': received[0] if received else None,
                   'elapsedSeconds': round(time.monotonic() - started, 3),
                   'lastCaptureError': capture_error,
                   'note': 'Exit 137 alone does not prove OOM; inspect memory.events and runner logs.'}
        try:
            (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        except OSError as error:
            summary['lastCaptureError'] = type(error).__name__
        print('[build-resource-result] ' + json.dumps(summary), flush=True)
        return code
    finally:
        # Never leave build descendants running after a cancelled monitor.
        if proc is not None and (received or proc.poll() != 0):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    sys.exit(main())
