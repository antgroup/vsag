#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Build, relocate and smoke-test an explicitly scoped native C++ runtime bundle."""
import argparse
import csv
import gzip
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time

from pack_mixed_cpu_acceptance import unpack
from run_mixed_cpu_acceptance import sha, write_json


def resolved_dependencies(text):
    if 'not found' in text:
        raise ValueError('Unresolved dynamic dependency')
    aliases = {}
    for line in text.splitlines():
        matched = re.search(r'^\s*(\S+) => (/\S+) \(', line)
        if matched:
            aliases[matched.group(1)] = Path(matched.group(2)).resolve(strict=True)
        else:
            matched = re.search(r'^\s*(/\S+) \(', line)
            if matched:
                aliases[Path(matched.group(1)).name] = Path(matched.group(1)).resolve(strict=True)
    return aliases


def collect(application, library_directory, env):
    pending = [application.resolve()]
    files = {}
    aliases = {}
    while pending:
        source = pending.pop()
        if str(source) in files:
            continue
        text = subprocess.check_output(['ldd', str(source)], env=dict(env,
            LD_LIBRARY_PATH=str(library_directory)), text=True)
        dependencies = resolved_dependencies(text)
        files[str(source)] = dict(source_sha256=sha(source), source_bytes=source.stat().st_size,
                                  ldd=text)
        for alias, target in dependencies.items():
            if alias in aliases and aliases[alias] != target:
                raise ValueError('Conflicting dependency alias: ' + alias)
            aliases[alias] = target
        pending.extend(dependencies.values())
    return files, aliases


def package_owner(source):
    candidates = [str(source), str(source).replace('/usr/lib/', '/lib/', 1)]
    for candidate in candidates:
        found = subprocess.run(['dpkg-query', '-S', candidate], text=True, capture_output=True)
        if found.returncode == 0:
            return found.stdout.split(': ', 1)[0]
    raise ValueError('System library has no package-owner receipt: ' + str(source))


def archive_bundle(root, archive):
    with archive.open('wb') as target, gzip.GzipFile(filename='', fileobj=target, mode='wb',
                                                   compresslevel=9, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as stream:
            for path in sorted(root.rglob('*')):
                if path.is_dir():
                    continue
                info = stream.gettarinfo(str(path), arcname=str(path.relative_to(root)))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                if path.is_symlink():
                    stream.addfile(info)
                else:
                    with path.open('rb') as source:
                        stream.addfile(info, source)


def safe_bundle_unpack(archive, destination):
    root = destination.resolve()
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream.getmembers():
            target = root / member.name
            if member.name.startswith('/') or root not in target.resolve().parents:
                raise ValueError('Unsafe bundle member')
            if member.issym():
                if Path(member.linkname).is_absolute() or root not in (target.parent / member.linkname).resolve().parents:
                    raise ValueError('Unsafe bundle alias')
            elif not member.isfile():
                raise ValueError('Unexpected bundle entry type')
            stream.extract(member, root)


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_dir():
            continue
        name = str(path.relative_to(root))
        result[name] = dict(kind='symlink', target=os.readlink(path)) if path.is_symlink() else dict(
            kind='file', bytes=path.stat().st_size, sha256=sha(path))
    return result


def build_bundle(application, library_directory, root, feature, env, full):
    files, aliases = collect(application, library_directory, env)
    for name in ['bin', 'lib', 'licenses']:
        (root / name).mkdir(parents=True)
    destinations = {}
    licenses = {}
    for source_text, record in files.items():
        source = Path(source_text)
        target = root / ('bin/probe' if source == application.resolve() else 'lib/' + source.name)
        assert not target.exists(), 'Duplicate payload basename'
        shutil.copyfile(source, target)
        subprocess.run(['strip', '--strip-unneeded', str(target)], check=True)
        target.chmod(0o755)
        assert sha(source) == record['source_sha256'], 'Source ELF changed'
        destinations[source] = target
        record.update(packaged_path=str(target.relative_to(root)), stripped_bytes=target.stat().st_size,
                      stripped_sha256=sha(target))
        if source != application.resolve() and 'vsag' not in source.name:
            owner = package_owner(source)
            copyright_file = Path('/usr/share/doc') / owner.split(':')[0] / 'copyright'
            assert copyright_file.is_file(), owner
            license_name = owner.replace(':', '-') + '-copyright'
            if license_name not in licenses:
                shutil.copyfile(copyright_file, root / 'licenses' / license_name)
                licenses[license_name] = dict(package=owner, source=str(copyright_file),
                                             sha256=sha(copyright_file))
    for alias, source in aliases.items():
        target = root / 'lib' / alias
        actual = destinations[source]
        if target != actual:
            assert not target.exists() and not target.is_symlink(), alias
            target.symlink_to(actual.name)
    shutil.copyfile(feature / 'LICENSE', root / 'licenses/VSAG-Apache-2.0')
    for path in sorted(Path('/usr/share/common-licenses').iterdir()):
        if path.is_file():
            shutil.copyfile(path, root / 'licenses' / ('common-' + path.name))
    loader = next(p for p in (root / 'lib').iterdir() if p.name.startswith('ld-linux') and not p.is_symlink())
    inhibit = ':' + ':'.join(sorted(aliases))
    script = '#!/bin/sh\nset -eu\nROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
    script += 'export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
    script += 'STORAGE=${3:-fp32}\n'
    if full:
        script += 'export VSAG_FULL_PROFILE=small\n[ "$STORAGE" != rabitq8 ] || STORAGE=rabitq3x5\n'
        arguments = '"$1" "$2" "$STORAGE" 512 0'
    else:
        script += 'export VSAG_GRAPH_STORAGE="$STORAGE" VSAG_GRAPH_DEGREE=16 VSAG_GRAPH_EF=128 VSAG_GRAPH_QUERY_EF=512\n'
        arguments = '"$1" "$2" 0 1'
    script += f'exec "$ROOT/lib/{loader.name}" --inhibit-cache --inhibit-rpath "{inhibit}" --library-path "$ROOT/lib" "$ROOT/bin/probe" {arguments}\n'
    (root / 'run').write_text(script)
    (root / 'run').chmod(0o755)
    return files, licenses, loader.name


def smoke(root, dataset, storage, snapshot, env, output):
    command = [str(root / 'run'), str(dataset), str(snapshot), storage]
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    mapped = None
    try:
        for _ in range(200):
            if process.poll() is not None:
                break
            try:
                lines = Path(f'/proc/{process.pid}/maps').read_text().splitlines()
            except FileNotFoundError:
                continue
            paths = {line.split()[-1] for line in lines if '/' in line and not line.endswith('(deleted)')}
            if str(root / 'bin/probe') in paths:
                mapped = sorted(p for p in paths if '.so' in p or p == str(root / 'bin/probe'))
                break
            time.sleep(.01)
        stdout, stderr = process.communicate(timeout=120)
    except BaseException:
        process.kill()
        process.communicate()
        raise
    output.with_suffix('.stdout').write_text(stdout)
    output.with_suffix('.stderr').write_text(stderr)
    assert process.returncode == 0, stderr
    assert mapped and all(root in Path(p).parents for p in mapped), mapped
    lines = stdout.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith(('base_count,', 'mode,base_count,')))
    metrics = next(csv.DictReader(io.StringIO('\n'.join(lines[header:header+2]))))
    assert int(metrics['base_count']) == 10000 and float(metrics['recall_at_k']) >= .95
    return dict(argv=command, exit=process.returncode, mapped_elf_paths=mapped, metrics=metrics,
                snapshot_bytes=snapshot.stat().st_size, snapshot_sha256=sha(snapshot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=Path, required=True)
    parser.add_argument('--lite-install', type=Path, required=True)
    parser.add_argument('--full-install', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    assert sha(args.source_archive) == '0c6697115a21dc210ab63cc0dfd659c53614d39ffc89115292765a1565ca9b54', 'Untrusted consumer-source archive'
    assert sha(args.lite_install / 'lib/libvsag-lite.so') == '3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
    assert sha(args.full_install / 'lib/libvsag.so') == '9aeaa550703f116bef2959769d4f2f03da254b07c906ba941d8186224cc70c72'
    env = {k:v for k,v in os.environ.items() if not k.startswith('VSAG_') and k not in
           ['LD_PRELOAD','LD_LIBRARY_PATH','LD_AUDIT','LD_DEBUG']}
    env.update(LC_ALL='C', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    report = dict(profile='Native C++ index application + recursive dynamic ELF runtime + loader + launcher + license texts',
        excludes=['SDK headers/static archives', 'Python bindings', 'container/OS/kernel/shell', 'datasets/snapshots', 'dlopen plugins'],
        source_archive_sha256=sha(args.source_archive), redistribution='measurement prototype only; license notices copied, no legal certification',
        strip=subprocess.check_output(['strip','--version'],text=True).splitlines()[0], packages={})
    with tempfile.TemporaryDirectory(prefix='vsag-runtime-package-',dir='/dev/shm') as directory:
        scratch = Path(directory)
        sources = scratch / 'sources'
        unpack(args.source_archive, sources)
        for mode, install, filename in [('lite',args.lite_install,'source-graph_crud_quality.cpp'),
                                        ('full',args.full_install,'source-full_rabitq_dataset_main.cpp')]:
            binary = scratch / (mode+'-probe')
            source = sources / filename
            command = ['c++','-std=c++17','-O3','-DNDEBUG','-I'+str(install/'include'),str(source),
                '-L'+str(install/'lib'),'-Wl,-rpath,'+str(install/'lib'),'-lvsag-lite' if mode=='lite' else '-lvsag','-o',str(binary)]
            with (args.output/(mode+'-compile.log')).open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=log,check=True)
            root = scratch / (mode+'-built')
            files, licenses, loader = build_bundle(binary,install/'lib',root,args.feature,env,mode=='full')
            contents = inventory(root)
            archive = scratch / (mode+'.tar.gz')
            archive_bundle(root,archive)
            relocated = scratch / ('relocated-'+mode)
            safe_bundle_unpack(archive,relocated)
            assert inventory(relocated) == contents
            listing = subprocess.check_output([str(relocated/'lib'/loader),'--inhibit-cache','--library-path',
                str(relocated/'lib'),'--list',str(relocated/'bin/probe')],env=env,text=True)
            (args.output/(mode+'-loader-list.txt')).write_text(listing)
            checks = [smoke(relocated,args.dataset,storage,scratch/(mode+'-'+storage+'.snapshot'),env,
                            args.output/(mode+'-'+storage)) for storage in ['fp32','fp16','rabitq8']]
            report['packages'][mode] = dict(compile=command,consumer_source_sha256=sha(source),files=files,
                licenses=licenses,inventory=contents,payload_bytes=sum(v.get('bytes',0) for v in contents.values()),
                tar_gzip_bytes=archive.stat().st_size,tar_gzip_sha256=sha(archive),relocation_checks=checks,
                source_libraries_unchanged=True)
        report['payload_reduction_percent'] = (1-report['packages']['lite']['payload_bytes']/report['packages']['full']['payload_bytes'])*100
        report['compressed_reduction_percent'] = (1-report['packages']['lite']['tar_gzip_bytes']/report['packages']['full']['tar_gzip_bytes'])*100
        write_json(args.output/'report.json',report)
    write_json(args.output/'manifest.json',{p.name:sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps({k:report[k] for k in ['payload_reduction_percent','compressed_reduction_percent']}))


if __name__ == '__main__':
    main()
