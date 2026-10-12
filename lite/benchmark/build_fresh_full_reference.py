#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Fresh root-CMake native Full build; cached dependency sources, never cached objects."""
import argparse
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from run_mixed_cpu_acceptance import sha, write_json


@contextmanager
def cached_archive_url(archive):
    # CMake 3.22 rejects file URLs mixed with the official HTTPS fallbacks.
    # Serve the unchanged, hash-checked source archive on loopback only.
    handler = partial(SimpleHTTPRequestHandler, directory=str(archive.resolve().parent))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield 'http://127.0.0.1:' + str(server.server_port) + '/' + archive.name
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--deps', type=Path, required=True)
    parser.add_argument('--antlr-archive', type=Path, required=True)
    parser.add_argument('--install', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = 'dbb2fe0bec685ae7a343b1d2060d0553a98e20d2'
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()
    assert head == expected and not subprocess.check_output(['git','status','--porcelain','-uno'],cwd=args.source)
    assert not args.install.exists() and not args.output.exists(), 'Never overwrite a reference'
    args.output.mkdir(parents=True)
    assert hashlib.md5(args.antlr_archive.read_bytes()).hexdigest() == '3b75610fc8a827119258cba09a068be5'
    env = dict(os.environ, TMPDIR='/dev/shm')
    records = []
    def run(tag,command):
        begin=time.monotonic()
        with (args.output/(tag+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=args.source,env=env,stdout=log,stderr=log)
        records.append(dict(tag=tag,argv=command,exit=result.returncode,wall_seconds=time.monotonic()-begin))
        write_json(args.output/'commands.json',records)
        assert result.returncode == 0, tag+' failed; log retained'
    with cached_archive_url(args.antlr_archive) as archive_url, tempfile.TemporaryDirectory(prefix='vsag-fresh-full-',dir='/dev/shm') as directory:
        env['VSAG_THIRDPARTY_ANTLR4_4_13_2'] = archive_url
        build=Path(directory)/'build'
        assert not build.exists()
        command=['cmake','-G','Ninja','-S',str(args.source),'-B',str(build),
            '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DENABLE_CCACHE=OFF','-DENABLE_TESTS=OFF','-DENABLE_EXAMPLES=OFF',
            '-DENABLE_TOOLS=OFF','-DENABLE_PYBINDS=OFF','-DENABLE_NODE_BINDS=OFF',
            '-DENABLE_INTEL_MKL=OFF','-DVSAG_USE_SYSTEM_DEPS=AUTO',
            '-DVSAG_USE_SYSTEM_OPENBLAS=ON','-DVSAG_USE_SYSTEM_FMT=ON',
            '-DENABLE_ASAN=OFF','-DENABLE_TSAN=OFF','-DENABLE_COVERAGE=OFF',
            '-DENABLE_THIN_LTO=OFF','-DNUM_BUILDING_JOBS=2',
            '-DCMAKE_INSTALL_PREFIX='+str(args.install)]
        for name,folder in [('CPUINFO','cpuinfo-src'),('NLOHMANN_JSON','nlohmann_json-src'),
                            ('ROARINGBITMAP','roaringbitmap-src'),('THREAD_POOL','thread_pool-src'),
                            ('TSL','tsl-src')]:
            sentinel = 'ThreadPool.h' if name == 'THREAD_POOL' else 'CMakeLists.txt'
            assert (args.deps/folder/sentinel).exists(), folder
            command.append('-DFETCHCONTENT_SOURCE_DIR_'+name+'='+str(args.deps/folder))
        run('configure',command)
        assert not list(build.rglob('*.o')), 'Objects already existed before build'
        print('CONFIGURED empty build directory; compiling native Full core',flush=True)
        run('build',['cmake','--build',str(build),'--target','vsag','--parallel','2'])
        candidates={p.resolve() for p in (build/'src').glob('libvsag.so*') if p.is_file()}
        assert len(candidates)==1
        library=candidates.pop()
        core_objects={str(p.relative_to(build)):dict(bytes=p.stat().st_size,sha256=sha(p))
                      for p in sorted((build/'src').rglob('*.o'))}
        assert len(core_objects)>50, 'Insufficient selected core objects'
        write_json(args.output/'core-objects.json',core_objects)
        shutil.copyfile(build/'CMakeCache.txt',args.output/'CMakeCache.txt')
        shutil.copyfile(build/'compile_commands.json',args.output/'compile_commands.json')
        shutil.copyfile(build/'.ninja_log',args.output/'ninja-build.log')
        if (args.source/'src/version.h').exists():
            shutil.copyfile(args.source/'src/version.h',args.output/'generated-version.h')
        (args.install/'lib').mkdir(parents=True)
        retained=args.install/'lib'/library.name
        original=dict(bytes=library.stat().st_size,sha256=sha(library))
        shutil.copyfile(library,retained)
        subprocess.run(['strip','--strip-unneeded',str(retained)],check=True)
        assert sha(library)==original['sha256'], 'Original build library modified'
        for alias in ['libvsag.so','libvsag.so.0.0']:
            if alias != retained.name:
                (args.install/'lib'/alias).symlink_to(retained.name)
        shutil.copytree(args.source/'include',args.install/'include')
        headers={str(p.relative_to(args.install/'include')):sha(p)
                 for p in sorted((args.install/'include').rglob('*')) if p.is_file()}
        receipt=dict(full_source_head=head,full_build_boundary='New empty build directory; selected native Full core and bundled targets compiled from source; cached source/system libraries reused',
            fresh_core=True,ccache=False,prior_object_count=0,core_object_count=len(core_objects),
            full_headers_sha256=headers,library_sha256=sha(retained),library_bytes=retained.stat().st_size,
            original_library=original,install=str(args.install.resolve()),
            compiler=subprocess.check_output(['c++','--version'],text=True),
            system=subprocess.check_output(['uname','-a'],text=True),
            dependency_sources=str(args.deps),antlr_archive_sha256=sha(args.antlr_archive),
            full_suite='not run',strip_on_copy_only=True)
        (args.output/'ldd.txt').write_text(subprocess.check_output(['ldd',str(retained)],text=True))
        write_json(args.output/'receipt.json',receipt)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()==head
        assert not subprocess.check_output(['git','status','--porcelain','-uno'],cwd=args.source)
        print('FRESH FULL RETAINED',receipt['library_sha256'],receipt['library_bytes'],len(core_objects),flush=True)


if __name__=='__main__':
    main()
