import glob
import os
import shutil
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Debug" if self.debug else "Release"
        mkl_static_link = os.environ.get("MKL_STATIC_LINK", "ON")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DENABLE_PYBINDS=ON",
            "-DENABLE_TESTS=OFF",
            f"-DMKL_STATIC_LINK={mkl_static_link}",
            "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -s",
        ]

        build_args = []
        if self.compiler.compiler_type != "msvc":
            jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 4))
            build_args += ["--", f"-j{jobs}"]

        env = os.environ.copy()
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        subprocess.check_call(
            ["cmake", src_dir] + cmake_args, cwd=build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_temp, env=env
        )

        lib_patterns = ["libvsag.so*", "libvsag.dylib", "*vsag*.dll"]
        for pattern in lib_patterns:
            for lib_path in glob.glob(os.path.join(build_temp, "**", pattern), recursive=True):
                shutil.copy(lib_path, extdir)

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

def _read_version():
    here = os.path.dirname(os.path.abspath(__file__))
    _version_file = os.path.join(here, 'pyvsag', '_version.py')
    
    # 1. Try pre-generated _version.py (from prepare_python_build.sh / CI)
    if os.path.exists(_version_file):
        ns = {}
        with open(_version_file) as f:
            exec(f.read(), ns)
        if ns.get('__version__'):
            return ns['__version__']

    # 2. Fallback: detect version from git tags via setuptools_scm (local dev install)
    try:
        from setuptools_scm import get_version
        return get_version(
            root=os.path.join(here, '..'),
            version_scheme='release-branch-semver',
            local_scheme='no-local-version',
        )
    except Exception as e:
        import sys
        print(f"Warning: setuptools_scm failed to detect version: {e}. Falling back to '0.0.0'.", file=sys.stderr)

    return '0.0.0'

def _read_long_description():
    here = os.path.dirname(os.path.abspath(__file__))
    readme = os.path.join(here, '..', 'README.md')
    if os.path.exists(readme):
        with open(readme, encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    version=_read_version(),
    long_description=_read_long_description(),
    long_description_content_type='text/markdown',
    distclass=BinaryDistribution,
    ext_modules=[CMakeExtension("pyvsag._pyvsag")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
