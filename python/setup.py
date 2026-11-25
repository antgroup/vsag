import collections
import glob
import os
import platform
import shutil
import subprocess
import sys
# Set this before setuptools reads the platform configuration. All subprocesses,
# including OpenBLAS and the private OpenMP build, inherit the same target.
if sys.platform == "darwin":
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "15.0")

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.dist import Distribution

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Builds libvsag_c, which pyvsag loads through cffi; there is no Python extension module."""

    def get_ext_filename(self, ext_name):
        suffix = "dylib" if sys.platform == "darwin" else "so"
        return os.path.join("pyvsag", "libvsag_c." + suffix)

    @staticmethod
    def _report_ccache_stats():
        """Prints ccache statistics so that a build log shows whether the cache is effective.

        The per-result breakdown is opt-in: it only runs when CCACHE_LOGFILE is set, and it
        is a temporary diagnostic for finding why ccache misses.
        """
        try:
            subprocess.call(["ccache", "-s"])
            log = os.environ.get("CCACHE_LOGFILE")
            if log and os.path.exists(log):
                results = collections.Counter()
                with open(log, errors="replace") as f:
                    for line in f:
                        if "Result:" in line:
                            results[line.split("Result:", 1)[1].strip()] += 1
                if not results:
                    print("ccache result: no 'Result:' lines found in CCACHE_LOGFILE")
                for name, count in results.most_common():
                    print(f"ccache result: {name}: {count}")
        except OSError as e:
            print(f"ccache statistics unavailable: {e}")

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.join(self.build_lib, "pyvsag"))
        os.makedirs(extdir, exist_ok=True)
        cfg = "Debug" if self.debug else "Release"
        mkl_static_link = os.environ.get("MKL_STATIC_LINK", "ON")
        # The released wheel has no libaio dependency; a local build can opt back in.
        libaio = os.environ.get("VSAG_ENABLE_LIBAIO", "OFF").strip().upper() or "OFF"
        use_system_deps = os.environ.get("VSAG_USE_SYSTEM_DEPS", "OFF").strip().upper() or "OFF"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DVSAG_USE_SYSTEM_DEPS:STRING={use_system_deps}",
            "-DENABLE_PYBINDS=ON",
            "-DENABLE_TESTS=OFF",
            f"-DENABLE_LIBAIO={libaio}",
            f"-DMKL_STATIC_LINK={mkl_static_link}",
            "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG"
            + ("" if sys.platform == "darwin" else " -s"),
        ]
        if sys.platform == "darwin":
            cmake_args.extend([
                "-DVSAG_OPENMP_STATIC=ON",
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=" + os.environ["MACOSX_DEPLOYMENT_TARGET"],
            ])
            if os.environ.get("LIBOMP_PREFIX"):
                cmake_args.append("-DLIBOMP_PREFIX=" + os.environ["LIBOMP_PREFIX"])
        use_ccache = os.environ.get("VSAG_ENABLE_CCACHE", "OFF").strip().upper() == "ON"
        if use_ccache:
            cmake_args.append("-DENABLE_CCACHE=ON")
            try:
                subprocess.call(["ccache", "-z"])
            except OSError as e:
                print(f"ccache unavailable, skipping stats reset: {e}")

        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 4))
        # Third-party builds (OpenBLAS, ...) use the same parallelism as the main build.
        cmake_args.append(f"-DNUM_BUILDING_JOBS={jobs}")
        env = os.environ.copy()
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        subprocess.check_call(["cmake", src_dir] + cmake_args, cwd=build_temp, env=env)
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "vsag_c", "--", f"-j{jobs}"],
            cwd=build_temp,
            env=env,
        )

        if use_ccache:
            self._report_ccache_stats()

        libs = glob.glob(os.path.join(build_temp, "**", "libvsag_c.*"), recursive=True)
        if len(libs) != 1:
            raise RuntimeError("expected exactly one libvsag_c in the build tree, found: %s" % libs)
        shutil.copy(libs[0], extdir)
        shutil.copy(
            os.path.join(src_dir, "include", "vsag", "vsag_c_api.h"),
            os.path.join(extdir, "_vsag_c_api.h"),
        )
        shutil.copy(
            os.path.join(src_dir, "include", "vsag", "vsag_eval_c_api.h"),
            os.path.join(extdir, "_vsag_eval_c_api.h"),
        )


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    """The library is loaded through cffi, so the wheel does not depend on the Python ABI."""

    def get_tag(self):
        _, _, plat = bdist_wheel.get_tag(self)
        if sys.platform == "darwin":
            # CMake builds one native architecture, even when Python is universal2.
            # Preserve the deployment target calculated from the built library.
            os_name, major, minor, _ = plat.split("_", 3)
            plat = f"{os_name}_{major}_{minor}_{platform.machine()}"
        return "py3", "none", plat


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
    license_expression=("Apache-2.0 AND Apache-2.0 WITH LLVM-exception"
                        if sys.platform == "darwin" else "Apache-2.0"),
    license_files=(["LICENSE", "NOTICE", "licenses/LLVM-OpenMP.txt"]
                   if sys.platform == "darwin" else ["LICENSE"]),
    long_description=_read_long_description(),
    long_description_content_type='text/markdown',
    distclass=BinaryDistribution,
    ext_modules=[CMakeExtension("pyvsag._pyvsag")],
    cmdclass={"build_ext": CMakeBuild, "bdist_wheel": PlatformWheel},
    zip_safe=False,
)
