import glob
import os
import shutil
import subprocess
import sys
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
        return os.path.join("pyvsag", "libvsag_c.so")

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
            "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -s",
        ]
        if os.environ.get("VSAG_ENABLE_CCACHE", "OFF").strip().upper() == "ON":
            cmake_args.append("-DENABLE_CCACHE=ON")

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

        libs = glob.glob(os.path.join(build_temp, "**", "libvsag_c.*"), recursive=True)
        if len(libs) != 1:
            raise RuntimeError("expected exactly one libvsag_c in the build tree, found: %s" % libs)
        shutil.copy(libs[0], extdir)
        shutil.copy(
            os.path.join(src_dir, "include", "vsag", "vsag_c_api.h"),
            os.path.join(extdir, "_vsag_c_api.h"),
        )


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    """The library is loaded through cffi, so the wheel does not depend on the Python ABI."""

    def get_tag(self):
        _, _, plat = bdist_wheel.get_tag(self)
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
    long_description=_read_long_description(),
    long_description_content_type='text/markdown',
    distclass=BinaryDistribution,
    ext_modules=[CMakeExtension("pyvsag._pyvsag")],
    cmdclass={"build_ext": CMakeBuild, "bdist_wheel": PlatformWheel},
    zip_safe=False,
)
