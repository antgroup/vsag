@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

set LLVM_DIR=C:\wangzelin.wzl\compilation\LLVM18\bin

if exist build_win rmdir /s /q build_win
mkdir build_win

cmake -S . -B build_win -G Ninja ^
  -DCMAKE_C_COMPILER="%LLVM_DIR%\clang-cl.exe" ^
  -DCMAKE_CXX_COMPILER="%LLVM_DIR%\clang-cl.exe" ^
  -DCMAKE_LINKER="%LLVM_DIR%\lld-link.exe" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DENABLE_TESTS=OFF ^
  -DENABLE_EXAMPLES=OFF ^
  -DENABLE_TOOLS=OFF ^
  -DENABLE_INTEL_MKL=OFF ^
  -DENABLE_LIBAIO=OFF ^
  -DENABLE_PYBINDS=OFF ^
  -DENABLE_WERROR=OFF ^
  -DFETCHCONTENT_QUIET=OFF

if errorlevel 1 (
    echo CMake configuration FAILED
    exit /b 1
)

echo CMake configuration SUCCEEDED
echo.
echo To build, run:
echo   cmake --build build_win --parallel 16
