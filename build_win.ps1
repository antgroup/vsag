$ErrorActionPreference = "Continue"

# Initialize MSVC environment
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

$LLVM_DIR = "C:\wangzelin.wzl\compilation\LLVM18\bin"
$NINJA_DIR = "C:\wangzelin.wzl\compilation\ninja"

if (Test-Path build_win) {
    Remove-Item -Recurse -Force build_win
}
New-Item -ItemType Directory -Path build_win | Out-Null

# CMake configure
cmake -S . -B build_win -G Ninja `
    -DCMAKE_MAKE_PROGRAM="$NINJA_DIR\ninja_wrap.exe" `
    -DCMAKE_C_COMPILER="$LLVM_DIR\clang-cl.exe" `
    -DCMAKE_CXX_COMPILER="$LLVM_DIR\clang-cl.exe" `
    -DCMAKE_LINKER="$LLVM_DIR\lld-link.exe" `
    -DCMAKE_RC_COMPILER=rc `
    -DCMAKE_BUILD_TYPE=Release `
    -DENABLE_TESTS=OFF `
    -DENABLE_EXAMPLES=OFF `
    -DENABLE_TOOLS=OFF `
    -DENABLE_INTEL_MKL=OFF `
    -DENABLE_LIBAIO=OFF `
    -DENABLE_PYBINDS=OFF `
    -DENABLE_WERROR=OFF `
    -DFETCHCONTENT_QUIET=OFF

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration FAILED"
    exit 1
}

Write-Host "CMake configuration SUCCEEDED" -ForegroundColor Green
Write-Host ""
Write-Host "Building..."

# CMake build
cmake --build build_win --parallel 10

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build FAILED"
    exit 1
}

Write-Host "Build SUCCEEDED" -ForegroundColor Green
