# VSAG 多构建目录标准重建命令清单

本文整理当前仓库中 5 个构建目录各自对应的一组标准重建命令，目标是复现这些目录当前在 `CMakeCache.txt` 中体现出来的配置状态。

适用源码目录：

- `/home/gubaoyuan.gby/vsag-main`

说明：

1. 下面的命令以“重建当前目录状态”为目标，不是按目录名字面含义猜测，而是按该目录现有 `CMakeCache.txt` 的实际开关组合整理。
2. 如果你希望彻底重建，建议先删除对应构建目录再重新执行。
3. 所有命令均使用 out-of-source 构建，因此每个目录会各自生成独立的 `_deps`、`.vsag-downloads`、第三方依赖和编译产物。

## 1. 当前 5 个目录对应的配置状态

| 构建目录 | Build Type | ENABLE_TESTS | ENABLE_MOCKIMPL | ENABLE_TOOLS | ENABLE_PYBINDS |
| --- | --- | --- | --- | --- | --- |
| `build-cache-dev` | `Release` | `ON` | `OFF` | `OFF` | `OFF` |
| `build-cache-test` | `Release` | `ON` | `OFF` | `ON` | `OFF` |
| `build-cache-tools` | `Debug` | `OFF` | `OFF` | `ON` | `OFF` |
| `build-release` | `Release` | `OFF` | `OFF` | `ON` | `OFF` |
| `build-test` | `Release` | `ON` | `ON` | `OFF` | `OFF` |

可以看到，这 5 个目录不是一次 `make release` 自动生成的，而是通过多次 `cmake -B <不同目录>` 配置出来的独立构建树。

## 2. 通用约定

下面所有命令都默认：

- 源码目录：`/home/gubaoyuan.gby/vsag-main`
- 安装前缀：`/usr/local/`
- 三方并行编译参数：`6`
- `ENABLE_INTEL_MKL=OFF`
- `ENABLE_LIBAIO=ON`
- `ENABLE_EXAMPLES=OFF`
- `ENABLE_PYBINDS=OFF`
- `CMAKE_EXPORT_COMPILE_COMMANDS=1`

如果你只想重建其中一个目录，直接复制对应小节里的代码块执行即可。

## 3. build-cache-dev

当前状态对应：

- `CMAKE_BUILD_TYPE=Release`
- `ENABLE_TESTS=ON`
- `ENABLE_MOCKIMPL=OFF`
- `ENABLE_TOOLS=OFF`

标准重建命令：

```bash
rm -rf /home/gubaoyuan.gby/vsag-main/build-cache-dev

cmake -S /home/gubaoyuan.gby/vsag-main \
  -B /home/gubaoyuan.gby/vsag-main/build-cache-dev \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DCMAKE_COLOR_DIAGNOSTICS=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local/ \
  -DNUM_BUILDING_JOBS=6 \
  -DENABLE_INTEL_MKL=OFF \
  -DENABLE_LIBAIO=ON \
  -DENABLE_TESTS=ON \
  -DENABLE_MOCKIMPL=OFF \
  -DENABLE_TOOLS=OFF \
  -DENABLE_EXAMPLES=OFF \
  -DENABLE_PYBINDS=OFF

cmake --build /home/gubaoyuan.gby/vsag-main/build-cache-dev --parallel 6
```

## 4. build-cache-test

当前状态对应：

- `CMAKE_BUILD_TYPE=Release`
- `ENABLE_TESTS=ON`
- `ENABLE_MOCKIMPL=OFF`
- `ENABLE_TOOLS=ON`

标准重建命令：

```bash
rm -rf /home/gubaoyuan.gby/vsag-main/build-cache-test

cmake -S /home/gubaoyuan.gby/vsag-main \
  -B /home/gubaoyuan.gby/vsag-main/build-cache-test \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DCMAKE_COLOR_DIAGNOSTICS=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local/ \
  -DNUM_BUILDING_JOBS=6 \
  -DENABLE_INTEL_MKL=OFF \
  -DENABLE_LIBAIO=ON \
  -DENABLE_TESTS=ON \
  -DENABLE_MOCKIMPL=OFF \
  -DENABLE_TOOLS=ON \
  -DENABLE_EXAMPLES=OFF \
  -DENABLE_PYBINDS=OFF

cmake --build /home/gubaoyuan.gby/vsag-main/build-cache-test --parallel 6
```

如果只想编译单测目标，也可以把第二条命令改成：

```bash
cmake --build /home/gubaoyuan.gby/vsag-main/build-cache-test --target unittests --parallel 6
```

## 5. build-cache-tools

当前状态对应：

- `CMAKE_BUILD_TYPE=Debug`
- `ENABLE_TESTS=OFF`
- `ENABLE_MOCKIMPL=OFF`
- `ENABLE_TOOLS=ON`

标准重建命令：

```bash
rm -rf /home/gubaoyuan.gby/vsag-main/build-cache-tools

cmake -S /home/gubaoyuan.gby/vsag-main \
  -B /home/gubaoyuan.gby/vsag-main/build-cache-tools \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DCMAKE_COLOR_DIAGNOSTICS=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local/ \
  -DNUM_BUILDING_JOBS=6 \
  -DENABLE_INTEL_MKL=OFF \
  -DENABLE_LIBAIO=ON \
  -DENABLE_TESTS=OFF \
  -DENABLE_MOCKIMPL=OFF \
  -DENABLE_TOOLS=ON \
  -DENABLE_EXAMPLES=OFF \
  -DENABLE_PYBINDS=OFF

cmake --build /home/gubaoyuan.gby/vsag-main/build-cache-tools --parallel 6
```

## 6. build-release

当前状态对应：

- `CMAKE_BUILD_TYPE=Release`
- `ENABLE_TESTS=OFF`
- `ENABLE_MOCKIMPL=OFF`
- `ENABLE_TOOLS=ON`

标准重建命令：

```bash
rm -rf /home/gubaoyuan.gby/vsag-main/build-release

cmake -S /home/gubaoyuan.gby/vsag-main \
  -B /home/gubaoyuan.gby/vsag-main/build-release \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DCMAKE_COLOR_DIAGNOSTICS=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local/ \
  -DNUM_BUILDING_JOBS=6 \
  -DENABLE_INTEL_MKL=OFF \
  -DENABLE_LIBAIO=ON \
  -DENABLE_TESTS=OFF \
  -DENABLE_MOCKIMPL=OFF \
  -DENABLE_TOOLS=ON \
  -DENABLE_EXAMPLES=OFF \
  -DENABLE_PYBINDS=OFF

cmake --build /home/gubaoyuan.gby/vsag-main/build-release --parallel 6
```

注意：

- 这份 `build-release` 的当前状态是 `ENABLE_TOOLS=ON`
- 因此它不等价于仓库根目录里“默认参数下”的 `make release`
- 如果直接执行 `make release`，默认只会生成 `build-release`，但默认并不会把 tools 打开

## 7. build-test

当前状态对应：

- `CMAKE_BUILD_TYPE=Release`
- `ENABLE_TESTS=ON`
- `ENABLE_MOCKIMPL=ON`
- `ENABLE_TOOLS=OFF`

标准重建命令：

```bash
rm -rf /home/gubaoyuan.gby/vsag-main/build-test

cmake -S /home/gubaoyuan.gby/vsag-main \
  -B /home/gubaoyuan.gby/vsag-main/build-test \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DCMAKE_COLOR_DIAGNOSTICS=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local/ \
  -DNUM_BUILDING_JOBS=6 \
  -DENABLE_INTEL_MKL=OFF \
  -DENABLE_LIBAIO=ON \
  -DENABLE_TESTS=ON \
  -DENABLE_MOCKIMPL=ON \
  -DENABLE_TOOLS=OFF \
  -DENABLE_EXAMPLES=OFF \
  -DENABLE_PYBINDS=OFF

cmake --build /home/gubaoyuan.gby/vsag-main/build-test --parallel 6
```

## 8. 一次性重建全部 5 个目录

如果你要一次性全部重建，可以按下面顺序逐个执行。

```bash
rm -rf /home/gubaoyuan.gby/vsag-main/build-cache-dev
rm -rf /home/gubaoyuan.gby/vsag-main/build-cache-test
rm -rf /home/gubaoyuan.gby/vsag-main/build-cache-tools
rm -rf /home/gubaoyuan.gby/vsag-main/build-release
rm -rf /home/gubaoyuan.gby/vsag-main/build-test

cmake -S /home/gubaoyuan.gby/vsag-main -B /home/gubaoyuan.gby/vsag-main/build-cache-dev -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_COLOR_DIAGNOSTICS=ON -DCMAKE_INSTALL_PREFIX=/usr/local/ -DNUM_BUILDING_JOBS=6 -DENABLE_INTEL_MKL=OFF -DENABLE_LIBAIO=ON -DENABLE_TESTS=ON -DENABLE_MOCKIMPL=OFF -DENABLE_TOOLS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_PYBINDS=OFF
cmake --build /home/gubaoyuan.gby/vsag-main/build-cache-dev --parallel 6

cmake -S /home/gubaoyuan.gby/vsag-main -B /home/gubaoyuan.gby/vsag-main/build-cache-test -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_COLOR_DIAGNOSTICS=ON -DCMAKE_INSTALL_PREFIX=/usr/local/ -DNUM_BUILDING_JOBS=6 -DENABLE_INTEL_MKL=OFF -DENABLE_LIBAIO=ON -DENABLE_TESTS=ON -DENABLE_MOCKIMPL=OFF -DENABLE_TOOLS=ON -DENABLE_EXAMPLES=OFF -DENABLE_PYBINDS=OFF
cmake --build /home/gubaoyuan.gby/vsag-main/build-cache-test --parallel 6

cmake -S /home/gubaoyuan.gby/vsag-main -B /home/gubaoyuan.gby/vsag-main/build-cache-tools -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_COLOR_DIAGNOSTICS=ON -DCMAKE_INSTALL_PREFIX=/usr/local/ -DNUM_BUILDING_JOBS=6 -DENABLE_INTEL_MKL=OFF -DENABLE_LIBAIO=ON -DENABLE_TESTS=OFF -DENABLE_MOCKIMPL=OFF -DENABLE_TOOLS=ON -DENABLE_EXAMPLES=OFF -DENABLE_PYBINDS=OFF
cmake --build /home/gubaoyuan.gby/vsag-main/build-cache-tools --parallel 6

cmake -S /home/gubaoyuan.gby/vsag-main -B /home/gubaoyuan.gby/vsag-main/build-release -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_COLOR_DIAGNOSTICS=ON -DCMAKE_INSTALL_PREFIX=/usr/local/ -DNUM_BUILDING_JOBS=6 -DENABLE_INTEL_MKL=OFF -DENABLE_LIBAIO=ON -DENABLE_TESTS=OFF -DENABLE_MOCKIMPL=OFF -DENABLE_TOOLS=ON -DENABLE_EXAMPLES=OFF -DENABLE_PYBINDS=OFF
cmake --build /home/gubaoyuan.gby/vsag-main/build-release --parallel 6

cmake -S /home/gubaoyuan.gby/vsag-main -B /home/gubaoyuan.gby/vsag-main/build-test -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_COLOR_DIAGNOSTICS=ON -DCMAKE_INSTALL_PREFIX=/usr/local/ -DNUM_BUILDING_JOBS=6 -DENABLE_INTEL_MKL=OFF -DENABLE_LIBAIO=ON -DENABLE_TESTS=ON -DENABLE_MOCKIMPL=ON -DENABLE_TOOLS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_PYBINDS=OFF
cmake --build /home/gubaoyuan.gby/vsag-main/build-test --parallel 6
```

## 9. 与 Makefile 目标的关系

为了避免误解，需要区分“目录名”和“Makefile 目标名”：

1. 当前 `build-release` 不是默认 `make release` 的纯默认状态，因为它打开了 `ENABLE_TOOLS=ON`
2. 当前 `build-test` 不是 `make test` 的直接结果，因为 `make test` 默认使用 `./build/` 且是 `Debug`
3. 当前 `build-cache-dev` 也不是 `make dev` 的直接结果，因为 `make dev` 默认是 `Debug`，并且还会打开 `PYBINDS`、`TOOLS`、`EXAMPLES`、`MOCKIMPL`

因此，这 5 个目录最合理的管理方式是：

1. 把它们视为 5 套独立的 CMake 构建树
2. 后续继续使用固定的 `-B` 目录名维护各自用途
3. 不要假设单个 `make release` 可以自动恢复全部目录