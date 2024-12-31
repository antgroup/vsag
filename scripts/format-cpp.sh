#!/bin/bash

find include/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
find src/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
find python_bindings/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
find examples/cpp/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
find mockimpl/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
find tests/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
find tools/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
