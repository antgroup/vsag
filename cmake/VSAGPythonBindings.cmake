include_guard (GLOBAL)

if (NOT ENABLE_PYBINDS)
    return ()
endif ()

# find_package(Python3) will find the python interpreter.
# To use a specific python, set the CMake variable Python3_EXECUTABLE.
# For example: cmake -DPython3_EXECUTABLE=/usr/bin/python3 ...
find_package (Python3 REQUIRED COMPONENTS Interpreter Development.Module)

message (STATUS "Python3 found:")
message (STATUS "  Executable: ${Python3_EXECUTABLE}")
message (STATUS "  Version: ${Python3_VERSION}")
message (STATUS "  Include dirs: ${Python3_INCLUDE_DIRS}")
message (STATUS "  Module suffix: ${Python3_SOABI}")

include (extern/pybind11/pybind11.cmake)

pybind11_add_module (_pyvsag
    python_bindings/module.cpp
    python_bindings/index_binding.cpp
    python_bindings/logging_binding.cpp)
target_compile_options (_pyvsag PRIVATE -fopenmp)
target_link_libraries (_pyvsag PRIVATE pybind11::module vsag)
if (NOT APPLE)
    target_link_options (_pyvsag PRIVATE -static-libgcc)
endif ()

message (STATUS "Building _pyvsag for Python ${Python3_VERSION}")
message (STATUS "Expected output: _pyvsag${Python3_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
