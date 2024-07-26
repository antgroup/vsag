include(FetchContent)

FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
        URL_HASH MD5=49e92f92244021912a56935918c927d0
)

FetchContent_MakeAvailable(pybind11)
