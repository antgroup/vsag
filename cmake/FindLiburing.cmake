
# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (ENABLE_LIBURING)
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
        pkg_check_modules(PC_LIBURING QUIET liburing)
    endif()

    if (PC_LIBURING_FOUND)
        message(STATUS "Found liburing via pkg-config: ${PC_LIBURING_LIBRARIES}")
        set(HAVE_LIBURING 1)
        add_definitions(-DHAVE_LIBURING=1)
    else()
        # fallback to find_library
        find_library(URING_LIBRARY NAMES io_uring PATHS /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu)
        if (URING_LIBRARY)
            message(STATUS "Found liburing: ${URING_LIBRARY}")
            set(HAVE_LIBURING 1)
            add_definitions(-DHAVE_LIBURING=1)
        else()
            message(WARNING "liburing not found, disabling io_uring support")
            set(HAVE_LIBURING 0)
            add_definitions(-DHAVE_LIBURING=0 -DNO_LIBURING=1)
        endif()
    endif()
else()
    message(STATUS "liburing support disabled by user")
    set(HAVE_LIBURING 0)
    add_definitions(-DHAVE_LIBURING=0 -DNO_LIBURING=1)
endif()
