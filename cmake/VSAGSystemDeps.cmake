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

include_guard (GLOBAL)

set (HAVE_LIBAIO 0)
unset (AIO_LIBRARY)

if (ENABLE_LIBAIO)
    find_library (AIO_LIBRARY aio)
    if (AIO_LIBRARY)
        set (HAVE_LIBAIO 1)
        message (STATUS "Found libaio: ${AIO_LIBRARY}")
    else ()
        message (WARNING "libaio not found, disabling async I/O support")
    endif ()
else ()
    message (STATUS "libaio support disabled by user")
endif ()

add_compile_definitions (HAVE_LIBAIO=${HAVE_LIBAIO})
if (NOT HAVE_LIBAIO)
    add_compile_definitions (NO_LIBAIO=1)
endif ()

find_program (CCACHE_PROGRAM ccache)
if (ENABLE_CCACHE)
    if (CCACHE_PROGRAM)
        message (STATUS "Compiling with ccache")
        set_property (GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
        set_property (GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
    else ()
        message (STATUS "ENABLE_CCACHE is ON, but ccache was not found")
    endif ()
endif ()
