
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

# AIO library detection
if (ENABLE_LIBAIO)
    find_library (AIO_LIBRARY aio)
    if (AIO_LIBRARY)
        message (STATUS "Found libaio: ${AIO_LIBRARY}")
        set (HAVE_LIBAIO 1)
        add_definitions (-DHAVE_LIBAIO=1)
    else ()
        message (WARNING "libaio not found, disabling async I/O support")
        set (HAVE_LIBAIO 0)
        add_definitions (-DHAVE_LIBAIO=0)
        add_definitions (-DNO_LIBAIO=1)
    endif ()
else ()
    message (STATUS "libaio support disabled by user")
    set (HAVE_LIBAIO 0)
    add_definitions (-DHAVE_LIBAIO=0)
    add_definitions (-DNO_LIBAIO=1)
endif ()
