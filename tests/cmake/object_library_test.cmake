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

if (NOT VSAG_SOURCE_DIR)
    message (FATAL_ERROR "VSAG_SOURCE_DIR is required")
endif ()

file (READ "${VSAG_SOURCE_DIR}/src/CMakeLists.txt" _vsag_cmake)
file (READ "${VSAG_SOURCE_DIR}/mockimpl/CMakeLists.txt" _mock_cmake)

function (require_text content text description)
    string (FIND "${content}" "${text}" _position)
    if (_position EQUAL -1)
        message (FATAL_ERROR "Missing ${description}")
    endif ()
endfunction ()

function (reject_text content text description)
    string (FIND "${content}" "${text}" _position)
    if (NOT _position EQUAL -1)
        message (FATAL_ERROR "Found ${description}")
    endif ()
endfunction ()

require_text ("${_vsag_cmake}" [=[add_library (vsag_objects OBJECT ${VSAG_SRCS})]=]
              "single VSAG source-owning OBJECT target")
require_text ("${_vsag_cmake}" [=[add_library (vsag SHARED $<TARGET_OBJECTS:vsag_objects>)]=]
              "shared VSAG object consumption")
require_text ("${_vsag_cmake}"
              [=[add_library (vsag_static STATIC $<TARGET_OBJECTS:vsag_objects>)]=]
              "static VSAG object consumption")
reject_text ("${_vsag_cmake}" [=[add_library (vsag SHARED ${VSAG_SRCS})]=]
             "duplicate shared VSAG source compilation")
reject_text ("${_vsag_cmake}" [=[add_library (vsag_static STATIC ${VSAG_SRCS})]=]
             "duplicate static VSAG source compilation")

require_text ("${_mock_cmake}"
              [=[add_library (vsag_mockimpl_objects OBJECT ${MOCK_SRCS})]=]
              "single mock source-owning OBJECT target")
require_text ("${_mock_cmake}"
              [=[add_library (vsag_mockimpl SHARED $<TARGET_OBJECTS:vsag_mockimpl_objects>)]=]
              "shared mock object consumption")
require_text ("${_mock_cmake}"
              [=[add_library (vsag_mockimpl_static STATIC $<TARGET_OBJECTS:vsag_mockimpl_objects>)]=]
              "static mock object consumption")
reject_text ("${_mock_cmake}" [=[add_library (vsag_mockimpl SHARED ${MOCK_SRCS})]=]
             "duplicate shared mock source compilation")
reject_text ("${_mock_cmake}" [=[add_library (vsag_mockimpl_static STATIC ${MOCK_SRCS})]=]
             "duplicate static mock source compilation")

message (STATUS "VSAG and mock source lists each have one object-library owner")
