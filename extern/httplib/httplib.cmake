
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

include(FetchContent)

# cpp-httplib is a header-only library, we only need the main header
set(httplib_urls
    https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.35.0.tar.gz
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/cpp-httplib/v0.35.0.tar.gz
)

if(DEFINED ENV{VSAG_THIRDPARTY_HTTPLIB})
  message(STATUS "Using local path for httplib: $ENV{VSAG_THIRDPARTY_HTTPLIB}")
  list(PREPEND httplib_urls "$ENV{VSAG_THIRDPARTY_HTTPLIB}")
endif()

FetchContent_Declare(
    httplib
    URL ${httplib_urls}
    URL_HASH MD5=564fdf8b1acbc780fadb2e55fe6a0db6
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30
)

FetchContent_MakeAvailable(httplib)

# httplib is header-only, just add include directory
include_directories(${httplib_SOURCE_DIR})

# Define macro to indicate httplib is available
add_definitions(-DCPP_HTTPLIB_OPENSSL_SUPPORT)
