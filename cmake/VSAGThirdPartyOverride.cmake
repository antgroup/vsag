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

function (_vsag_normalize_thirdparty_component value output)
    string (TOUPPER "${value}" normalized)
    string (REGEX REPLACE "[^A-Z0-9]+" "_" normalized "${normalized}")
    string (REGEX REPLACE "^_+" "" normalized "${normalized}")
    string (REGEX REPLACE "_+$" "" normalized "${normalized}")
    set (${output} "${normalized}" PARENT_SCOPE)
endfunction ()

function (_vsag_classify_thirdparty_pin dependency pin kind_output component_output digest_output)
    _vsag_normalize_thirdparty_component ("${dependency}" dependency_component)

    set (numeric_version "")
    string (REGEX MATCH "([0-9]+(\\.[0-9]+)+)$" numeric_version "${pin}")
    if (numeric_version)
        string (LENGTH "${pin}" pin_length)
        string (LENGTH "${numeric_version}" version_length)
        math (EXPR prefix_length "${pin_length} - ${version_length}")
        string (SUBSTRING "${pin}" 0 ${prefix_length} version_prefix)

        if (version_prefix MATCHES "[vV]$")
            string (REGEX REPLACE "[vV]$" "" version_prefix "${version_prefix}")
        endif ()
        _vsag_normalize_thirdparty_component ("${version_prefix}" prefix_component)

        if (NOT prefix_component OR prefix_component STREQUAL dependency_component)
            _vsag_normalize_thirdparty_component ("${numeric_version}" version_component)
            set (${kind_output} "VERSION" PARENT_SCOPE)
            set (${component_output} "${version_component}" PARENT_SCOPE)
            set (${digest_output} "" PARENT_SCOPE)
            return ()
        endif ()
    endif ()

    string (LENGTH "${pin}" pin_length)
    if ((pin_length EQUAL 40 OR pin_length EQUAL 64) AND pin MATCHES "^[0-9A-Fa-f]+$")
        string (TOUPPER "${pin}" commit_digest)
        set (${kind_output} "COMMIT" PARENT_SCOPE)
        set (${component_output} "" PARENT_SCOPE)
        set (${digest_output} "${commit_digest}" PARENT_SCOPE)
        return ()
    endif ()

    _vsag_normalize_thirdparty_component ("${pin}" tag_component)
    string (SHA256 tag_digest "${pin}")
    string (TOUPPER "${tag_digest}" tag_digest)
    set (${kind_output} "TAG" PARENT_SCOPE)
    set (${component_output} "${tag_component}" PARENT_SCOPE)
    set (${digest_output} "${tag_digest}" PARENT_SCOPE)
endfunction ()

function (vsag_thirdparty_pinned_variable dependency pin output)
    set (one_value_args HASH_LENGTH)
    set (multi_value_args COLLISION_PINS)
    cmake_parse_arguments (ARG "" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if (ARG_UNPARSED_ARGUMENTS)
        message (FATAL_ERROR
                 "Unexpected pinned-variable arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif ()
    if (ARG_HASH_LENGTH)
        set (hash_length "${ARG_HASH_LENGTH}")
    else ()
        set (hash_length 12)
    endif ()
    if (NOT hash_length MATCHES "^[0-9]+$")
        message (FATAL_ERROR "Pinned-variable hash length must be an integer")
    endif ()

    _vsag_normalize_thirdparty_component ("${dependency}" dependency_component)
    if (NOT dependency_component)
        message (FATAL_ERROR "A pinned variable requires a non-empty dependency name")
    endif ()
    if ("${pin}" STREQUAL "")
        message (FATAL_ERROR "A pinned variable requires a non-empty pin")
    endif ()

    _vsag_classify_thirdparty_pin (
        "${dependency}" "${pin}" pin_kind pin_component pin_digest)
    if (pin_kind STREQUAL "VERSION")
        set (pin_suffix "${pin_component}")
    else ()
        string (LENGTH "${pin_digest}" digest_length)
        if (hash_length LESS 1 OR hash_length GREATER digest_length)
            message (FATAL_ERROR
                     "Pinned-variable hash length must be between 1 and ${digest_length}")
        endif ()
        foreach (collision_pin IN LISTS ARG_COLLISION_PINS)
            if (collision_pin STREQUAL pin)
                continue ()
            endif ()
            _vsag_classify_thirdparty_pin (
                "${dependency}" "${collision_pin}" collision_kind collision_component
                collision_digest)
            if (NOT collision_kind STREQUAL pin_kind)
                continue ()
            endif ()
            if (pin_kind STREQUAL "TAG" AND NOT collision_component STREQUAL pin_component)
                continue ()
            endif ()

            set (prefix_is_unique FALSE)
            while (NOT prefix_is_unique)
                string (SUBSTRING "${pin_digest}" 0 ${hash_length} pin_prefix)
                string (SUBSTRING "${collision_digest}" 0 ${hash_length} collision_prefix)
                if (NOT pin_prefix STREQUAL collision_prefix)
                    set (prefix_is_unique TRUE)
                elseif (hash_length LESS digest_length)
                    math (EXPR hash_length "${hash_length} + 1")
                else ()
                    message (FATAL_ERROR
                             "Pins '${pin}' and '${collision_pin}' cannot have unique pinned "
                             "variables for dependency '${dependency}'")
                endif ()
            endwhile ()
        endforeach ()

        string (SUBSTRING "${pin_digest}" 0 ${hash_length} pin_prefix)
        if (pin_kind STREQUAL "COMMIT")
            set (pin_suffix "COMMIT_${pin_prefix}")
        elseif (pin_component)
            set (pin_suffix "TAG_${pin_component}_H${pin_prefix}")
        else ()
            set (pin_suffix "TAG_H${pin_prefix}")
        endif ()
    endif ()

    set (${output} "VSAG_THIRDPARTY_${dependency_component}_${pin_suffix}" PARENT_SCOPE)
endfunction ()

function (vsag_resolve_thirdparty_override dependency pin urls_variable)
    set (multi_value_args COLLISION_PINS)
    cmake_parse_arguments (ARG "" "" "${multi_value_args}" ${ARGN})
    if (ARG_UNPARSED_ARGUMENTS)
        message (FATAL_ERROR
                 "Unexpected third-party override arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif ()
    if (NOT DEFINED ${urls_variable})
        message (FATAL_ERROR "Third-party URL variable '${urls_variable}' is not defined")
    endif ()

    _vsag_normalize_thirdparty_component ("${dependency}" dependency_component)
    set (legacy_variable "VSAG_THIRDPARTY_${dependency_component}")
    vsag_thirdparty_pinned_variable (
        "${dependency}" "${pin}" pinned_variable COLLISION_PINS ${ARG_COLLISION_PINS})
    set (resolved_urls "${${urls_variable}}")

    if (DEFINED ENV{${pinned_variable}})
        list (PREPEND resolved_urls "$ENV{${pinned_variable}}")
        if (DEFINED ENV{${legacy_variable}})
            set (legacy_behavior "legacy fallback ${legacy_variable} is ignored")
        else ()
            set (legacy_behavior "legacy fallback ${legacy_variable} is unset")
        endif ()
        message (STATUS
                 "Third-party override: dependency=${dependency}, pin=${pin}, source=pinned, "
                 "variable=${pinned_variable}; ${legacy_behavior}; default URLs remain download "
                 "fallbacks")
    elseif (DEFINED ENV{${legacy_variable}})
        list (PREPEND resolved_urls "$ENV{${legacy_variable}}")
        message (DEPRECATION
                 "Third-party override: dependency=${dependency}, pin=${pin}, source=legacy, "
                 "variable=${legacy_variable}; use pinned variable ${pinned_variable}; default "
                 "URLs remain download fallbacks")
    else ()
        message (STATUS
                 "Third-party override: dependency=${dependency}, pin=${pin}, source=default, "
                 "variable=none; expected pinned variable ${pinned_variable}; deprecated legacy "
                 "fallback ${legacy_variable} is unset")
    endif ()

    set (${urls_variable} "${resolved_urls}" PARENT_SCOPE)
endfunction ()
