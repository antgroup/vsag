include_guard (GLOBAL)

set (roaringbitmap_version 3.0.1)
set (roaringbitmap_release_url
    "https://github.com/RoaringBitmap/CRoaring/releases/download/v${roaringbitmap_version}")
set (roaringbitmap_source_dir
    "${DOWNLOAD_DIR}/roaringbitmap-v${roaringbitmap_version}")
file (MAKE_DIRECTORY "${roaringbitmap_source_dir}")

function (vsag_download_roaringbitmap_asset asset_name asset_sha256)
    set (asset_path "${roaringbitmap_source_dir}/${asset_name}")
    file (LOCK "${asset_path}.lock"
        GUARD FUNCTION
        TIMEOUT 120
        RESULT_VARIABLE lock_result)
    if (NOT "${lock_result}" STREQUAL "0")
        message (FATAL_ERROR "Failed to lock the CRoaring cache for ${asset_name}: ${lock_result}")
    endif ()

    if (EXISTS "${asset_path}")
        file (SHA256 "${asset_path}" cached_sha256)
        if ("${cached_sha256}" STREQUAL "${asset_sha256}")
            message (STATUS "Using cached CRoaring asset: ${asset_path}")
            return ()
        endif ()
        message (STATUS "Replacing CRoaring asset with unexpected hash: ${asset_path}")
        file (REMOVE "${asset_path}")
    endif ()

    set (asset_url "${roaringbitmap_release_url}/${asset_name}")
    set (temporary_path "${asset_path}.tmp")
    file (REMOVE "${temporary_path}")
    message (STATUS "Downloading CRoaring asset: ${asset_url}")
    file (DOWNLOAD
        "${asset_url}"
        "${temporary_path}"
        STATUS download_status
        LOG download_log
        SHOW_PROGRESS
        TLS_VERIFY ON
        INACTIVITY_TIMEOUT 15
        TIMEOUT 90)
    list (GET download_status 0 download_code)
    list (GET download_status 1 download_message)
    message (VERBOSE "CRoaring download log for ${asset_url}:\n${download_log}")
    if (NOT download_code EQUAL 0)
        file (REMOVE "${temporary_path}")
        message (FATAL_ERROR
            "Failed to download CRoaring ${asset_name} from ${asset_url}: ${download_message}")
    endif ()

    file (SHA256 "${temporary_path}" downloaded_sha256)
    if (NOT "${downloaded_sha256}" STREQUAL "${asset_sha256}")
        file (REMOVE "${temporary_path}")
        message (FATAL_ERROR
            "CRoaring ${asset_name} SHA-256 mismatch from ${asset_url}: expected "
            "${asset_sha256}, got ${downloaded_sha256}")
    endif ()
    file (RENAME "${temporary_path}" "${asset_path}")
endfunction ()

# These are the official amalgamation assets attached to the pinned CRoaring release.
vsag_download_roaringbitmap_asset (
    roaring.c d1fe3f22b11968d6edf6648b55c96a8df57f5875453be0409c8d5b4afeec3353)
vsag_download_roaringbitmap_asset (
    roaring.h 57e0abc5e3f48e7c47743e49f1261747b41686993949896b3809f255ab5e282b)
vsag_download_roaringbitmap_asset (
    roaring.hh 81d9cfc5704ea8fddd53bfe0b20b4678c8d3143d1a47ec684c009a9100c1f530)

# Preserve upstream target names for compatibility, even though roaring also exposes this include
# path directly. Match upstream BUILD_SHARED_LIBS behavior without installing CRoaring.
add_library (roaring-headers INTERFACE)
add_library (roaring-headers-cpp INTERFACE)
target_include_directories (roaring-headers INTERFACE "${roaringbitmap_source_dir}")
target_include_directories (roaring-headers-cpp INTERFACE "${roaringbitmap_source_dir}")

add_library (roaring "${roaringbitmap_source_dir}/roaring.c")
add_library (roaring::roaring ALIAS roaring)
set_target_properties (roaring PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    SOVERSION 15
    VERSION 3.0.0
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories (roaring PUBLIC "${roaringbitmap_source_dir}")
target_link_libraries (roaring PUBLIC roaring-headers roaring-headers-cpp)

if (DISABLE_AVX_FORCE OR NOT COMPILER_AVX_SUPPORTED)
    target_compile_definitions (roaring PRIVATE ROARING_DISABLE_AVX=1)
endif ()
if (DISABLE_AVX512_FORCE OR NOT COMPILER_AVX512_SUPPORTED)
    target_compile_definitions (roaring PRIVATE CROARING_COMPILER_SUPPORTS_AVX512=0)
endif ()
if (CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    target_compile_options (roaring PRIVATE -Wno-unused-function)
endif ()

if (NOT TARGET vsag_roaring_headers)
    add_library (vsag_roaring_headers INTERFACE)
endif ()
target_include_directories (vsag_roaring_headers INTERFACE "${roaringbitmap_source_dir}")
