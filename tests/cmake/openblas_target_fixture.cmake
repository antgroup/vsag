if (NOT DEFINED VSAG_SOURCE_DIR)
    message (FATAL_ERROR "VSAG_SOURCE_DIR is required")
endif ()

include ("${VSAG_SOURCE_DIR}/cmake/VSAGHelpers.cmake")

vsag_get_openblas_target_arg (openblas_target_arg)
message (STATUS "OPENBLAS_TARGET_ARG=${openblas_target_arg}")
