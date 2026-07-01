#
# Copyright by The HDF Group.
# All rights reserved.
#
# This file is part of HDF5.  The full HDF5 copyright notice, including
# terms governing use, modification, and redistribution, is contained in
# the COPYING file, which can be found at the root of the source code
# distribution tree, or in https://www.hdfgroup.org/licenses.
# If you do not have access to either file, you may request a copy from
# help@hdfgroup.org.
#
#-----------------------------------------------------------------------------
# HDF5 Config file for compiling against hdf5 build/install directory
#-----------------------------------------------------------------------------

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was hdf5-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

string(TOUPPER hdf5 HDF5_PACKAGE_NAME)

set (${HDF5_PACKAGE_NAME}_VALID_COMPONENTS
    static
    shared
    C
    CXX
    Fortran
    HL
    CXX_HL
    Fortran_HL
    Java
    Tools
)

#-----------------------------------------------------------------------------
# User Options
#-----------------------------------------------------------------------------
# Languages:
#-----------------------------------------------------------------------------
set (${HDF5_PACKAGE_NAME}_BUILD_FORTRAN        OFF)
set (${HDF5_PACKAGE_NAME}_BUILD_CPP_LIB        ON)
set (${HDF5_PACKAGE_NAME}_BUILD_JAVA           OFF)
set (${HDF5_PACKAGE_NAME}_INSTALL_MOD_FORTRAN "")
#-----------------------------------------------------------------------------
# Features:
#-----------------------------------------------------------------------------
set (${HDF5_PACKAGE_NAME}_BUILD_HL_LIB          ON)
set (${HDF5_PACKAGE_NAME}_BUILD_SHARED_LIBS     YES)
set (${HDF5_PACKAGE_NAME}_BUILD_STATIC_LIBS     YES)
set (${HDF5_PACKAGE_NAME}_ENABLE_THREADSAFE     OFF)
set (${HDF5_PACKAGE_NAME}_ENABLE_PARALLEL       OFF)
set (${HDF5_PACKAGE_NAME}_DEFAULT_API_VERSION  "v114")
set (${HDF5_PACKAGE_NAME}_ENABLE_DEPRECATED_SYMBOLS ON)
set (${HDF5_PACKAGE_NAME}_BUILD_DIMENSION_SCALES_WITH_NEW_REF )
#-----------------------------------------------------------------------------
set (${HDF5_PACKAGE_NAME}_BUILD_TOOLS           ON)
set (${HDF5_PACKAGE_NAME}_BUILD_HL_GIF_TOOLS    OFF)
set (${HDF5_PACKAGE_NAME}_BUILD_STATIC_TOOLS    OFF)
#-----------------------------------------------------------------------------
set (${HDF5_PACKAGE_NAME}_ENABLE_NONSTANDARD_FEATURE_FLOAT16 OFF)
#-----------------------------------------------------------------------------
set (${HDF5_PACKAGE_NAME}_ENABLE_Z_LIB_SUPPORT  ON)
set (${HDF5_PACKAGE_NAME}_ENABLE_SZIP_SUPPORT   OFF)
set (${HDF5_PACKAGE_NAME}_ENABLE_SZIP_ENCODING  ON)
#-----------------------------------------------------------------------------
set (${HDF5_PACKAGE_NAME}_ENABLE_MAP_API        OFF)
set (${HDF5_PACKAGE_NAME}_ENABLE_DIRECT_VFD     OFF)
set (${HDF5_PACKAGE_NAME}_ENABLE_MIRROR_VFD     OFF)
set (${HDF5_PACKAGE_NAME}_ENABLE_ROS3_VFD       OFF)
set (${HDF5_PACKAGE_NAME}_ENABLE_HDFS_VFD       OFF)
set (${HDF5_PACKAGE_NAME}_ENABLE_SUBFILING_VFD  OFF)
set (${HDF5_PACKAGE_NAME}_ENABLE_PLUGIN_SUPPORT OFF)
#-----------------------------------------------------------------------------
set (${HDF5_PACKAGE_NAME}_PACKAGE_EXTLIBS       OFF)
set (${HDF5_PACKAGE_NAME}_EXPORT_LIBRARIES      hdf5-static;hdf5-shared;hdf5_tools-static;hdf5_tools-shared;hdf5_hl-static;hdf5_hl-shared;hdf5_cpp-static;hdf5_cpp-shared;hdf5_hl_cpp-static;hdf5_hl_cpp-shared)
set (${HDF5_PACKAGE_NAME}_ARCHITECTURE         "")
set (${HDF5_PACKAGE_NAME}_TOOLSET              "")

#-----------------------------------------------------------------------------
# Dependencies
#-----------------------------------------------------------------------------
if (${HDF5_PACKAGE_NAME}_ENABLE_PARALLEL)
  set (${HDF5_PACKAGE_NAME}_MPI_C_INCLUDE_PATH "")
  set (${HDF5_PACKAGE_NAME}_MPI_C_LIBRARIES    "")
  if (${HDF5_PACKAGE_NAME}_BUILD_FORTRAN)
    set (${HDF5_PACKAGE_NAME}_MPI_Fortran_INCLUDE_PATH "")
    set (${HDF5_PACKAGE_NAME}_MPI_Fortran_LIBRARIES    "")
  endif ()
  set (${HDF5_PACKAGE_NAME}_PARALLEL_FILTERED_WRITES )
  set (${HDF5_PACKAGE_NAME}_LARGE_PARALLEL_IO        )

  find_package(MPI QUIET REQUIRED)
endif ()

if (${HDF5_PACKAGE_NAME}_ENABLE_THREADSAFE OR ${HDF5_PACKAGE_NAME}_ENABLE_SUBFILING_VFD)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads QUIET REQUIRED)
endif ()

if (${HDF5_PACKAGE_NAME}_BUILD_JAVA)
  set (${HDF5_PACKAGE_NAME}_JAVA_INCLUDE_DIRS
      ${PACKAGE_PREFIX_DIR}/lib/jarhdf5-1.14.4.jar
      ${PACKAGE_PREFIX_DIR}/lib/slf4j-api-2.0.6.jar
      ${PACKAGE_PREFIX_DIR}/lib/slf4j-nop-2.0.6.jar
  )
endif ()

#-----------------------------------------------------------------------------
# Directories
#-----------------------------------------------------------------------------
set (${HDF5_PACKAGE_NAME}_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include" "${${HDF5_PACKAGE_NAME}_MPI_C_INCLUDE_DIRS}")

set (${HDF5_PACKAGE_NAME}_SHARE_DIR "${PACKAGE_PREFIX_DIR}/cmake")
set_and_check (${HDF5_PACKAGE_NAME}_BUILD_DIR "${PACKAGE_PREFIX_DIR}")

if (${HDF5_PACKAGE_NAME}_BUILD_CPP_LIB)
  set (${HDF5_PACKAGE_NAME}_INCLUDE_DIR_CPP "${PACKAGE_PREFIX_DIR}/include")
endif ()

if (${HDF5_PACKAGE_NAME}_BUILD_HL_LIB)
  set (${HDF5_PACKAGE_NAME}_INCLUDE_DIR_HL "${PACKAGE_PREFIX_DIR}/include")
endif ()

if (${HDF5_PACKAGE_NAME}_BUILD_HL_LIB AND ${HDF5_PACKAGE_NAME}_BUILD_CPP_LIB)
  set (${HDF5_PACKAGE_NAME}_INCLUDE_DIR_HL_CPP "${PACKAGE_PREFIX_DIR}/include")
endif ()

if (${HDF5_PACKAGE_NAME}_BUILD_TOOLS)
  set (${HDF5_PACKAGE_NAME}_INCLUDE_DIR_TOOLS "${PACKAGE_PREFIX_DIR}/include")
  set_and_check (${HDF5_PACKAGE_NAME}_TOOLS_DIR "${PACKAGE_PREFIX_DIR}/bin")
endif ()

#-----------------------------------------------------------------------------
# Version Strings
#-----------------------------------------------------------------------------
set (${HDF5_PACKAGE_NAME}_VERSION_STRING 1.14.4)
set (${HDF5_PACKAGE_NAME}_VERSION_MAJOR  1.14)
set (${HDF5_PACKAGE_NAME}_VERSION_MINOR  4)

#-----------------------------------------------------------------------------
# Don't include targets if this file is being picked up by another
# project which has already built hdf5 as a subproject
#-----------------------------------------------------------------------------
if (NOT TARGET "hdf5")
  if (${HDF5_PACKAGE_NAME}_ENABLE_Z_LIB_SUPPORT AND ${HDF5_PACKAGE_NAME}_PACKAGE_EXTLIBS)
    include (${PACKAGE_PREFIX_DIR}/cmake/-targets.cmake)
  endif ()
  if (${HDF5_PACKAGE_NAME}_ENABLE_SZIP_SUPPORT AND ${HDF5_PACKAGE_NAME}_PACKAGE_EXTLIBS)
    include (${PACKAGE_PREFIX_DIR}/cmake/-targets.cmake)
  endif ()
  include (${PACKAGE_PREFIX_DIR}/cmake/hdf5-targets.cmake)
endif ()

# Handle default component(static) :
if (NOT ${HDF5_PACKAGE_NAME}_FIND_COMPONENTS)
  if (${HDF5_PACKAGE_NAME}_BUILD_STATIC_LIBS)
    set (${HDF5_PACKAGE_NAME}_LIB_TYPE)
    set (${HDF5_PACKAGE_NAME}_FIND_COMPONENTS C HL static)
    set (${HDF5_PACKAGE_NAME}_FIND_REQUIRED_static_C true)
  else ()
    set (${HDF5_PACKAGE_NAME}_LIB_TYPE)
    set (${HDF5_PACKAGE_NAME}_FIND_COMPONENTS C HL shared)
    set (${HDF5_PACKAGE_NAME}_FIND_REQUIRED_shared_C true)
  endif ()
endif ()

# Handle requested components:
list (REMOVE_DUPLICATES ${HDF5_PACKAGE_NAME}_FIND_COMPONENTS)
foreach (comp IN LISTS ${HDF5_PACKAGE_NAME}_FIND_COMPONENTS)
  if (comp STREQUAL "shared")
    list (REMOVE_ITEM ${HDF5_PACKAGE_NAME}_FIND_COMPONENTS ${comp})
    set (${HDF5_PACKAGE_NAME}_LIB_TYPE ${${HDF5_PACKAGE_NAME}_LIB_TYPE} ${comp})

    if (${HDF5_PACKAGE_NAME}_BUILD_FORTRAN AND ${HDF5_PACKAGE_NAME}_INSTALL_MOD_FORTRAN STREQUAL "SHARED")
      set (${HDF5_PACKAGE_NAME}_INCLUDE_DIR_FORTRAN "${PACKAGE_PREFIX_DIR}/include/shared")
    endif ()
  elseif (comp STREQUAL "static")
    list (REMOVE_ITEM ${HDF5_PACKAGE_NAME}_FIND_COMPONENTS ${comp})
    set (${HDF5_PACKAGE_NAME}_LIB_TYPE ${${HDF5_PACKAGE_NAME}_LIB_TYPE} ${comp})

    if (${HDF5_PACKAGE_NAME}_BUILD_FORTRAN AND ${HDF5_PACKAGE_NAME}_INSTALL_MOD_FORTRAN STREQUAL "STATIC")
      set (${HDF5_PACKAGE_NAME}_INCLUDE_DIR_FORTRAN "${PACKAGE_PREFIX_DIR}/include/static")
    endif ()
  endif ()
endforeach ()
foreach (libtype IN LISTS ${HDF5_PACKAGE_NAME}_LIB_TYPE)
  foreach (comp IN LISTS ${HDF5_PACKAGE_NAME}_FIND_COMPONENTS)
    set (hdf5_comp2)
    if (comp STREQUAL "C")
      set (hdf5_comp "hdf5")
    elseif (comp STREQUAL "CXX")
      set (hdf5_comp "hdf5_cpp")
    elseif (comp STREQUAL "HL")
      set (hdf5_comp "hdf5_hl")
    elseif (comp STREQUAL "CXX_HL")
      set (hdf5_comp "hdf5_hl_cpp")
    elseif (comp STREQUAL "Java")
      set (hdf5_comp "hdf5_java")
    elseif (comp STREQUAL "Tools")
      set (hdf5_comp "hdf5_tools")
    elseif (comp STREQUAL "Fortran")
      set (hdf5_comp2 "hdf5_f90cstub")
      set (hdf5_comp "hdf5_fortran")
    elseif (comp STREQUAL "Fortran_HL")
      set (hdf5_comp2 "hdf5_hl_f90cstub")
      set (hdf5_comp "hdf5_hl_fortran")
    endif ()
    if (comp STREQUAL "Java")
      list (FIND ${HDF5_PACKAGE_NAME}_EXPORT_LIBRARIES "${hdf5_comp}" HAVE_COMP)
    else ()
      list (FIND ${HDF5_PACKAGE_NAME}_EXPORT_LIBRARIES "${hdf5_comp}-${libtype}" HAVE_COMP)
    endif ()
    if (${HAVE_COMP} LESS 0)
      if (comp STREQUAL "Java")
        set (${HDF5_PACKAGE_NAME}_${comp}_FOUND 0)
      else ()
        set (${HDF5_PACKAGE_NAME}_${libtype}_${comp}_FOUND 0)
      endif ()
    else ()
      if (hdf5_comp2)
        list (FIND ${HDF5_PACKAGE_NAME}_EXPORT_LIBRARIES "${hdf5_comp2}-${libtype}" HAVE_COMP2)
        if (${HAVE_COMP2} LESS 0)
          set (${HDF5_PACKAGE_NAME}_${libtype}_${comp}_FOUND 0)
        else ()
          set (${HDF5_PACKAGE_NAME}_${libtype}_${comp}_FOUND 1)
          string(TOUPPER ${HDF5_PACKAGE_NAME}_${comp}_${libtype}_LIBRARY COMP_LIBRARY)
          set (${COMP_LIBRARY} ${${COMP_LIBRARY}} ${hdf5_comp2}-${libtype} ${hdf5_comp}-${libtype})
        endif ()
      elseif (comp STREQUAL "Java")
        set (${HDF5_PACKAGE_NAME}_${comp}_FOUND 1)
        string(TOUPPER ${HDF5_PACKAGE_NAME}_${comp}_LIBRARY COMP_LIBRARY)
        set (${COMP_LIBRARY} ${${COMP_LIBRARY}} ${hdf5_comp})
      else ()
        set (${HDF5_PACKAGE_NAME}_${libtype}_${comp}_FOUND 1)
        string(TOUPPER ${HDF5_PACKAGE_NAME}_${comp}_${libtype}_LIBRARY COMP_LIBRARY)
        set (${COMP_LIBRARY} ${${COMP_LIBRARY}} ${hdf5_comp}-${libtype})
      endif ()
    endif ()
  endforeach ()
endforeach ()

foreach (libtype IN LISTS ${HDF5_PACKAGE_NAME}_LIB_TYPE)
  check_required_components(${HDF5_PACKAGE_NAME}_${libtype})
endforeach ()
