
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was vsagConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

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

# vsagConfig.cmake — entry point for `find_package(vsag CONFIG REQUIRED)`.
#
# Usage:
#
#   find_package(vsag CONFIG REQUIRED)
#   target_link_libraries(my_app PRIVATE vsag::vsag)
#
# Imported targets:
#   vsag::vsag — the shared library (recommended for consumption).
#
# vsag's public headers in <vsag/*.h> only depend on the C++ standard library,
# so no transitive find_dependency() calls are required here.

include ("${CMAKE_CURRENT_LIST_DIR}/vsagTargets.cmake")

check_required_components (vsag)
