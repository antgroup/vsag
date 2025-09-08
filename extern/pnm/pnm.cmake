
set(PNM_LIBRARY -Wl,--whole-archive ${CMAKE_CURRENT_SOURCE_DIR}/extern/pnm/lib/libpnm_engine_static.a -Wl,--no-whole-archive -lz numa dl rt)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/pnm/include)