add_library (owned_c_probe OBJECT owned_c_probe.c)
add_library (owned_cxx_probe OBJECT owned_cxx_probe.cpp)
add_library (owned_asm_probe OBJECT owned_asm_probe.S)
add_executable (owned_link_probe owned_link_probe.c)
