
add_subdirectory (simd)
add_subdirectory (io)
add_subdirectory (quantization)

#https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz
if (NOT EXISTS "/tmp/openjdk-11.0.2_linux-x64_bin.tar.gz")
    file(DOWNLOAD
            http://aivolvo-dev.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/tbase/third-party/openjdk-11.0.2_linux-x64_bin.tar.gz
            /tmp/openjdk-11.0.2_linux-x64_bin.tar.gz
            TIMEOUT 60
            STATUS status
            TLS_VERIFY OFF
    )

    list(GET status 0 error_code)
    list(GET status 1 error_msg)

    if (error_code)
        message(FATAL_ERROR "Failed to download openjdk-11.0.2_linux-x64_bin.tar.gz: ${error_msg}")
    else ()
        message(STATUS "Downloaded openjdk-11.0.2_linux-x64_bin.tar.gz successfully")
        execute_process(COMMAND tar -xvf /tmp/openjdk-11.0.2_linux-x64_bin.tar.gz
                WORKING_DIRECTORY /tmp
                RESULT_VARIABLE result)
        if (result)
            message(FATAL_ERROR "Failed to extract openjdk-11.0.2_linux-x64_bin.tar.gz: ${result}")
        else ()
            message(STATUS "Extracted openjdk-11.0.2_linux-x64_bin.tar.gz successfully")
        endif ()
    endif ()
elseif (NOT EXISTS "/tmp/jdk-11.0.2")
    execute_process(COMMAND tar -xvf /tmp/openjdk-11.0.2_linux-x64_bin.tar.gz
            WORKING_DIRECTORY /tmp
            RESULT_VARIABLE result)
    if (result)
        message(FATAL_ERROR "Failed to extract openjdk-11.0.2_linux-x64_bin.tar.gz: ${result}")
    else ()
        message(STATUS "Extracted openjdk-11.0.2_linux-x64_bin.tar.gz successfully")
    endif ()
else ()
    execute_process(COMMAND /tmp/jdk-11.0.2/bin/java -jar ${CMAKE_BINARY_DIR}/antlr4/install/antlr-4.13.2-complete.jar -Dlanguage=Cpp ${CMAKE_CURRENT_SOURCE_DIR}/attr/grammar/FC.g4 -listener -visitor -o ${CMAKE_CURRENT_SOURCE_DIR}/attr/
            WORKING_DIRECTORY ./
            RESULT_VARIABLE result)
    if (result)
        message(FATAL_ERROR "Failed to generate antlr Cpp: ${result}")
    else ()
        message(STATUS "Generate antlr Cpp successfully")
    endif ()
endif ()

file (GLOB CPP_SRCS "*.cpp")
list (FILTER CPP_SRCS EXCLUDE REGEX "_test.cpp")

file(GLOB CPP_ATTR_SRCS "attr/*.cpp")
list(FILTER CPP_ATTR_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_IMPL_SRCS "impl/*.cpp")
file (GLOB CPP_IMPL_EXTRA_SRCS "impl/**/*.cpp")
list (APPEND CPP_IMPL_SRCS ${CPP_IMPL_EXTRA_SRCS})
list (FILTER CPP_IMPL_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_INDEX_SRCS "index/*.cpp")
list (FILTER CPP_INDEX_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_DATA_CELL_SRCS "data_cell/*.cpp")
list (FILTER CPP_DATA_CELL_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_ALGORITHM_SRCS "algorithm/*.cpp")
file (GLOB CPP_ALGORITHM_EXTRA_SRCS "algorithm/**/*.cpp")
list (APPEND CPP_ALGORITHM_SRCS ${CPP_ALGORITHM_EXTRA_SRCS})
list (FILTER CPP_ALGORITHM_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_UTILS_SRCS "utils/*.cpp")
list (FILTER CPP_UTILS_SRCS EXCLUDE REGEX "_test.cpp")

set(VSAG_SRCS ${CPP_SRCS} ${CPP_ATTR_SRCS} ${CPP_INDEX_SRCS} ${CPP_IMPL_SRCS}
        ${CPP_DATA_CELL_SRCS} ${CPP_ALGORITHM_SRCS} ${CPP_UTILS_SRCS})

add_library (vsag SHARED ${VSAG_SRCS})
add_library (vsag_static STATIC ${VSAG_SRCS})

set(VSAG_DEP_LIBS antlr4-runtime diskann pthread m dl simd io quantizer fmt::fmt-header-only nlohmann_json::nlohmann_json roaring)
target_link_libraries (vsag ${VSAG_DEP_LIBS} coverage_config)
target_link_libraries (vsag_static ${VSAG_DEP_LIBS} coverage_config)

maybe_add_dependencies(vsag antlr4 spdlog roaring openblas boost mkl)
maybe_add_dependencies(vsag_static antlr4 spdlog roaring openblas boost mkl)
