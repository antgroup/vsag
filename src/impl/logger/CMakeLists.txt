
set (LOGGER_SRC
    default_logger.cpp
    default_logger.h
    logger.h
    logger.cpp
)

add_library (logger OBJECT ${LOGGER_SRC})
target_link_libraries (logger PRIVATE fmt::fmt coverage_config)
maybe_add_dependencies (logger spdlog)
