
set (FACTORY_SRC
        engine.cpp
        factory.cpp
        resource.cpp
        resource_owner_wrapper.h
)

add_library (factory OBJECT ${FACTORY_SRC})
target_link_libraries (factory PUBLIC fmt::fmt coverage_config)
maybe_add_dependencies (factory spdlog)
