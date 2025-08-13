
include(FetchContent)

FetchContent_Declare(
        robin
        URL https://github.com/Tessil/robin-map/archive/refs/tags/v1.4.0.tar.gz
        # this url is maintained by the vsag project, if it's broken, please try
        #  the latest commit or contact the vsag project

        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_MakeAvailable(robin)
include_directories(${robin_SOURCE_DIR}/include)