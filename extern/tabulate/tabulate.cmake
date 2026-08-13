include (FetchContent)

set (tabulate_urls
    https://github.com/p-ranav/tabulate/archive/3a58301067bbc03da89ae5a51b3e05b7da719d38.tar.gz
)
vsag_resolve_thirdparty_override (
    TABULATE 3a58301067bbc03da89ae5a51b3e05b7da719d38 tabulate_urls)
FetchContent_Declare (
    tabulate
    URL ${tabulate_urls}
    URL_HASH MD5=9d396f30fcc513abbb970773c8ddf8ff
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30)

FetchContent_MakeAvailable (tabulate)
