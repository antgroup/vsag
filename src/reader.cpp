#include <fmt/format.h>

#include <future>

#include "vsag/readerset.h"
#include "vsag_exception.h"

namespace vsag {

bool
Reader::MultiRead(uint8_t* dests, const uint64_t* lens, const uint64_t* offsets, uint64_t count) {
    std::atomic<bool> succeed(true);
    std::string error_message;
    std::atomic<uint64_t> counter(count);
    std::promise<void> total_promise;
    uint8_t* dest = dests;
    auto total_future = total_promise.get_future();

    for (int i = 0; i < count; ++i) {
        uint64_t offset = offsets[i];
        uint64_t size = lens[i];
        if (size + offset > Size()) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("ReaderIO MultiReadImpl size mismatch: "
                                            "offset {}, size {}, total size {}",
                                            offset,
                                            size,
                                            Size()));
        }
        auto callback = [&counter, &total_promise, &succeed, &error_message](
                            IOErrorCode code, const std::string& message) {
            if (code != vsag::IOErrorCode::IO_SUCCESS) {
                bool expected = true;
                if (succeed.compare_exchange_strong(expected, false)) {
                    error_message = message;
                }
            }
            if (--counter == 0) {
                total_promise.set_value();
            }
        };
        AsyncRead(offset, size, dest, callback);
        dest += size;
    }

    total_future.wait();
    return succeed.load();
}

}  // namespace vsag
