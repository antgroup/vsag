// SPDX-License-Identifier: Apache-2.0
// Copyright 2026-present the vsag project
// Linux-only benchmark support; see README.md for scope and invocation.
#include <iostream>

#include "storage/tlv_section.h"
namespace vsag {
void
ReadForwardBlockPayload(StreamReader& reader,
                        const StreamBlockHeader& header,
                        const std::function<void(StreamReader&)>& deserialize) {
    std::cerr << "legacy_adapter_payload_bytes=" << header.value_len << '\n';
    ReadSeekableBlockPayload(reader, header, deserialize);
}
}  // namespace vsag
