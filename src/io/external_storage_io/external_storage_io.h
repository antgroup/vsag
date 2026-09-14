// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "index_common_param_fwd.h"
#include "io/backend/external_storage_backend.h"
#include "io/cache/optional_page_cache.h"
#include "io/core/byte_io.h"
#include "io/external_storage_io/external_storage_io_parameter.h"

namespace vsag {

class ExternalStorageIO : public ByteIO<ExternalStorageBackend, OptionalPageCache> {
public:
    using Base = ByteIO<ExternalStorageBackend, OptionalPageCache>;

    explicit ExternalStorageIO(Allocator* allocator);
    // Bind handles on a private parameter copy; the input remains reusable with other storage sets.
    ExternalStorageIO(const ExternalStorageIOParamPtr& param, const IndexCommonParam& common_param);
    ExternalStorageIO(const IOParamPtr& param, const IndexCommonParam& common_param);
};

}  // namespace vsag
