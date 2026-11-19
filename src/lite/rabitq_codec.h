// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

// RaBitQ codec: umbrella include for the lite RaBitQ module.
// See the individual headers for the focused concern:
//   rabitq_transform.h  — FHT, Kac walk, Model (Transform / InverseTransform)
//   rabitq_storage.h    — Encoded, EncodedMetadata, EncodedView, EncodedRecords
//   rabitq_train.h      — Model training (centroid + random flips)
//   rabitq_encode.h     — fast_encode, normalize, encode_normalized, encode
//   rabitq_decode.h     — read_plane_code, decode, scalar_filter_centered_ip
//
// Include just the header(s) you need when only a subset of the codec is used.

#include <numeric>

#include "lite/rabitq_constants.h"
#include "lite/rabitq_decode.h"
#include "lite/rabitq_encode.h"
#include "lite/rabitq_storage.h"
#include "lite/rabitq_train.h"
#include "lite/rabitq_transform.h"
