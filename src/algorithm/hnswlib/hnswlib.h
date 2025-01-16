
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

#include "hnswalg.h"
#include "hnswalg_static.h"
#include "space_ip.h"
#include "space_l2.h"
