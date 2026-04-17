// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#ifndef DISKANN_DLLEXPORT
#ifdef _WINDOWS

#ifdef _WINDLL
#define DISKANN_DLLEXPORT __declspec(dllexport)
#else
#define DISKANN_DLLEXPORT __declspec(dllimport)
#endif

#else
#define DISKANN_DLLEXPORT
#endif
#endif
