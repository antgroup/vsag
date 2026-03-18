#pragma once

#if defined(VSAG_USE_MKL_HEADERS)
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#ifndef blasint
#define blasint MKL_INT
#endif
#else
#include <cblas.h>
#include <lapacke.h>
#ifndef blasint
#define blasint lapack_int
#endif
#endif
