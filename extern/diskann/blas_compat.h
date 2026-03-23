#pragma once

#if defined(VSAG_USE_MKL_HEADERS)
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
using vsag_blasint = MKL_INT;
#else
#include <cblas.h>
#include <lapacke.h>
using vsag_blasint = lapack_int;
#endif
