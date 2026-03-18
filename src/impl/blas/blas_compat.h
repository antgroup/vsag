#pragma once

#if defined(VSAG_USE_MKL_HEADERS)
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
