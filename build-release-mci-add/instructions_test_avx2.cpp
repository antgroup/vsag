#include <immintrin.h>
int main() { __m256 a, b, c; c = _mm256_fmadd_ps(a, b, c); return 0; }