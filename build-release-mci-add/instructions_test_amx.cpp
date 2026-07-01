#include <immintrin.h>
int main() {
    _tile_loadconfig((const void*)0);
    _tile_zero(0);
    _tile_dpbuud(0, 1, 2);
    _tile_dpbf16ps(0, 1, 2);
    __m512bh a = (__m512bh)_mm512_setzero_si512();
    __m512bh b = (__m512bh)_mm512_setzero_si512();
    __m512  c = _mm512_setzero_ps();
    c = _mm512_dpbf16_ps(c, a, b);
    _tile_release();
    return 0;
}