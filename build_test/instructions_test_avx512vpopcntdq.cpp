#include <immintrin.h>
int main() { __m512i a, b; b = _mm512_popcnt_epi64(a); return 0; }