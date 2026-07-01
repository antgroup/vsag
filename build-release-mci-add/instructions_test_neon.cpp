#include <arm_neon.h>
int main() { float32x4_t a, b; a = vdupq_n_f32(1.0f); b = vdupq_n_f32(2.0f); a = vaddq_f32(a, b); return 0; }