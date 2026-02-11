#include <arm_sve.h>
int main() {
    svbool_t pg = svptrue_b32();
    svfloat32_t a = svdup_f32(1.0f);
    svfloat32_t b = svdup_f32(2.0f);
    a = svadd_f32_x(pg, a, b);
    return 0;
}