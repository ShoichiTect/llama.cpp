#include "common.h"

// inkling2: clip(x) = x * min(threshold/(sqrt(sum(x^2)) + 1e-6), 1), the norm is over ne0 (the head dim)
// one thread per row: the rows are tiny (ne0 <= 256) and the launch count is what matters
kernel void kernel_inkling2_clip(
        constant ggml_metal_kargs_inkling2_clip & args,
        device const float * src0,
        device       float * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t r = (int64_t) gid;
    if (r >= args.nrows) {
        return;
    }

    device const float * x = src0 + r*args.ne0;
    device       float * y = dst  + r*args.ne0;

    float sum = 0.0f;
    for (int i = 0; i < args.ne0; ++i) {
        const float v = x[i];
        sum += v*v;
    }

    float f = args.threshold/(sqrt(sum) + 1e-6f);
    f = f > 1.0f ? 1.0f : f;
    f = f < 0.0f ? 0.0f : f;

    for (int i = 0; i < args.ne0; ++i) {
        y[i] = x[i]*f;
    }
}
