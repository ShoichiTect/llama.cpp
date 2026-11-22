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

// the short convolution: out[ch,t] = (cur[ch,t] + sum_lag x[t+lag,ch]*w[ch,lag]
//   + sum_lag x[t+(n_cols-lag),ch]*dyn[lag,head(ch),t]) * scale
// where the padded row k is hist[k] for k < n_cols and cur[k - n_cols] after that
kernel void kernel_inkling2_conv(
        constant ggml_metal_kargs_inkling2_conv & args,
        device const float * cur,
        device const float * w,
        device const float * hist,
        device const float * dyn,
        device       float * out,
        uint gid [[thread_position_in_grid]]) {
    const int64_t nc = (int64_t) args.n_channels*args.n_tokens;
    if ((int64_t) gid >= nc) {
        return;
    }

    const int ch = (int) (gid % args.n_channels);
    const int t  = (int) (gid / args.n_channels);

    const int s    = t/args.n_seq_tokens;
    const int tpos = t%args.n_seq_tokens;

    device const float * hc = (device const float *) ((device const char *) hist
            + (int64_t) (args.hist_off + ch)*args.nb1_h + (int64_t) s*args.nb2_h);

    float acc = 0.0f;
    for (int lag = 0; lag < args.n_conv; ++lag) {
        const int k = tpos + lag;
        const float v = k < args.n_cols
            ? hc[k]
            : cur[ch + (int64_t) args.n_channels*(s*args.n_seq_tokens + (k - args.n_cols))];
        acc += v*w[(int64_t) ch*args.n_conv + lag];
    }

    if (args.has_dyn) {
        const int head = ch/args.hsz;
        for (int lag = 0; lag < args.n_conv; ++lag) {
            const int k = tpos + (args.n_cols - lag);
            const float v = k < args.n_cols
                ? hc[k]
                : cur[ch + (int64_t) args.n_channels*(s*args.n_seq_tokens + (k - args.n_cols))];
            acc += v*dyn[(int64_t) lag + (int64_t) args.n_conv*(head + (int64_t) args.n_heads*t)];
        }
    }

    out[ch + (int64_t) args.n_channels*t] = (cur[ch + (int64_t) args.n_channels*t] + acc)*args.scale;
}

// the recurrent state store: the last n_cols rows of the padded buffer, as [lag, channel]
// one thread per (channel, seq) so that the loads happen before the stores of the same channel
kernel void kernel_inkling2_conv_state(
        constant ggml_metal_kargs_inkling2_conv_state & args,
        device const float * cur,
        device const float * hist,
        device       float * cell,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = (int64_t) args.n_channels*args.n_seqs;
    if ((int64_t) gid >= n) {
        return;
    }

    const int ch = (int) (gid % args.n_channels);
    const int s  = (int) (gid / args.n_channels);

    device const float * hc = (device const float *) ((device const char *) hist
            + (int64_t) (args.hist_off + ch)*args.nb1_h + (int64_t) s*args.nb2_h);
    device       float * cp = (device       float *) ((device       char *) cell + (int64_t) s*args.nb1_cell) + (int64_t) ch*args.n_cols;

    float v[8];
    for (int l = 0; l < args.n_cols; ++l) {
        const int k = args.n_seq_tokens + l;
        v[l] = k < args.n_cols
            ? hc[k]
            : cur[ch + (int64_t) args.n_channels*(s*args.n_seq_tokens + (k - args.n_cols))];
    }
    for (int l = 0; l < args.n_cols; ++l) {
        cp[l] = v[l];
    }
}

// the controller tail: out = (cur - tanh(forget)*cur) + tanh(gain)*message
// the tanhs are the same MSL tanh that the ggml unary kernel uses, and the fp
// contraction is off so that the result stays bit-identical to the 3-node chain
kernel void kernel_inkling2_ctrl(
        constant ggml_metal_kargs_inkling2_ctrl & args,
        device const float * cur,
        device const float * msg,
        device const float * gain,
        device const float * forget,
        device       float * out,
        uint gid [[thread_position_in_grid]]) {
    const int64_t ch = (int64_t) gid % args.ne0;
    const int64_t t  = (int64_t) gid / args.ne0;
    if (t >= args.ne1) {
        return;
    }

    const device float * pc = (device const float *) ((device const char *) cur    + ch*4 + t*args.nb1_cur);
    const device float * pm = (device const float *) ((device const char *) msg    + ch*4 + t*args.nb1_msg);
    const device float * pa = (device const float *) ((device const char *) gain   + ch*4 + t*args.nb1_gain);
    const device float * pf = (device const float *) ((device const char *) forget + ch*4 + t*args.nb1_forget);

    const float      c = pc[0];
    volatile float   p = tanh(pf[0])*c;   // the volatile stops the fma contraction
    const float      q = tanh(pa[0])*pm[0];

    out[gid] = (c - p) + q;
}

// rms_norm(cur) || rms_norm(message): the reduction is the same one as kernel_rms_norm_f32
// (float4 chunks, per-simdgroup simd_sum, then across the simdgroups), so the result is bit-identical
kernel void kernel_inkling2_gate(
        constant ggml_metal_kargs_inkling2_gate & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int64_t row = tgpig.x;

    device const float4 * xc = (device const float4 *) (src0 + row*args.nb1_c);
    device const float4 * xm = (device const float4 *) (src1 + row*args.nb1_m);
    device       float4 * y  = (device       float4 *) (dst  + row*args.nb1_d);

    for (int part = 0; part < 2; ++part) {
        device const float4 * x = part ? xm : xc;
        device       float4 * o = part ? y + args.ne0_t : y;

        // the previous part still reads shmem_f32 (sumf = shmem_f32[tiisg]) below, so wait for it
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            shmem_f32[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sumf = 0.0f;
        for (int i = tpitg.x; i < args.ne0_t; i += ntg.x) {
            sumf += dot(x[i], x[i]);
        }
        sumf = simd_sum(sumf);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shmem_f32[sgitg] = sumf;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sumf = shmem_f32[tiisg];
        sumf = simd_sum(sumf);

        const float scale = 1.0f/sqrt(sumf/args.ne0 + args.eps);

        for (int i = tpitg.x; i < args.ne0_t; i += ntg.x) {
            o[i] = x[i]*scale;
        }
    }
}

// situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
// the same expression order as the 9-node chain, so the result is bit-identical
kernel void kernel_inkling2_situ(
        constant ggml_metal_kargs_inkling2_situ & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t i0  = (int64_t) gid % args.ne0;
    const int64_t row = (int64_t) gid / args.ne0;
    if (row >= args.nrows) {
        return;
    }

    const device float * g = (device const float *) (src0 + row*args.nb1_g + i0*4);
    const device float * u = (device const float *) (src1 + row*args.nb1_u + i0*4);
    device       float * o = (device       float *) (dst  + (row*args.ne0 + i0)*4);

    const float gv = g[0];

    float act = args.beta*tanh(gv*(1.0f/args.beta)) * (1.0f/(1.0f + exp(-gv)));

    if (args.lb > 0.0f) {
        act *= args.lb*tanh(u[0]*(1.0f/args.lb));
    } else {
        act *= u[0];
    }

    o[0] = act;
}
