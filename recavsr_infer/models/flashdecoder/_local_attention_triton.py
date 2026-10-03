"""FlashDecoder spatial-local GQA, adapted from RefVSR local attention.

Only spatial visibility changes; R=1/4 preserves the original temporal groups.
Fused dQ/dK/dV without atomic accumulation.

Programs own 4x16 spatial query tiles (forward/dQ) or key tiles (dK/dV).
KV-plane window sizes are compile-time constants. Both forward and reverse
neighborhoods preserve exact half-open, truncated semantics. GQA heads are
reduced inside the key-owned backward program, so every gradient is written
once. Inputs already contain global RoPE; gradients flow through it normally.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _forward(
    Q,
    K,
    V,
    OUT,
    LSE,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    WHS: tl.constexpr,
    WWS: tl.constexpr,
    COLS: tl.constexpr,
    QF: tl.constexpr,
    NF: tl.constexpr,
    Q_OFFSET: tl.constexpr,
    TW: tl.constexpr,
    R: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    n: tl.constexpr = HEIGHT * WIDTH
    nf: tl.constexpr = NF
    tile_y, tile_x, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qt = tile_y // triton.cdiv(HEIGHT, BH)
    tile_y = tile_y % triton.cdiv(HEIGHT, BH)
    batch, head = bh // HQ, bh % HQ
    kv_head = head // (HQ // HK)
    row = tl.arange(0, BM)
    qy, qx = tile_y * BH + row // BW, tile_x * BW + row % BW
    qi = qt * n + qy * WIDTH + qx
    valid_q = (qy < HEIGHT) & (qx < WIDTH)
    d = tl.arange(0, D)
    q = tl.load(
        Q + ((batch * HQ + head) * QF * n + qi[:, None]) * D + d[None, :],
        valid_q[:, None],
        0,
    )
    acc = tl.full((BM, D), 0, tl.float32)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.full((BM,), 0, tl.float32)
    scale: tl.constexpr = D**-0.5 * 1.4426950408889634
    for source in tl.static_range(len(WHS)):
        # Preserve FlashDecoder's original block-causal groups, including
        # bidirectional visibility inside each four-frame refinement group.
        plane = ((qt + Q_OFFSET) // R - TW + 1) * R + source
        valid_plane = (plane >= 0) & (plane < NF)
        if valid_plane:
            wh = WHS[source]
            ww = WWS[source]
            cols = COLS
            rows = min(wh + BH - 1, HEIGHT)
            top, left = (
                tl.maximum(tile_y * BH - wh // 2, 0),
                tl.maximum(tile_x * BW - ww // 2, 0),
            )
            for offset in range(tl.cdiv(rows * cols, BN)):
                p = offset * BN + tl.arange(0, BN)
                ky, kx = top + p // cols, left + p % cols
                valid_k = (
                    (p < rows * cols)
                    & (ky >= 0)
                    & (ky < HEIGHT)
                    & (kx >= 0)
                    & (kx < WIDTH)
                )
                valid_k = valid_k & valid_plane
                ki = plane * n + ky * WIDTH + kx
                k = tl.load(
                    K
                    + ((batch * HK + kv_head) * nf * n + ki[None, :]) * D
                    + d[:, None],
                    valid_k[None, :],
                    0,
                )
                scores = tl.dot(q, k).to(tl.float32) * scale
                allowed = (
                    valid_k[None, :]
                    & (ky[None, :] >= qy[:, None] - wh // 2)
                    & (ky[None, :] < qy[:, None] + wh // 2)
                    & (kx[None, :] >= qx[:, None] - ww // 2)
                    & (kx[None, :] < qx[:, None] + ww // 2)
                )
                scores = tl.where(allowed, scores, -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, 1))
                safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
                probabilities = tl.exp2(scores - safe_max[:, None])
                correction = tl.exp2(maximum - safe_max)
                v = tl.load(
                    V
                    + ((batch * HK + kv_head) * nf * n + ki[:, None]) * D
                    + d[None, :],
                    valid_k[:, None],
                    0,
                )
                acc = acc * correction[:, None] + tl.dot(probabilities.to(v.dtype), v)
                denominator = denominator * correction + tl.sum(probabilities, 1)
                maximum = new_max
    result = acc / denominator[:, None]
    tl.store(
        OUT + ((batch * HQ + head) * QF * n + qi[:, None]) * D + d[None, :],
        result,
        valid_q[:, None],
    )
    tl.store(
        LSE + (batch * HQ + head) * QF * n + qi, maximum + tl.log2(denominator), valid_q
    )


@triton.jit
def _backward_q(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    WHS: tl.constexpr,
    WWS: tl.constexpr,
    COLS: tl.constexpr,
    QF: tl.constexpr,
    NF: tl.constexpr,
    Q_OFFSET: tl.constexpr,
    TW: tl.constexpr,
    R: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    n: tl.constexpr = HEIGHT * WIDTH
    nf: tl.constexpr = NF
    tile_y, tile_x, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qt = tile_y // triton.cdiv(HEIGHT, BH)
    tile_y = tile_y % triton.cdiv(HEIGHT, BH)
    batch, head = bh // HQ, bh % HQ
    kv_head = head // (HQ // HK)
    row, d = tl.arange(0, BM), tl.arange(0, D)
    qy, qx = tile_y * BH + row // BW, tile_x * BW + row % BW
    qi = qt * n + qy * WIDTH + qx
    valid_q = (qy < HEIGHT) & (qx < WIDTH)
    qoff = ((batch * HQ + head) * QF * n + qi[:, None]) * D + d[None, :]
    q = tl.load(Q + qoff, valid_q[:, None], 0)
    do = tl.load(DO + qoff, valid_q[:, None], 0)
    lse = tl.load(LSE + (batch * HQ + head) * QF * n + qi, valid_q, 0)
    delta = tl.load(DELTA + (batch * HQ + head) * QF * n + qi, valid_q, 0)
    dq = tl.full((BM, D), 0, tl.float32)
    scale: tl.constexpr = D**-0.5
    for source in tl.static_range(len(WHS)):
        # Preserve FlashDecoder's original block-causal groups, including
        # bidirectional visibility inside each four-frame refinement group.
        plane = ((qt + Q_OFFSET) // R - TW + 1) * R + source
        valid_plane = (plane >= 0) & (plane < NF)
        if valid_plane:
            wh = WHS[source]
            ww = WWS[source]
            cols = COLS
            rows = min(wh + BH - 1, HEIGHT)
            top, left = (
                tl.maximum(tile_y * BH - wh // 2, 0),
                tl.maximum(tile_x * BW - ww // 2, 0),
            )
            for offset in range(tl.cdiv(rows * cols, BN)):
                p = offset * BN + tl.arange(0, BN)
                ky, kx = top + p // cols, left + p % cols
                valid_k = (
                    (p < rows * cols)
                    & (ky >= 0)
                    & (ky < HEIGHT)
                    & (kx >= 0)
                    & (kx < WIDTH)
                )
                valid_k = valid_k & valid_plane
                ki = plane * n + ky * WIDTH + kx
                koff = ((batch * HK + kv_head) * nf * n + ki[None, :]) * D + d[:, None]
                k = tl.load(K + koff, valid_k[None, :], 0)
                v = tl.load(V + koff, valid_k[None, :], 0)
                allowed = (
                    valid_q[:, None]
                    & valid_k[None, :]
                    & (ky[None, :] >= qy[:, None] - wh // 2)
                    & (ky[None, :] < qy[:, None] + wh // 2)
                    & (kx[None, :] >= qx[:, None] - ww // 2)
                    & (kx[None, :] < qx[:, None] + ww // 2)
                )
                scores = tl.dot(q, k).to(tl.float32) * (scale * 1.4426950408889634)
                prob = tl.where(allowed, tl.exp2(scores - lse[:, None]), 0.0)
                dp = tl.dot(do, v).to(tl.float32)
                ds = (prob * (dp - delta[:, None]) * scale).to(k.dtype)
                dq += tl.dot(ds, tl.trans(k))
    tl.store(DQ + qoff, dq, valid_q[:, None])


@triton.jit
def _backward_kv(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    WH: tl.constexpr,
    WW: tl.constexpr,
    NF: tl.constexpr,
    BATCH: tl.constexpr,
    PLANE_START: tl.constexpr,
    QF: tl.constexpr,
    Q_OFFSET: tl.constexpr,
    TW: tl.constexpr,
    R: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    n: tl.constexpr = HEIGHT * WIDTH
    tile_y, tile_x, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    plane = PLANE_START + bh // (BATCH * HK)
    batch, kv_head = (bh // HK) % BATCH, bh % HK
    row, d = tl.arange(0, BM), tl.arange(0, D)
    ky, kx = tile_y * BH + row // BW, tile_x * BW + row % BW
    ki = plane * n + ky * WIDTH + kx
    valid_k = (ky < HEIGHT) & (kx < WIDTH)
    koff = ((batch * HK + kv_head) * NF * n + ki[:, None]) * D + d[None, :]
    k = tl.load(K + koff, valid_k[:, None], 0)
    v = tl.load(V + koff, valid_k[:, None], 0)
    dk = tl.full((BM, D), 0, tl.float32)
    dv = tl.full((BM, D), 0, tl.float32)
    scale: tl.constexpr = D**-0.5
    cols: tl.constexpr = min(
        triton.next_power_of_2(WW + BW - 1), triton.next_power_of_2(WIDTH)
    )
    rows: tl.constexpr = min(WH + BH - 1, HEIGHT)
    # Invert k in [q-r,q+r): q in [k-r+1,k+r+1).
    top, left = (
        tl.maximum(tile_y * BH - WH // 2 + 1, 0),
        tl.maximum(tile_x * BW - WW // 2 + 1, 0),
    )
    # Reverse the same block visibility for key-owned gradients.
    first_qt = tl.maximum((plane // R) * R - Q_OFFSET, 0)
    last_qt = tl.minimum((plane // R + TW) * R - Q_OFFSET, QF)
    for qt in range(first_qt, last_qt):
        for group in range(HQ // HK):
            head = kv_head * (HQ // HK) + group
            for offset in range(tl.cdiv(rows * cols, BN)):
                p = offset * BN + tl.arange(0, BN)
                qy, qx = top + p // cols, left + p % cols
                valid_q = (
                    (p < rows * cols)
                    & (qy >= 0)
                    & (qy < HEIGHT)
                    & (qx >= 0)
                    & (qx < WIDTH)
                )
                qi = qt * n + qy * WIDTH + qx
                qoff = ((batch * HQ + head) * QF * n + qi[None, :]) * D + d[:, None]
                q = tl.load(Q + qoff, valid_q[None, :], 0)
                do = tl.load(DO + qoff, valid_q[None, :], 0)
                lse = tl.load(LSE + (batch * HQ + head) * QF * n + qi, valid_q, 0)
                delta = tl.load(DELTA + (batch * HQ + head) * QF * n + qi, valid_q, 0)
                allowed = (
                    valid_k[:, None]
                    & valid_q[None, :]
                    & (ky[:, None] >= qy[None, :] - WH // 2)
                    & (ky[:, None] < qy[None, :] + WH // 2)
                    & (kx[:, None] >= qx[None, :] - WW // 2)
                    & (kx[:, None] < qx[None, :] + WW // 2)
                )
                scores = tl.dot(k, q).to(tl.float32) * (scale * 1.4426950408889634)
                prob = tl.where(allowed, tl.exp2(scores - lse[None, :]), 0.0)
                dp = tl.dot(v, do).to(tl.float32)
                ds = (prob * (dp - delta[None, :]) * scale).to(q.dtype)
                dk += tl.dot(ds, tl.trans(q))
                dv += tl.dot(prob.to(do.dtype), tl.trans(do))
    tl.store(DK + koff, dk, valid_k[:, None])
    tl.store(DV + koff, dv, valid_k[:, None])


class _LocalAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query, key, value, height, width, window, group_size, temporal_window
    ):
        q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
        b, hq, tokens, d = q.shape
        hk = k.shape[1]
        qf, nf = tokens // (height * width), k.shape[2] // (height * width)
        q_offset = nf - qf
        wh, ww = window
        whs, wws = (
            (wh,) * (temporal_window * group_size),
            (ww,) * (temporal_window * group_size),
        )
        output = torch.empty_like(q)
        lse = torch.empty((b, hq, tokens), device=q.device, dtype=torch.float32)
        grid = (qf * triton.cdiv(height, 4), triton.cdiv(width, 16), b * hq)
        with torch.cuda.device(q.device):
            _forward[grid](
                q,
                k,
                v,
                output,
                lse,
                height,
                width,
                hq,
                hk,
                d,
                whs,
                wws,
                min(triton.next_power_of_2(ww + 15), triton.next_power_of_2(width)),
                qf,
                nf,
                q_offset,
                temporal_window,
                group_size,
                4,
                16,
                64,
                64,
                num_warps=4,
                num_stages=2,
            )
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.spec = (height, width, window, group_size, temporal_window)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors
        height, width, window, group_size, temporal_window = ctx.spec
        b, hq, tokens, d = q.shape
        hk = k.shape[1]
        qf, nf = tokens // (height * width), k.shape[2] // (height * width)
        q_offset = nf - qf
        wh, ww = window
        whs, wws = (
            (wh,) * (temporal_window * group_size),
            (ww,) * (temporal_window * group_size),
        )
        do = grad_output.contiguous()
        delta = (output.float() * do.float()).sum(-1).contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        with torch.cuda.device(q.device):
            grid_q = (qf * triton.cdiv(height, 4), triton.cdiv(width, 16), b * hq)
            _backward_q[grid_q](
                q,
                k,
                v,
                do,
                lse,
                delta,
                dq,
                height,
                width,
                hq,
                hk,
                d,
                whs,
                wws,
                min(triton.next_power_of_2(ww + 15), triton.next_power_of_2(width)),
                qf,
                nf,
                q_offset,
                temporal_window,
                group_size,
                4,
                16,
                64,
                64,
                num_warps=4,
                num_stages=2,
            )
            grid_k = (triton.cdiv(height, 4), triton.cdiv(width, 16), b * hk * nf)
            _backward_kv[grid_k](
                q,
                k,
                v,
                do,
                lse,
                delta,
                dk,
                dv,
                height,
                width,
                hq,
                hk,
                d,
                wh,
                ww,
                nf,
                b,
                0,
                qf,
                q_offset,
                temporal_window,
                group_size,
                4,
                16,
                64,
                64,
                num_warps=4,
                num_stages=2,
            )
        return dq, dk, dv, None, None, None, None, None


def triton_local_attention(
    query, key, value, *, height, width, window, group_size, temporal_window
):
    """First-order CUDA BF16/FP16 GQA; validated by the model's dispatch helper."""
    return _LocalAttention.apply(
        query, key, value, height, width, window, group_size, temporal_window
    )
