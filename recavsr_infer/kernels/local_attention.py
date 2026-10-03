"""ReCaVSR spatial attention: paired query latents, sparse physical KV slots.

The kernel is newly implemented for 6-latent bidirectional prefixes and
2-latent blocks. Temporal visibility is determined solely by the caller's
router. No causal-time mask, ring-position RoPE, or replicated image tiles.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from recavsr_infer.runtime.rope import apply_temporal


@triton.jit
def _paired_local(
    Q,
    CK,
    CV,
    PK,
    PV,
    SLOTS,
    PPOS,
    QPOS,
    COS,
    SIN,
    OUT,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    cs2: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    ps0: tl.constexpr,
    ps1: tl.constexpr,
    ps2: tl.constexpr,
    pv0: tl.constexpr,
    pv1: tl.constexpr,
    pv2: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    QT: tl.constexpr,
    CAP: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    TD: tl.constexpr,
    WH: tl.constexpr,
    WW: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    spatial_tile, temporal_tile, bh = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = bh // HEADS, bh % HEADS
    y0 = spatial_tile // triton.cdiv(WIDTH, BW) * BH
    x0 = spatial_tile % triton.cdiv(WIDTH, BW) * BW
    r = tl.arange(0, BM)
    qt = temporal_tile * BT + r // (BH * BW)
    qy = y0 + r // BW % BH
    qx = x0 + r % BW
    good_q = (qt < QT) & (qy < HEIGHT) & (qx < WIDTH)
    qi = qt * (HEIGHT * WIDTH) + qy * WIDTH + qx
    d = tl.arange(0, DIM)
    q = tl.load(
        Q + batch * qs0 + qi[:, None] * qs1 + head * qs2 + d[None, :],
        good_q[:, None],
        0,
    ).to(tl.float32)
    # Swap rotary partners in registers instead of loading Q a second time.
    qp = tl.gather(q, tl.broadcast_to((d ^ 1)[None, :], (BM, DIM)), axis=1)
    pos = tl.load(QPOS + qt, good_q, 0)
    tc = tl.load(
        COS + pos[:, None] * (TD // 2) + d[None, :] // 2,
        good_q[:, None] & (d[None, :] < TD),
        1,
    )
    ts = tl.load(
        SIN + pos[:, None] * (TD // 2) + d[None, :] // 2,
        good_q[:, None] & (d[None, :] < TD),
        0,
    )
    sign = tl.where(d % 2 == 0, -1.0, 1.0)
    q = (q * tc + qp * ts * sign[None, :]).to(Q.dtype.element_ty)
    top = tl.minimum(tl.maximum(qy - WH // 2, 0), HEIGHT - WH)
    left = tl.minimum(tl.maximum(qx - WW // 2, 0), WIDTH - WW)
    base_y = tl.minimum(tl.maximum(y0 - WH // 2, 0), HEIGHT - WH)
    base_x = tl.minimum(tl.maximum(x0 - WW // 2, 0), WIDTH - WW)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.full((BM,), 0, tl.float32)
    acc = tl.full((BM, DIM), 0, tl.float32)
    for plane in range(CAP + QT):
        if plane < CAP:
            slot = tl.load(SLOTS + plane)
            tpos = tl.load(PPOS + plane)
            kp, vp = PK, PV
            k0, k1, k2 = ps0, ps1, ps2
            v0, v1, v2 = pv0, pv1, pv2
        else:
            slot = plane - CAP
            tpos = tl.load(QPOS + slot)
            kp, vp = CK, CV
            k0, k1, k2 = cs0, cs1, cs2
            v0, v1, v2 = vs0, vs1, vs2
        # An empty physical slot has no visible keys; skip its entire plane.
        # Current planes remain fully bidirectional for both QT=2 and QT=6.
        if slot >= 0:
            for chunk in range(triton.cdiv(ROWS * COLS, BN)):
                p = chunk * BN + tl.arange(0, BN)
                ky, kx = base_y + p // COLS, base_x + p % COLS
                good_k = (p < ROWS * COLS) & (ky < HEIGHT) & (kx < WIDTH) & (slot >= 0)
                ki = tl.maximum(slot, 0) * (HEIGHT * WIDTH) + ky * WIDTH + kx
                k = tl.load(
                    kp + batch * k0 + ki[None, :] * k1 + head * k2 + d[:, None],
                    good_k[None, :],
                    0,
                ).to(tl.float32)
                # Preserve FP32 rotary arithmetic and the original BF16/FP16 cast.
                pair = tl.gather(
                    k, tl.broadcast_to((d ^ 1)[:, None], (DIM, BN)), axis=0
                )
                kc = tl.load(COS + tpos * (TD // 2) + d // 2, d < TD, 1)
                ks = tl.load(SIN + tpos * (TD // 2) + d // 2, d < TD, 0)
                k = (k * kc[:, None] + pair * ks[:, None] * sign[:, None]).to(
                    Q.dtype.element_ty
                )
                score = tl.dot(q, k).to(tl.float32) * (DIM**-0.5 * 1.4426950408889634)
                allowed = (
                    good_q[:, None]
                    & good_k[None, :]
                    & (ky[None, :] >= top[:, None])
                    & (ky[None, :] < top[:, None] + WH)
                    & (kx[None, :] >= left[:, None])
                    & (kx[None, :] < left[:, None] + WW)
                )
                score = tl.where(allowed, score, -float("inf"))
                next_max = tl.maximum(maximum, tl.max(score, 1))
                safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
                prob = tl.exp2(score - safe_max[:, None])
                correction = tl.exp2(maximum - safe_max)
                v = tl.load(
                    vp + batch * v0 + ki[:, None] * v1 + head * v2 + d[None, :],
                    good_k[:, None],
                    0,
                )
                acc = acc * correction[:, None] + tl.dot(prob.to(v.dtype), v)
                denominator = denominator * correction + tl.sum(prob, 1)
                maximum = next_max
    output = acc / tl.maximum(denominator[:, None], 1e-20)
    tl.store(
        OUT
        + ((batch * (QT * HEIGHT * WIDTH) + qi[:, None]) * HEADS + head) * DIM
        + d[None, :],
        output,
        good_q[:, None],
    )


@torch.library.custom_op("recavsr_local::paired", mutates_args=())
def paired_attention(
    q: torch.Tensor,
    ck: torch.Tensor,
    cv: torch.Tensor,
    pk: torch.Tensor,
    pv: torch.Tensor,
    slots: torch.Tensor,
    past_positions: torch.Tensor,
    query_positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    height: int,
    width: int,
    window_h: int,
    window_w: int,
    query_tile_h: int = 4,
    query_tile_w: int = 8,
    key_tile: int = 64,
) -> torch.Tensor:
    if q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("The paired kernel requires CUDA BF16/FP16.")
    if q.shape != ck.shape or q.shape != cv.shape or pk.shape != pv.shape:
        raise ValueError("Q/current KV and past KV shapes disagree.")
    span = height * width
    if height <= 0 or width <= 0 or q.shape[1] % span:
        raise ValueError("Invalid token geometry.")
    frames = q.shape[1] // span
    if frames not in (2, 6) or q.shape[-1] not in (32, 64, 128):
        raise ValueError(
            "The final kernel supports prefix=6 or block=2, head=32/64/128."
        )
    if slots.numel() * span != pk.shape[1] or past_positions.shape != slots.shape:
        raise ValueError("Cache slot metadata mismatch.")
    if query_positions.numel() != frames or cos.shape != sin.shape:
        raise ValueError("Rolling RoPE metadata mismatch.")
    if min(window_h, window_w) <= 0:
        raise ValueError("Spatial windows must be positive.")
    if any(x.stride(-1) != 1 for x in (q, ck, cv, pk, pv)):
        raise ValueError("Head dimensions must be contiguous.")
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    wh, ww = min(height, window_h), min(width, window_w)
    bh, bw, bt = query_tile_h, query_tile_w, 2
    if bh * bw * bt not in (32, 64, 128) or key_tile not in (32, 64, 128):
        raise ValueError("Unsupported kernel tile.")
    _paired_local[
        (
            triton.cdiv(height, bh) * triton.cdiv(width, bw),
            triton.cdiv(frames, bt),
            q.shape[0] * q.shape[2],
        )
    ](
        q,
        ck,
        cv,
        pk,
        pv,
        slots,
        past_positions,
        query_positions,
        cos,
        sin,
        out,
        *q.stride()[:3],
        *ck.stride()[:3],
        *cv.stride()[:3],
        *pk.stride()[:3],
        *pv.stride()[:3],
        height,
        width,
        frames,
        slots.numel(),
        q.shape[2],
        q.shape[3],
        cos.shape[1] * 2,
        wh,
        ww,
        bh,
        bw,
        bt,
        bh * bw * bt,
        key_tile,
        min(height, wh + bh - 1),
        min(width, ww + bw - 1),
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return out


@paired_attention.register_fake
def _fake(
    q,
    ck,
    cv,
    pk,
    pv,
    slots,
    past_positions,
    query_positions,
    cos,
    sin,
    height,
    width,
    window_h,
    window_w,
    query_tile_h=4,
    query_tile_w=8,
    key_tile=64,
):
    return torch.empty_like(q, memory_format=torch.contiguous_format)


def sdpa_attention(
    q,
    ck,
    cv,
    pk,
    pv,
    slots,
    past_positions,
    query_positions,
    cos,
    sin,
    height,
    width,
    window_h,
    window_w,
):
    """Full-spatial SDPA with the router's sparse temporal KV visibility."""
    if window_h < height or window_w < width:
        raise ValueError("SDPA requires a window covering the full spatial grid.")
    span = height * width
    if slots.numel():
        index = slots.clamp_min(0).long()
        past_k = pk.unflatten(1, (-1, span))[:, index].flatten(1, 2)
        past_v = pv.unflatten(1, (-1, span))[:, index].flatten(1, 2)
    else:
        past_k, past_v = pk, pv
    key = torch.cat((past_k, ck), 1)
    value = torch.cat((past_v, cv), 1)
    positions = torch.cat((past_positions, query_positions))
    q = apply_temporal(q, query_positions, cos, sin, span=span)
    key = apply_temporal(key, positions, cos, sin, span=span)
    valid = torch.cat((slots >= 0, torch.ones_like(query_positions, dtype=torch.bool)))
    mask = valid.repeat_interleave(span).view(1, 1, 1, -1)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=mask,
        dropout_p=0.0,
    ).transpose(1, 2)
