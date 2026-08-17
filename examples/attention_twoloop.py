"""Flat-batch 2D-tile attention kernels (dense + two-loop causal).

Both kernels index the flattened batch*heads dimension with a scalar
(``tile_b.begin`` with block_size=1) so every load/store is a clean 2D tile:
this keeps tensor-descriptor (TMA) indexing eligible on the TileIR backend
and avoids 3D reshape round-trips around ``hl.dot``.

The causal kernel splits the KV loop Ocean-style: phase 1 covers KV tiles
fully below the diagonal (no masking; with block_m % block_n == 0 the
compiler proves the loop bounds block-aligned and drops boundary masks),
phase 2 covers the diagonal tiles with a finite -1.0e6 fill. Softmax runs in
fp32 end-to-end with the qk scale folded into the running-max update.
"""

from __future__ import annotations

import math

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


@helion.kernel(static_shapes=True)
def dense_attention_2d(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """Dense (non-causal) attention forward with flat-batch 2D tiles."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = 1.0 / math.sqrt(head_dim) * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim], block_size=[1, None]):
        b = tile_b.begin
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q = q_view[b, tile_m, :]
        for tile_n in hl.tile(0, n_dim):
            k = k_view[b, tile_n, :]
            qk = hl.dot(q, k.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            p = torch.exp2(qk * qk_scale - m_ij[:, None])
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v = v_view[b, tile_n, :]
            acc = hl.dot(p.to(v.dtype), v, acc=acc)
            m_i = m_ij
        out[b, tile_m, :] = (acc / l_i[:, None]).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(static_shapes=True)
def causal_attention_twoloop(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """Causal attention forward: unmasked bulk loop + masked diagonal loop."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = 1.0 / math.sqrt(head_dim) * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim], block_size=[1, None]):
        b = tile_b.begin
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q = q_view[b, tile_m, :]
        # Phase 1: KV tiles fully below the diagonal -- no masking needed.
        for tile_n in hl.tile(0, tile_m.begin):
            k = k_view[b, tile_n, :]
            qk = hl.dot(q, k.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            p = torch.exp2(qk * qk_scale - m_ij[:, None])
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v = v_view[b, tile_n, :]
            acc = hl.dot(p.to(v.dtype), v, acc=acc)
            m_i = m_ij
        # Phase 2: diagonal KV tiles -- masked with a finite fill value.
        for tile_n in hl.tile(tile_m.begin, tile_m.end):
            k = k_view[b, tile_n, :]
            qk = hl.dot(q, k.T, out_dtype=torch.float32)
            qk = torch.where(
                tile_m.index[:, None] >= tile_n.index[None, :],
                qk,
                -1.0e6,
            )
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            p = torch.exp2(qk * qk_scale - m_ij[:, None])
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v = v_view[b, tile_n, :]
            acc = hl.dot(p.to(v.dtype), v, acc=acc)
            m_i = m_ij
        out[b, tile_m, :] = (acc / l_i[:, None]).to(out.dtype)
    return out.view(q_in.size())


def check(z: int, h: int, s: int, d: int) -> None:
    q = torch.randn(z, h, s, d, device=DEVICE, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    run_example(
        dense_attention_2d,
        lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v),
        (q, k, v),
        atol=5e-2,
        rtol=2e-2,
    )
    run_example(
        causal_attention_twoloop,
        lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        ),
        (q, k, v),
        atol=5e-2,
        rtol=2e-2,
    )


def main() -> None:
    check(2, 8, 4096, 64)


if __name__ == "__main__":
    main()
