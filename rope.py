from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    freqs_cis: (seqlen, head_dim // 2)
    x:         (..., seqlen, ..., head_dim // 2)  —— 实际用在复数表示的 q/k 上
    """
    ndim = x.ndim
    # 我们假设时间维在 dim=1，最后一维是要旋转的维度
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # 生成形状形如 (1, seqlen, 1, ..., head_dim//2)
    shape = [1] * ndim
    shape[1] = x.shape[1]
    shape[-1] = x.shape[-1]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    query: (batch_size, seqlen, n_local_heads, head_dim)
    key:   (batch_size, seqlen, n_local_kv_heads, head_dim)
    """
    bsz, seqlen, _, _ = query.shape
    device = query.device
    dtype = query.dtype

    assert head_dim % 2 == 0, "RoPE 需要 head_dim 为偶数"
    half_dim = head_dim // 2

    # 1) 构造每个维度的频率：inv_freq[i] = theta^{-2i / head_dim}
    dim_idx = torch.arange(half_dim, device=device, dtype=torch.float32)  # (half_dim,)
    inv_freq = 1.0 / (theta ** (2 * dim_idx / head_dim))                  # (half_dim,)

    # 2) 位置：0,1,...,max_seq_len-1，然后只取前 seqlen 个
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # (max_seq_len,)
    angles = torch.einsum("p,d->pd", positions, inv_freq)                      # (max_seq_len, half_dim)

    # 只用实际序列长度部分
    angles = angles[:seqlen]                                                  # (seqlen, half_dim)

    # 3) 用极坐标形式生成复数：cos(θ) + i·sin(θ)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)                  # (seqlen, half_dim), complex64

    # 4) 把 q/k 视作复数：最后一维拆成 (half_dim, 2)
    #    view_as_complex 要求最后一维 size=2，因此 reshape(..., half_dim, 2)
    q_float = query.float().reshape(*query.shape[:-1], -1, 2)                 # (..., half_dim, 2)
    k_float = key.float().reshape(*key.shape[:-1], -1, 2)

    q_complex = torch.view_as_complex(q_float)                                # (b, seqlen, n_heads, half_dim), complex
    k_complex = torch.view_as_complex(k_float)

    # 5) 利用 reshape_for_broadcast 把 freqs_cis broadcast 到 q/k 的形状
    freqs_cis_broadcast = reshape_for_broadcast(freqs_cis, q_complex)         # (1, seqlen, 1, half_dim)

    # 6) 复数乘法实现旋转
    q_rot = q_complex * freqs_cis_broadcast
    k_rot = k_complex * freqs_cis_broadcast

    # 7) 把复数再还原回实数张量，形状恢复成原始的 (b, seqlen, n_heads, head_dim)
    q_out = torch.view_as_real(q_rot).reshape(bsz, seqlen, -1, head_dim)      # (..., 2) -> head_dim
    k_out = torch.view_as_real(k_rot).reshape(bsz, seqlen, -1, head_dim)

    # 8) 转回原来的 dtype（可能是 float16/bfloat16）
    q_out = q_out.to(dtype=dtype)
    k_out = k_out.to(dtype=dtype)

    return q_out, k_out