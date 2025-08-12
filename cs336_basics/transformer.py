import math
import torch
import torch.nn as nn
from einops import einsum, reduce, rearrange


def softmax(x: torch.Tensor, dimension: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dimension, keepdim=True)[0]
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dimension, keepdim=True)

def scaled_dot_product_attention(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
) -> torch.Tensor:
    attention_weights = einsum(queries, keys, "batch_size ... seq_len_n d_k, batch_size ... seq_len_m d_k -> batch_size ... seq_len_n seq_len_m")
    key_dim = queries.shape[-1]
    attention_weights = attention_weights / math.sqrt(key_dim)
    if mask is not None:
        # set attention weights at masked indexes to -inf so that soft max sets it to 0
        attention_weights = torch.where(mask, attention_weights, float("-inf"))
    attention_weights_normalized = softmax(attention_weights, -1)
    return einsum(attention_weights_normalized, values, "batch_size ... seq_len_n seq_len_m, batch_size ... seq_len_m d_v -> batch_size ... seq_len_n d_v")

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        weight_tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std_deviation = 2.0 / (in_features + out_features)
        nn.init.trunc_normal_(weight_tensor, mean=0, std=std_deviation, a=-3*math.sqrt(std_deviation), b=3*std_deviation)
        self.weight = nn.Parameter(weight_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None ):
        super().__init__()
        embed_tensor = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(embed_tensor, mean=0, std=1.0, a=-3, b=3)
        self.embeddings = nn.Parameter(embed_tensor)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = reduce(x * x, 'batch seq d_model -> batch seq 1', 'mean')
        rms_norm = torch.sqrt(mean_square + self.eps)
        result = x / rms_norm * self.gain
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff > 0 else (8.0 * d_model) / 3
        # ensure dimension of ff layer is a multiple of 64
        if self.d_ff % 64 != 0:
            self.d_ff += 64 - (self.d_ff % 64)

        # W2 @ (SiLU(W1 @ x) . (W3 @ x))
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_forward = self.linear1.forward(x)
        # W2 @ (SiLU(W1 @ x) . (W3 @ x))
        return self.linear2.forward(w1_forward * torch.sigmoid(w1_forward) * self.linear3.forward(x))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention_weights = Linear(d_model, 3 * self.num_heads * self.d_k, device=device, dtype=dtype)
        self.output_weights = Linear(self.num_heads * self.d_k, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project inputs to concatenated QKV for all heads at once
        projections = self.attention_weights.forward(x)
        # Shape transform to split into Q, K, V and heads
        # projections: (batch, seq, 3 * num_heads * d_k)
        # kqv_heads: (3, batch, num_heads, seq, d_k)
        kqv_heads = rearrange(
            projections,
            "batch seq (kqv h d_k) -> kqv batch h seq d_k",
            kqv=3,
            h=self.num_heads,
            d_k=self.d_k,
        )
        Q, K, V = kqv_heads[0], kqv_heads[1], kqv_heads[2]

        # Create a causal mask (allow attending to self and previous positions only)
        seq_len = x.shape[-2]
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # Batched scaled dot-product attention across all heads
        attended = scaled_dot_product_attention(Q, K, V, mask=causal)

        # Concatenate heads and project out
        concat_heads = rearrange(attended, "batch h seq d_k -> batch seq (h d_k)")
        return self.output_weights.forward(concat_heads)

