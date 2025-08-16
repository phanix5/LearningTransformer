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

class RotaryPositionalEmbedding(nn.Module):
    rotary_matrix: torch.Tensor
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        rotation_matrices = []
        for i in range(max_seq_len):
            rotation_values = torch.tensor([i / (math.pow(theta, (2 * (k // 2)) / float(d_k))) for k in range(d_k)], dtype=dtype)
            cos_values = torch.cos(rotation_values)
            mask = torch.arange(rotation_values.shape[0], dtype=dtype) % 2 == 0
            sin_values = torch.sin(torch.where(mask, rotation_values, 0))[:-1]
            rotation_matrices.append(torch.diag(cos_values) + torch.diag(sin_values, diagonal=-1) + torch.diag(-sin_values, diagonal=1))
        self.register_buffer('rotary_matrix', torch.stack(rotation_matrices, dim=0).to(device))
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        rotary_matrices = self.rotary_matrix[token_positions]
        return einsum(rotary_matrices, x, '... seq_len i j, ... seq_len j -> ... seq_len i')
        

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

class MultiHeadAttentionRope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention_weights = Linear(d_model, 3 * self.num_heads * self.d_k, device=device, dtype=dtype)
        self.output_weights = Linear(self.num_heads * self.d_k, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
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

        # Apply RoPE to Q & K vectors
        Q = self.rope.forward(Q, token_positions)
        K = self.rope.forward(K, token_positions)
        

        # Create a causal mask (allow attending to self and previous positions only)
        seq_len = x.shape[-2]
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # Batched scaled dot-product attention across all heads
        attended = scaled_dot_product_attention(Q, K, V, mask=causal)

        # Concatenate heads and project out
        concat_heads = rearrange(attended, "batch h seq d_k -> batch seq (h d_k)")
        return self.output_weights.forward(concat_heads)

class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 max_seq_len: int, 
                 theta: float,
                 weights: dict[str, torch.Tensor] | None = None, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.pre_mha_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.pre_ff_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadAttentionRope(d_model, num_heads, theta, max_seq_len, device = device, dtype = dtype)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

        if weights is not None:
            qkv_combined = torch.cat((weights['attn.q_proj.weight'], weights['attn.k_proj.weight'], weights['attn.v_proj.weight']), dim=0)
            self.mha.attention_weights.load_state_dict({'weight': qkv_combined})
            self.mha.output_weights.load_state_dict({'weight': weights['attn.output_proj.weight']})

            self.pre_mha_norm.load_state_dict({'gain': weights['ln1.weight']})
            self.pre_ff_norm.load_state_dict({'gain': weights['ln2.weight']})

            self.ff.load_state_dict({
                'linear1.weight': weights['ffn.w1.weight'],
                'linear2.weight': weights['ffn.w2.weight'],
                'linear3.weight': weights['ffn.w3.weight'],
            })
    
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        residual = in_features
        seq_len = residual.shape[-2]
        batch_size = residual.shape[0]
        token_positions = torch.arange(seq_len, device=residual.device).unsqueeze(0).expand(batch_size, -1)
        residual += self.mha.forward(self.pre_mha_norm.forward(residual), token_positions)
        residual += self.ff.forward(self.pre_ff_norm.forward(residual))
        return residual

class TransformerLM(nn.Module):
    def __init__(self,
                 voacb_size: int,
                 d_model: int, 
                 context_length: int,
                 num_layers: int,
                 num_heads: int, 
                 d_ff: int, 
                 theta: float,
                 weights: dict[str, torch.Tensor] | None = None, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.embedding = Embedding(voacb_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = [TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype) for i in range(num_layers)]
        self.post_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output = Linear(d_model, voacb_size, device=device, dtype=dtype)

        if weights is not None:
            # Token embeddings
            if 'token_embeddings.weight' in weights:
                self.embedding.load_state_dict({'embeddings': weights['token_embeddings.weight']})

            # Per-layer transformer block weights
            for layer_index, block in enumerate(self.transformer_blocks):
                prefix = f"layers.{layer_index}."

                # Attention projections: concatenate Q, K, V rows
                q_key = prefix + 'attn.q_proj.weight'
                k_key = prefix + 'attn.k_proj.weight'
                v_key = prefix + 'attn.v_proj.weight'
                o_key = prefix + 'attn.output_proj.weight'

                if all(k in weights for k in (q_key, k_key, v_key)):
                    qkv_combined = torch.cat((weights[q_key], weights[k_key], weights[v_key]), dim=0)
                    block.mha.attention_weights.load_state_dict({'weight': qkv_combined})

                if o_key in weights:
                    block.mha.output_weights.load_state_dict({'weight': weights[o_key]})

                # Layer norms
                ln1_key = prefix + 'ln1.weight'
                ln2_key = prefix + 'ln2.weight'
                if ln1_key in weights:
                    block.pre_mha_norm.load_state_dict({'gain': weights[ln1_key]})
                if ln2_key in weights:
                    block.pre_ff_norm.load_state_dict({'gain': weights[ln2_key]})

                # Feed-forward weights
                ffn_w1_key = prefix + 'ffn.w1.weight'
                ffn_w2_key = prefix + 'ffn.w2.weight'
                ffn_w3_key = prefix + 'ffn.w3.weight'
                if all(k in weights for k in (ffn_w1_key, ffn_w2_key, ffn_w3_key)):
                    block.ff.load_state_dict({
                        'linear1.weight': weights[ffn_w1_key],
                        'linear2.weight': weights[ffn_w2_key],
                        'linear3.weight': weights[ffn_w3_key],
                    })

            # Final RMSNorm
            if 'ln_final.weight' in weights:
                self.post_norm.load_state_dict({'gain': weights['ln_final.weight']})

            # LM head
            if 'lm_head.weight' in weights:
                self.output.load_state_dict({'weight': weights['lm_head.weight']})

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding.forward(in_indices)
        x = embeddings
        for block in self.transformer_blocks:
            x = block.forward(x)
        x = self.post_norm.forward(x)
        logits = self.output.forward(x)
        return logits
