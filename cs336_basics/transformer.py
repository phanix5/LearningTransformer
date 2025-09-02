import math
import typing
import torch
import torch.nn as nn
from einops import einsum, reduce, rearrange


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute a numerically stable softmax over a given dimension.

    Args:
      x (torch.Tensor): (..., d_k, ...).
      dim (int): Dimension along which to compute softmax.

    Returns:
      torch.Tensor: Tensor of the same shape as `x` whose values sum to 1 along `dim`.
    
    Matrix Multiplication FLOPS:
      None
    """
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with optional masking.

    Args:
      queries (torch.Tensor): Query tensor of shape `(batch, ..., seq_len_q, d_k)`.
      keys (torch.Tensor): Key tensor of shape `(batch, ..., seq_len_k, d_k)`.
      values (torch.Tensor): Value tensor of shape `(batch, ..., seq_len_k, d_v)`.
      mask (torch.Tensor | None): Boolean mask broadcastable to attention
        weights of shape `(batch, ..., seq_len_q, seq_len_k)`, where `True`
        keeps a position and `False` masks it.

    Returns:
      torch.Tensor: Context tensor of shape `(batch, ..., seq_len_q, d_v)`.

    Notes:
      Uses `1/sqrt(d_k)` scaling and a numerically stable softmax.
    """
    attention_weights = einsum(queries, keys, "batch_size ... seq_len_n d_k, batch_size ... seq_len_m d_k -> batch_size ... seq_len_n seq_len_m")
    key_dim = queries.shape[-1]
    attention_weights = attention_weights / math.sqrt(key_dim)
    if mask is not None:
        # set attention weights at masked indexes to -inf so that soft max sets it to 0
        attention_weights = torch.where(mask, attention_weights, float("-inf"))
    attention_weights_normalized = softmax(attention_weights, -1)
    return einsum(attention_weights_normalized, values, "batch_size ... seq_len_n seq_len_m, batch_size ... seq_len_m d_v -> batch_size ... seq_len_n d_v")

class Linear(nn.Module):
    """
    A bias-free linear projection layer.

    Applies a linear transformation with weight of shape `(out_features, in_features)`
    to the last dimension of the input.

    Args:
      in_features (int): Size of each input sample (`d_in`).
      out_features (int): Size of each output sample (`d_out`).

    Parameters:
      weight (torch.nn.Parameter): Learnable weight matrix of shape `(d_out, d_in)`.

    Notes:
      Weights are initialized with a truncated normal distribution with standard
      deviation `2 / (in_features + out_features)`.
    """
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        weight_tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std_deviation = 2.0 / (in_features + out_features)
        nn.init.trunc_normal_(weight_tensor, mean=0, std=std_deviation, a=-3*math.sqrt(std_deviation), b=3*std_deviation)
        self.weight = nn.Parameter(weight_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear projection to the last dimension.

        Computes `y = x @ W.T` using `einsum` with pattern
        `... d_in, d_out d_in -> ... d_out`.

        Args:
          x (torch.Tensor): Input tensor of shape `(..., d_in)`.

        Returns:
          torch.Tensor: Output tensor of shape `(..., d_out)`.

        Matrix Multiplication FLOPS:
          O(d_in × d_out) per element of the leading dimensions of `x`.
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    """
    Learnable token embedding table.

    Args:
      num_embeddings (int): Vocabulary size.
      embedding_dim (int): Embedding dimensionality (`d_model`).

    Parameters:
      embeddings (torch.nn.Parameter): Weight matrix of shape `(num_embeddings, embedding_dim)`.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None ):
        super().__init__()
        embed_tensor = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(embed_tensor, mean=0, std=1.0, a=-3, b=3)
        self.embeddings = nn.Parameter(embed_tensor)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings for integer token IDs.

        Args:
          token_ids (torch.Tensor): Integer tensor of shape `(...,)` containing token indices in `[0, num_embeddings)`.

        Returns:
          torch.Tensor: Embeddings of shape `(..., embedding_dim)`.
        """
        return self.embeddings[token_ids]

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization without bias.

    Normalizes inputs along the last dimension and scales by a learned `gain`.

    Args:
      d_model (int): Size of the last dimension to normalize.
      eps (float): Small constant for numerical stability.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the last dimension.

        Args:
          x (torch.Tensor): Input tensor of shape `(..., d_model)`.

        Returns:
          torch.Tensor: Tensor of the same shape as `x`.

        Notes:
          Computations are performed in float32 for stability and cast back to the input dtype.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = reduce(x * x, 'batch seq d_model -> batch seq 1', 'mean')
        rms_norm = torch.sqrt(mean_square + self.eps)
        result = x / rms_norm * self.gain
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block.

    Computes `Linear2(SiLU(Linear1(x)) * Linear3(x))` with hidden size `d_ff`.

    Args:
      d_model (int): Model dimensionality.
      d_ff (int): Hidden dimensionality; if non-positive, defaults to `8/3 * d_model` rounded up to a multiple of 64.
    """
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
        """
        Apply the SwiGLU transformation.

        Args:
          x (torch.Tensor): Input tensor of shape `(..., d_model)`.

        Returns:
          torch.Tensor: Output tensor of shape `(..., d_model)`.
        """
        w1_forward = self.linear1.forward(x)
        # W2 @ (SiLU(W1 @ x) . (W3 @ x))
        return self.linear2.forward(w1_forward * torch.sigmoid(w1_forward) * self.linear3.forward(x))

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary positional embedding (RoPE) using cached cos/sin and rotate-half.

    Args:
      theta (float): Base frequency for RoPE.
      d_k (int): Key/query dimensionality (must be even).
      max_seq_len (int): Maximum sequence length to precompute cached tables for.
    """
    cos_table: torch.Tensor
    sin_table: torch.Tensor
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires even d_k"
        # Precompute cos/sin tables for positions [0, max_seq_len)
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, d_k/2)
        cos_table = torch.cos(angles).repeat_interleave(2, dim=-1)  # (max_seq_len, d_k)
        sin_table = torch.sin(angles).repeat_interleave(2, dim=-1)
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding to queries/keys.

        Args:
          x (torch.Tensor): Input tensor of shape `(..., seq_len, d_k)`.
          token_positions (torch.Tensor): Integer positions of shape `(..., seq_len)` matching x's sequence dims.

        Returns:
          torch.Tensor: Tensor of the same shape as `x`.
        """
        # Slice cached tables using token positions along the sequence dimension.
        # token_positions is expected to match the sequence dims of x.
        # If x includes a head dimension (e.g., batch, heads, seq, d_k),
        # we broadcast cos/sin over that head dimension by inserting a singleton.
        cos_vals = self.cos_table[token_positions].to(dtype=x.dtype)
        sin_vals = self.sin_table[token_positions].to(dtype=x.dtype)

        # Ensure cos/sin have a broadcastable head dimension when x has heads
        # x shape could be (..., seq_len, d_k) or (batch, heads, seq_len, d_k)
        # Typical MHA shapes: x: (batch, heads, seq, d_k), token_positions: (batch, seq)
        # In that case, x.ndim == token_positions.ndim + 2 and we need a singleton head axis.
        if token_positions.dim() == x.dim() - 2:
            cos_vals = cos_vals.unsqueeze(1)
            sin_vals = sin_vals.unsqueeze(1)

        # Split even/odd channels
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos_e = cos_vals[..., 0::2]
        sin_e = sin_vals[..., 0::2]

        # Apply 2D rotation per (even, odd) pair
        rotated_even = x_even * cos_e - x_odd * sin_e
        rotated_odd = x_even * sin_e + x_odd * cos_e

        # Interleave back into last dimension
        out = torch.empty_like(x)
        out[..., 0::2] = rotated_even
        out[..., 1::2] = rotated_odd
        return out
        

class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention with causal masking.

    Args:
      d_model (int): Model dimensionality.
      num_heads (int): Number of attention heads.
    """
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention_weights = Linear(d_model, 3 * self.num_heads * self.d_k, device=device, dtype=dtype)
        self.output_weights = Linear(self.num_heads * self.d_k, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute causal self-attention over a sequence.

        Args:
          x (torch.Tensor): Input tensor of shape `(batch, seq_len, d_model)`.

        Returns:
          torch.Tensor: Output tensor of shape `(batch, seq_len, d_model)`.
        """
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
    """
    Multi-head self-attention with rotary positional embeddings (RoPE).

    Args:
      d_model (int): Model dimensionality.
      num_heads (int): Number of attention heads.
      theta (float): Base frequency for RoPE.
      max_seq_len (int): Maximum sequence length.
    """
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention_weights = Linear(d_model, 3 * self.num_heads * self.d_k, device=device, dtype=dtype)
        self.output_weights = Linear(self.num_heads * self.d_k, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute causal self-attention with RoPE.

        Args:
          x (torch.Tensor): Input tensor of shape `(batch, seq_len, d_model)`.
          token_positions (torch.Tensor): Integer tensor of shape `(batch, seq_len)` with absolute positions.

        Returns:
          torch.Tensor: Output tensor of shape `(batch, seq_len, d_model)`.
        """
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
    """
    Pre-norm Transformer block with RoPE attention and SwiGLU feed-forward.

    Args:
      d_model (int): Model dimensionality.
      num_heads (int): Number of attention heads.
      d_ff (int): Feed-forward hidden dimensionality.
      max_seq_len (int): Maximum context length for RoPE.
      theta (float): Base frequency for RoPE.
      weights (dict[str, torch.Tensor] | None): Optional state dict with weight tensors.
    """
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
        """
        Apply attention and feed-forward sublayers with residual connections.

        Args:
          in_features (torch.Tensor): Input tensor of shape `(batch, seq_len, d_model)`.

        Returns:
          torch.Tensor: Tensor of shape `(batch, seq_len, d_model)`.
        """
        residual = in_features
        seq_len = residual.shape[-2]
        batch_size = residual.shape[0]
        token_positions = torch.arange(seq_len, device=residual.device).unsqueeze(0).expand(batch_size, -1)
        x = residual + self.mha.forward(residual, token_positions)
        x = x + self.ff.forward(x)
        return x

class TransformerLM(nn.Module):
    """
    Minimal Transformer language model with tied architecture components.

    Args:
      voacb_size (int): Vocabulary size (note: parameter name kept as-is).
      d_model (int): Model dimensionality.
      context_length (int): Maximum input sequence length.
      num_layers (int): Number of Transformer blocks.
      num_heads (int): Number of attention heads per block.
      d_ff (int): Feed-forward hidden dimensionality.
      theta (float): Base frequency for RoPE.
      weights (dict[str, torch.Tensor] | None): Optional state dict with weight tensors.
    """
    def __init__(self,
                 vocab_size: int,
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
        self.context_length = context_length
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype) for i in range(num_layers)]
        )
        self.post_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output = Linear(d_model, vocab_size, device=device, dtype=dtype)

        if weights is not None:
            # Token embeddings
            if 'token_embeddings.weight' in weights:
                self.embedding.load_state_dict({'embeddings': weights['token_embeddings.weight']})

            # Per-layer transformer block weights
            for layer_index, block in enumerate(self.transformer_blocks):
                blk: TransformerBlock = typing.cast(TransformerBlock, block)
                prefix = f"layers.{layer_index}."

                # Attention projections: concatenate Q, K, V rows
                q_key = prefix + 'attn.q_proj.weight'
                k_key = prefix + 'attn.k_proj.weight'
                v_key = prefix + 'attn.v_proj.weight'
                o_key = prefix + 'attn.output_proj.weight'

                if all(k in weights for k in (q_key, k_key, v_key)):
                    qkv_combined = torch.cat((weights[q_key], weights[k_key], weights[v_key]), dim=0)
                    blk.mha.attention_weights.load_state_dict({'weight': qkv_combined})

                if o_key in weights:
                    blk.mha.output_weights.load_state_dict({'weight': weights[o_key]})

                # Layer norms
                ln1_key = prefix + 'ln1.weight'
                ln2_key = prefix + 'ln2.weight'
                if ln1_key in weights:
                    blk.pre_mha_norm.load_state_dict({'gain': weights[ln1_key]})
                if ln2_key in weights:
                    blk.pre_ff_norm.load_state_dict({'gain': weights[ln2_key]})

                # Feed-forward weights
                ffn_w1_key = prefix + 'ffn.w1.weight'
                ffn_w2_key = prefix + 'ffn.w2.weight'
                ffn_w3_key = prefix + 'ffn.w3.weight'
                if all(k in weights for k in (ffn_w1_key, ffn_w2_key, ffn_w3_key)):
                    blk.ff.load_state_dict({
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
        """
        Compute per-token logits for input token indices.

        Args:
          in_indices (torch.Tensor): Integer input of shape `(batch, seq_len)`.

        Returns:
          torch.Tensor: Logits of shape `(batch, seq_len, voacb_size)`.
        """
        embeddings = self.embedding.forward(in_indices)
        x = embeddings
        for block in self.transformer_blocks:
            x = block.forward(x)
        # x = self.post_norm.forward(x)
        logits = self.output.forward(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prefix_token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens from a prefix using temperature and optional top-p sampling.

        Args:
          prefix_token_ids: Tensor of shape (batch, seq_len) with starting tokens.
          max_new_tokens: Maximum number of tokens to generate.
          temperature: Softmax temperature (>0 for sampling, 0 for greedy/argmax).
          top_p: If provided in (0, 1], apply nucleus sampling with this cumulative probability.
          eos_token_id: If provided, stop generation per sequence after this token is produced.

        Returns:
          Tensor of shape (batch, seq_len + generated_len) with generated token ids.
        """
        self.eval()
        device = next(self.parameters()).device
        vocab_size = self.output.weight.shape[0]

        if prefix_token_ids.dim() == 1:
            prefix_token_ids = prefix_token_ids.unsqueeze(0)
        generated_ids = prefix_token_ids.to(device)

        finished: torch.Tensor | None = None
        for _ in range(max_new_tokens):
            # Only feed the last context_length tokens to respect model's max context
            input_cond = generated_ids[:, -self.context_length:]

            logits = self.forward(input_cond)
            next_token_logits = logits[:, -1, :]

            if temperature is not None and temperature <= 0:
                # Greedy decoding from logits directly
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                temp = 1.0 if temperature is None else float(temperature)
                scaled_logits = next_token_logits / max(temp, 1e-8)
                probs = torch.softmax(scaled_logits, dim=-1)

                if top_p is not None and 0.0 < top_p < 1.0:
                    # Nucleus sampling: keep minimal set with cumulative prob >= top_p
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    # Mask tokens where cumulative probability exceeds p (excluding first above-threshold token)
                    mask = cumulative > top_p
                    # Shift mask right to keep the first token that makes cumulative exceed p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
                    # Unsort back to original indices
                    probs = torch.zeros_like(probs)
                    probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
                    # Re-normalize
                    probs = probs / probs.clamp_min(1e-12).sum(dim=-1, keepdim=True)

                next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                if finished is None:
                    finished = torch.zeros(generated_ids.size(0), dtype=torch.bool, device=device)
                # For sequences already finished, keep appending eos to maintain tensor shape
                eos_fill = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished.unsqueeze(-1), eos_fill, next_token)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if eos_token_id is not None:
                assert generated_ids.size(1) >= 1
                new_finished = next_token.squeeze(-1) == eos_token_id
                finished = finished | new_finished if finished is not None else new_finished
                if torch.all(finished):
                    break

        return generated_ids
