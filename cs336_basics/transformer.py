import math
import torch
import torch.nn as nn
from einops import einsum

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

