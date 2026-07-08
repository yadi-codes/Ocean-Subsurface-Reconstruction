"""
Multi-Head Self Attention for ConvFormer.

Input:
    (B, N, D)

Output:
    (B, N, D)
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.config as config


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self Attention.

    This module will be reused for:

    - Temporal Attention
    - Spatial Attention
    """

    def __init__(
        self,
        embed_dim: int = config.EMBED_DIM,
        num_heads: int = config.NUM_HEADS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            "Embedding dimension must be divisible by number of heads."
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x : (B, N, D)

        Returns:
            (B, N, D)
        """

        B, N, D = x.shape

        # -------------------------
        # Q, K, V
        # -------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B,H,N,Dh)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # -------------------------
        # Attention
        # -------------------------

        scores = (q @ k.transpose(-2, -1)) * self.scale

        attention = torch.softmax(scores, dim=-1)

        attention = self.dropout(attention)

        out = attention @ v

        # -------------------------
        # Merge heads
        # -------------------------

        out = out.transpose(1, 2).contiguous()

        out = out.view(B, N, D)

        out = self.out_proj(out)

        return out