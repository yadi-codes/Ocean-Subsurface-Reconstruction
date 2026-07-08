"""
ConvFormer Encoder Block.

Input:
    (B, T, N, D)

Output:
    (B, T, N, D)
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.config as config

from models.attention import MultiHeadAttention


class MLP(nn.Module):
    """
    Feed Forward Network used inside the Transformer.
    """

    def __init__(
        self,
        embed_dim: int = config.EMBED_DIM,
        mlp_dim: int = config.MLP_DIM,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """
    One ConvFormer Encoder Block.

    Consists of:
        LayerNorm
        Temporal Attention
        Residual

        LayerNorm
        MLP
        Residual

        LayerNorm
        Spatial Attention
        Residual

        LayerNorm
        MLP
        Residual
    """

    def __init__(self):
        super().__init__()

        self.temporal_norm = nn.LayerNorm(config.EMBED_DIM)
        self.temporal_attn = MultiHeadAttention()

        self.temporal_mlp_norm = nn.LayerNorm(config.EMBED_DIM)
        self.temporal_mlp = MLP()

        self.spatial_norm = nn.LayerNorm(config.EMBED_DIM)
        self.spatial_attn = MultiHeadAttention()

        self.spatial_mlp_norm = nn.LayerNorm(config.EMBED_DIM)
        self.spatial_mlp = MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            (B,T,N,D)
        """

        B, T, N, D = x.shape

        # ----------------------------------
        # Temporal Attention
        # ----------------------------------

        temp = x.reshape(B * N, T, D)

        temp = temp + self.temporal_attn(
            self.temporal_norm(temp)
        )

        temp = temp + self.temporal_mlp(
            self.temporal_mlp_norm(temp)
        )

        temp = temp.reshape(B, N, T, D).permute(0, 2, 1, 3)

        # ----------------------------------
        # Spatial Attention
        # ----------------------------------

        spatial = temp.reshape(B * T, N, D)

        spatial = spatial + self.spatial_attn(
            self.spatial_norm(spatial)
        )

        spatial = spatial + self.spatial_mlp(
            self.spatial_mlp_norm(spatial)
        )

        spatial = spatial.reshape(B, T, N, D)

        return spatial