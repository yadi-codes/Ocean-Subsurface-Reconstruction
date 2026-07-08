"""
Patch Embedding for ConvFormer.

Input:
    (B, T, C, H, W)

Output:
    (B, T, N, D)

N = Number of patches
D = Embedding dimension
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.config as config


class PatchEmbedding(nn.Module):
    """
    Converts ConvLSTM feature maps into patch embeddings
    for the Transformer encoder.
    """

    def __init__(
        self,
        in_channels: int = config.HIDDEN_DIM,
        embed_dim: int = config.EMBED_DIM,
        patch_size: int = config.PATCH_SIZE,
    ):
        super().__init__()

        self.patch_size = patch_size

        # Performs patch extraction + linear projection
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, C, H, W)

        Returns
        -------
        (B, T, N, D)
        """

        B, T, C, H, W = x.shape

        outputs = []

        for t in range(T):

            # (B, C, H, W)
            xt = x[:, t]

            # (B, D, H', W')
            xt = self.proj(xt)

            # Save new spatial size
            _, D, Hp, Wp = xt.shape

            # (B, D, N)
            xt = xt.flatten(2)

            # (B, N, D)
            xt = xt.transpose(1, 2)

            outputs.append(xt)

        # (B, T, N, D)
        return torch.stack(outputs, dim=1)