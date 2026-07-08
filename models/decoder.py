"""
ConvFormer Decoder.

Converts patch embeddings back into spatial feature maps.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.config as config


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.patch_projection = nn.Linear(
            config.EMBED_DIM,
            config.PATCH_SIZE * config.PATCH_SIZE * config.EMBED_DIM,
        )

        self.prediction = nn.Sequential(

            nn.Conv2d(
                config.EMBED_DIM,
                128,
                kernel_size=3,
                padding=1,
            ),

            nn.GELU(),

            nn.Conv2d(
                128,
                config.OUTPUT_CHANNELS * config.DEPTH_LEVELS,
                kernel_size=1,
            ),
        )

    def forward(self, x):

        """
        x:
            (B, N, D)
        """

        B, N, D = x.shape

        x = self.patch_projection(x)

        x = x.view(
            B,
            config.PATCH_HEIGHT,
            config.PATCH_WIDTH,
            config.PATCH_SIZE,
            config.PATCH_SIZE,
            config.EMBED_DIM,
        )

        # Unpatchify

        x = x.permute(
            0,
            5,
            1,
            3,
            2,
            4,
        )

        x = x.reshape(
            B,
            config.EMBED_DIM,
            config.PATCH_HEIGHT * config.PATCH_SIZE,
            config.PATCH_WIDTH * config.PATCH_SIZE,
        )

        # Upsample back to original grid

        x = nn.functional.interpolate(
            x,
            size=(
                config.GRID_HEIGHT,
                config.GRID_WIDTH,
            ),
            mode="bilinear",
            align_corners=False,
        )

        x = self.prediction(x)

        x = x.view(
            B,
            config.OUTPUT_CHANNELS,
            config.DEPTH_LEVELS,
            config.GRID_HEIGHT,
            config.GRID_WIDTH,
        )

        return x