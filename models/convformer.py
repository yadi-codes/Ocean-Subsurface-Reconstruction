"""
Complete ConvFormer model.

Input:
    (B,T,5,H,W)

Output:
    (B,2,58,H,W)
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.config as config

from models.convlstm import ConvLSTM
from models.patch_embedding import PatchEmbedding
from models.encoder import EncoderBlock
from models.decoder import Decoder


class ConvFormer(nn.Module):

    def __init__(self):
        super().__init__()

        self.convlstm = ConvLSTM()

        self.patch_embedding = PatchEmbedding()

        self.encoder = nn.Sequential(
            *[
                EncoderBlock()
                for _ in range(config.NUM_ENCODER_LAYERS)
            ]
        )

        self.decoder = Decoder()

    def forward(self, x):

        # -------------------------
        # Positional Encoding
        # -------------------------

        x = self.convlstm(x)

        # -------------------------
        # Patch Embedding
        # -------------------------

        x = self.patch_embedding(x)

        # -------------------------
        # ConvFormer Encoder
        # -------------------------

        x = self.encoder(x)

        # -------------------------
        # Use latest timestep
        # -------------------------

        x = x[:, -1]

        # -------------------------
        # Decode
        # -------------------------

        x = self.decoder(x)

        return x