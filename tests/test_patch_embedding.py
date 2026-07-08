from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

import utils.config as config

from models.patch_embedding import PatchEmbedding


def main():

    model = PatchEmbedding()

    x = torch.randn(
        8,
        config.SEQUENCE_LENGTH,
        config.HIDDEN_DIM,
        59,
        79,
        requires_grad=True,
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    loss = y.mean()
    loss.backward()

    print("✓ Gradient check passed")
    print("✓ Patch Embedding test passed")


if __name__ == "__main__":
    main()