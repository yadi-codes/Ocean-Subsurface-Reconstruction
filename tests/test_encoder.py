from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

import utils.config as config

from models.encoder import EncoderBlock


def main():

    model = EncoderBlock()

    x = torch.randn(
        2,
        config.SEQUENCE_LENGTH,
        4661,
        config.EMBED_DIM,
        requires_grad=True,
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    loss = y.mean()
    loss.backward()

    print("✓ Gradient check passed")
    print("✓ Encoder test passed")


if __name__ == "__main__":
    main()