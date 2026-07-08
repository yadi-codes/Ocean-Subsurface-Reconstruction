from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

import utils.config as config

from models.decoder import Decoder


def main():

    model = Decoder()

    x = torch.randn(

        2,

        config.NUM_PATCHES,

        config.EMBED_DIM,

        requires_grad=True,
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    loss = y.mean()

    loss.backward()

    print("✓ Decoder test passed")


if __name__ == "__main__":
    main()