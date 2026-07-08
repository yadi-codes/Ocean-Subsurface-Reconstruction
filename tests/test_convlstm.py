from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

import utils.config as config

from models.convlstm import ConvLSTM


def main():

    model = ConvLSTM()

    x = torch.randn(
        8,
        config.SEQUENCE_LENGTH,
        config.INPUT_CHANNELS,
        59,
        79,
        requires_grad=True,
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    loss = y.mean()
    loss.backward()

    print("✓ Backpropagation successful")
    print("✓ ConvLSTM test passed")


if __name__ == "__main__":
    main()