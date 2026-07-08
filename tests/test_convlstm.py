from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from models.convlstm import ConvLSTM


def main():

    model = ConvLSTM(
        input_dim=5,
        hidden_dim=64,
    )

    x = torch.randn(
        8,
        3,
        5,
        59,
        79,
        requires_grad=True
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    # Gradient test
    loss = y.mean()
    loss.backward()

    print("✓ Backpropagation successful")


if __name__ == "__main__":
    main()