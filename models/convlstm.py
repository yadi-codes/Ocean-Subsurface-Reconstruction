"""
ConvLSTM implementation used as the positional encoding layer
for the ConvFormer architecture.

Input:
    (B, T, C, H, W)

Output:
    (B, T, hidden_dim, H, W)
"""

from typing import Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)

        gates = self.conv(combined)

        # Split into four gates
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c


class ConvLSTM(nn.Module):
    """
    ConvLSTM over a sequence.

    Input:
        (B,T,C,H,W)

    Output:
        (B,T,HIDDEN,H,W)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.cell = ConvLSTMCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):

        B, T, C, H, W = x.shape

        device = x.device

        h = torch.zeros(
            B,
            self.hidden_dim,
            H,
            W,
            device=device,
        )

        c = torch.zeros_like(h)

        outputs = []

        for t in range(T):

            h, c = self.cell(
                x[:, t],
                h,
                c,
            )

            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)

        return outputs