import torch
import torch.nn as nn


class LSTMWithLinearLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        self._lstm = nn.LSTM(input_size, hidden_size)
        self._linear = nn.Linear(hidden_size, output_size)

    def forward(self, x_t: torch.Tensor):
        _, (h_t, _) = self._lstm(x_t)

        return self._linear(h_t)
