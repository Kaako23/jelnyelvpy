"""LSTM model for sign language sequence classification."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """3-layer LSTM classifier."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x, (h, _) = self.lstm1(x)
        x, (h, _) = self.lstm2(x)
        x, (h, _) = self.lstm3(x)
        x = h[-1, :, :]  # last layer hidden: (batch, hidden)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
