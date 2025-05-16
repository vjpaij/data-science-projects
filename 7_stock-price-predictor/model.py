"""
This module contains the definition of the LSTM model used for predicting stock prices.

input_size: Number of expected features in the input. For univariate time series, this is typically 1.
hidden_size: Number of features in the hidden state.
num_layers: Number of recurrent layers.
dropout: Dropout probability for regularization.
"""

import torch
import torch.nn as nn

# Assuming these were the best params found by Optuna
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

