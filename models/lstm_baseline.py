import torch
import torch.nn as nn
import numpy as np

class LSTMBaseline(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.1):
        super(LSTMBaseline, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):

        batch_size = x.size(0)

        lstm_out, (h_n, c_n) = self.lstm(x)

        last_output = lstm_out[:, -1, :]

        output = self.dropout(last_output)
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)

        return output

    def get_model_info(self):

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'LSTM Baseline',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim
        }