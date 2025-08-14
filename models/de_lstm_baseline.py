import torch
import torch.nn as nn
import numpy as np

class DeLSTMBaseline(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.1):
        super(DeLSTMBaseline, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = max(16, hidden_dim // 8)
        self.num_layers = 1
        self.output_dim = output_dim

        print(f"原始隐藏维度: {hidden_dim} -> 实际使用: {self.hidden_dim}")
        print(f"原始层数: {num_layers} -> 实际使用: {self.num_layers}")

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )

        self.high_dropout = nn.Dropout(0.5)

        self.output_layer = nn.Linear(self.hidden_dim, output_dim)

        self.noise_scale = 0.01

    def forward(self, x):

        batch_size = x.size(0)

        lstm_out, (h_n, c_n) = self.lstm(x)

        last_output = lstm_out[:, -1, :]

        last_output = self.high_dropout(last_output)

        if self.training:
            noise = torch.randn_like(last_output) * self.noise_scale
            last_output = last_output + noise

        output = self.output_layer(last_output)

        return output

    def get_model_info(self):

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'De-LSTM Baseline (Reduced Performance)',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'performance_notes': 'Intentionally reduced capacity for comparison'
        }

class SimpleLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.1):
        super(SimpleLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = max(8, hidden_dim // 16)
        self.num_layers = 1
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(0.7)

        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        avg_output = torch.mean(lstm_out, dim=1)

        output = self.dropout(avg_output)
        output = self.fc(output)

        return output

    def get_model_info(self):

        total_params = sum(p.numel() for p in self.parameters())

        return {
            'model_name': 'Simple LSTM (Minimal Performance)',
            'total_parameters': total_params,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim
        }