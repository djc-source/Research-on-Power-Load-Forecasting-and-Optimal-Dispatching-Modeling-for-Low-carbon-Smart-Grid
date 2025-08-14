import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LSTMTransformerModel(nn.Module):

    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, transformer_dim, 
                 transformer_layers, num_heads, sequence_length, output_dim=1, dropout=0.1):
        super(LSTMTransformerModel, self).__init__()

        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.transformer_dim = transformer_dim
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )

        self.projection = nn.Linear(lstm_hidden_dim, transformer_dim)

        self.pos_encoding = PositionalEncoding(transformer_dim, max_len=sequence_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=transformer_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(transformer_dim, transformer_dim // 2)
        self.fc2 = nn.Linear(transformer_dim // 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        lstm_out, (h_n, c_n) = self.lstm(x)

        transformer_input = self.projection(lstm_out)

        transformer_input = transformer_input.transpose(0, 1)
        transformer_input = self.pos_encoding(transformer_input)
        transformer_input = transformer_input.transpose(0, 1)

        transformer_out = self.transformer_encoder(transformer_input)

        query = transformer_out
        key = transformer_out
        value = transformer_out

        attn_output, attn_weights = self.attention(query, key, value)

        h_out = attn_output[:, -1, :]

        output = self.dropout(h_out)
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)

        return output, attn_weights

    def get_model_info(self):

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'LSTM-Transformer Hybrid',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'lstm_layers': self.lstm_layers,
            'transformer_dim': self.transformer_dim,
            'transformer_layers': self.transformer_layers,
            'num_heads': self.num_heads,
            'sequence_length': self.sequence_length,
            'output_dim': self.output_dim
        }