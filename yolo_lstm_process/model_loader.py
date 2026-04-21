# model_loader.py
import torch
import torch.nn as nn
from ultralytics import YOLO

from config import DEVICE, YOLO_MODEL_PATH, LSTM_MODEL_PATH


class ViolenceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            last_hidden = torch.cat([forward_last, backward_last], dim=1)
        else:
            last_hidden = h_n[-1]

        last_hidden = self.dropout(last_hidden)
        
        logits = self.fc(last_hidden)
        return logits