
from functools import lru_cache
import torch
from ultralytics import YOLO

from .ai_config import (
    YOLO_MODEL_PATH,
    LSTM_MODEL_PATH,
    DEVICE,
    LSTM_INPUT_DIM,
    NUM_CLASSES,
)


class AttentionPooling(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.score = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x):
        attn = self.score(x).squeeze(-1)
        attn = torch.softmax(attn, dim=1)
        return torch.sum(x * attn.unsqueeze(-1), dim=1)


class ActionLSTMAttention(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.attn = AttentionPooling(out_dim)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(out_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(out_dim, hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        pooled = self.attn(out)
        return self.head(pooled)


class ActionLSTMLegacy(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(out_dim, hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out_last = out[:, -1, :]
        return self.head(out_last)


@lru_cache(maxsize=1)
def load_models():
 
    print('[INFO] Loading YOLO...')
    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    print('[INFO] Loading LSTM...')
    checkpoint = torch.load(str(LSTM_MODEL_PATH), map_location=DEVICE)
    state_dict = checkpoint['model_state_dict']

    input_dim = int(checkpoint.get('input_dim', LSTM_INPUT_DIM))
    hidden_size = int(checkpoint.get('hidden_size', 128))
    num_layers = int(checkpoint.get('num_layers', 2))
    bidirectional = bool(checkpoint.get('bidirectional', True))
    num_classes = int(checkpoint.get('num_classes', NUM_CLASSES))
    dropout = float(checkpoint.get('dropout', 0.3))

    use_attention = any(k.startswith('attn.') for k in state_dict.keys())

    if use_attention:
        print('[INFO] Using Attention LSTM')
        lstm_model = ActionLSTMAttention(input_dim, num_classes, hidden_size, num_layers, dropout, bidirectional)
    else:
        print('[INFO] Using Legacy LSTM')
        lstm_model = ActionLSTMLegacy(input_dim, num_classes, hidden_size, num_layers, dropout, bidirectional)

    lstm_model.load_state_dict(state_dict, strict=True)
    lstm_model.to(DEVICE)
    lstm_model.eval()

    print(f'[INFO] Models loaded successfully | LSTM input_dim={input_dim}')
    return yolo_model, lstm_model
