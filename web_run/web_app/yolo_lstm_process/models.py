import torch
import torch.nn as nn


# =========================
# MODEL DEFINITIONS
# =========================
class AttentionPooling(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.score = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.score(x).squeeze(-1)
        attn = torch.softmax(attn, dim=1)
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)
        return pooled


class ActionLSTMAttention(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = self.attn(out)
        logits = self.head(pooled)
        return logits


class ActionLSTMLegacy(torch.nn.Module):
    """
    Fallback nếu checkpoint không có attention.
    Vẫn yêu cầu input_dim đúng 576 nếu chạy realtime 18 feature.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out_last = out[:, -1, :]
        logits = self.head(out_last)
        return logits

