
import os
import json
import random
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# =========================
# PATH CONFIG
# =========================
DATA_DIR = r"C:\Train_LSTM\output\npy"
OUT_DIR  = r"C:\Train_LSTM\models"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["non_violence", "violence", "weapon"]
NUM_CLASSES = 3

# =========================
# TRAIN CONFIG
# =========================
USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTS = False
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 1.5

BATCH_SIZE = 48
N_EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-4
HIDDEN = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True
MAX_NORM = 5.0
PATIENCE = 10
SEED = 42

# =========================
# THRESHOLD CONFIG
# =========================
VIOLENCE_TH_GRID = np.round(np.arange(0.50, 0.81, 0.05), 2)
WEAPON_TH_GRID   = np.round(np.arange(0.65, 0.91, 0.05), 2) 

VIOLENCE_MARGIN_GRID = [0.00, 0.05, 0.08, 0.10]
WEAPON_MARGIN_GRID   = [0.00, 0.08, 0.12, 0.15]

FALSE_WEAPON_PENALTY = 0.15

# =========================
# SEED
# =========================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

# =========================
# DATA
# =========================
def load_build_metadata(data_dir: str) -> Dict:
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print("[WARN] Không thấy metadata.json. Sẽ dùng cấu hình suy luận từ shape.")
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split(data_dir: str) -> Tuple[np.ndarray, ...]:
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

    sw_path = os.path.join(data_dir, "sample_weights.npy")
    cw_path = os.path.join(data_dir, "class_weights.npy")
    sample_weights = np.load(sw_path) if os.path.exists(sw_path) else None
    class_weights = np.load(cw_path) if os.path.exists(cw_path) else None
    return X_train, y_train, X_val, y_val, X_test, y_test, sample_weights, class_weights


def adapt_input_shape(X: np.ndarray) -> np.ndarray:
    if X.ndim == 4:
        n, seq_len, win_size, feat_dim = X.shape
        return X.reshape(n, seq_len, win_size * feat_dim).astype(np.float32)
    if X.ndim == 3:
        return X.astype(np.float32)
    raise ValueError(f"Input phải là 3D hoặc 4D, nhận được shape={X.shape}")


build_metadata = load_build_metadata(DATA_DIR)
X_train, y_train, X_val, y_val, X_test, y_test, sample_weights_np, class_weights_np = load_split(DATA_DIR)

raw_train_shape = tuple(X_train.shape)
raw_val_shape = tuple(X_val.shape)
raw_test_shape = tuple(X_test.shape)

X_train = adapt_input_shape(X_train)
X_val = adapt_input_shape(X_val)
X_test = adapt_input_shape(X_test)

y_train = y_train.astype(np.int64)
y_val = y_val.astype(np.int64)
y_test = y_test.astype(np.int64)

if X_train.shape[0] == 0:
    raise ValueError("X_train rỗng. Hãy kiểm tra lại file build dữ liệu.")

seq_len = int(X_train.shape[1])
input_dim = int(X_train.shape[2])

if len(raw_train_shape) == 4:
    _, raw_seq_len, window_size, frame_feature_dim = raw_train_shape
else:
    raw_seq_len = seq_len
    window_size = int(build_metadata.get("window_size", 32))
    frame_feature_dim = int(input_dim // window_size) if window_size > 0 else 6

print(f"[INFO] Raw train shape: {raw_train_shape}")
print(f"[INFO] Raw val shape  : {raw_val_shape}")
print(f"[INFO] Raw test shape : {raw_test_shape}")
print(f"[INFO] Adapted train  : {X_train.shape}")
print(f"[INFO] Adapted val    : {X_val.shape}")
print(f"[INFO] Adapted test   : {X_test.shape}")
print(f"[INFO] seq_len={seq_len}, window_size={window_size}, frame_feature_dim={frame_feature_dim}, input_dim={input_dim}")
print(f"[INFO] sample_label_rule={build_metadata.get('sample_label_rule', 'unknown')}")

train_set = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_set = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_set = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

if USE_WEIGHTED_SAMPLER and sample_weights_np is not None and len(sample_weights_np) == len(train_set):
    sample_weights_t = torch.tensor(sample_weights_np, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=sample_weights_t, num_samples=len(sample_weights_t), replacement=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)
    print("[INFO] Using WeightedRandomSampler.")
else:
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print("[INFO] Using normal shuffled sampler.")

val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# =========================
# MODEL
# =========================
class AttentionPooling(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.score(x).squeeze(-1)
        attn = torch.softmax(attn, dim=1)
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)
        return pooled


class ActionLSTM(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.attn = AttentionPooling(out_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = self.attn(out)
        return self.head(pooled)


model = ActionLSTM(
    input_dim=input_dim,
    num_classes=NUM_CLASSES,
    hidden_size=HIDDEN,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

# =========================
# LOSS
# =========================
def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 1.5, weight=None):
    ce = F.cross_entropy(logits, targets, reduction="none", weight=weight)
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


weight_t = None
if USE_CLASS_WEIGHTS and class_weights_np is not None and len(class_weights_np) == NUM_CLASSES:
    weight_t = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    print("[INFO] Using class weights:", class_weights_np.tolist())
else:
    print("[INFO] No class weights.")

if USE_FOCAL_LOSS:
    def criterion(logits, targets):
        return focal_loss(logits, targets, gamma=FOCAL_GAMMA, weight=weight_t)
    print(f"[INFO] Using Focal Loss gamma={FOCAL_GAMMA}")
else:
    criterion = nn.CrossEntropyLoss(weight=weight_t)
    print("[INFO] Using CrossEntropyLoss")

# =========================
# METRIC / THRESHOLD
# =========================
def predict_by_threshold(
    probs: np.ndarray,
    violence_th: float,
    weapon_th: float,
    violence_margin: float,
    weapon_margin: float,
) -> np.ndarray:
    preds = []
    for p in probs:
        p_non, p_vio, p_wea = float(p[0]), float(p[1]), float(p[2])
        best_other_for_weapon = max(p_non, p_vio)
        best_other_for_violence = max(p_non, p_wea)

        if (p_wea >= weapon_th) and ((p_wea - best_other_for_weapon) >= weapon_margin):
            preds.append(2)
        elif (p_vio >= violence_th) and ((p_vio - best_other_for_violence) >= violence_margin):
            preds.append(1)
        else:
            preds.append(0)
    return np.asarray(preds, dtype=np.int64)


def compute_macro_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def collect_probs(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.softmax(logits, dim=1)
            ys.append(yb.numpy())
            ps.append(probs.cpu().numpy())
    y_true = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=np.int64)
    probs = np.concatenate(ps, axis=0) if ps else np.empty((0, NUM_CLASSES), dtype=np.float32)
    return y_true, probs


def tune_thresholds(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    best = {
        "score": -1.0,
        "macro_f1": -1.0,
        "acc": 0.0,
        "violence_th": 0.65,
        "weapon_th": 0.80,
        "violence_margin": 0.08,
        "weapon_margin": 0.12,
        "false_weapon_rate_from_non": 1.0,
    }

    non_mask = (y_true == 0)
    non_count = int(non_mask.sum())

    for vth in VIOLENCE_TH_GRID:
        for wth in WEAPON_TH_GRID:
            for vm in VIOLENCE_MARGIN_GRID:
                for wm in WEAPON_MARGIN_GRID:
                    pred = predict_by_threshold(probs, vth, wth, vm, wm)
                    metrics = compute_macro_metrics(y_true, pred)
                    false_weapon = int(((y_true == 0) & (pred == 2)).sum())
                    false_weapon_rate = false_weapon / max(non_count, 1)
                    score = metrics["f1"] - FALSE_WEAPON_PENALTY * false_weapon_rate

                    if score > best["score"]:
                        best.update({
                            "score": float(score),
                            "macro_f1": float(metrics["f1"]),
                            "acc": float(metrics["acc"]),
                            "violence_th": float(vth),
                            "weapon_th": float(wth),
                            "violence_margin": float(vm),
                            "weapon_margin": float(wm),
                            "false_weapon_rate_from_non": float(false_weapon_rate),
                        })
    return best


def save_cm(y_true, y_pred, fname_prefix: str):
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(OUT_DIR, f"{fname_prefix}.csv"), encoding="utf-8-sig"
    )

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title(fname_prefix)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{fname_prefix}.png"), dpi=200)
    plt.close()
    return cm


# =========================
# TRAIN LOOP
# =========================
train_losses = []
val_f1_argmaxs = []
val_f1_thresholds = []
best_val_score = -1.0
best_epoch = -1
best_thresholds = None
patience_counter = 0
best_path = os.path.join(OUT_DIR, "best_lstm.pth")

for ep in range(1, N_EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)
        optimizer.step()
        running_loss += loss.item() * Xb.size(0)

    train_loss = running_loss / max(len(train_set), 1)
    train_losses.append(train_loss)

    yv, pv = collect_probs(val_loader)
    pred_argmax = pv.argmax(axis=1)
    argmax_metrics = compute_macro_metrics(yv, pred_argmax)

    threshold_info = tune_thresholds(yv, pv)
    pred_th = predict_by_threshold(
        pv,
        threshold_info["violence_th"],
        threshold_info["weapon_th"],
        threshold_info["violence_margin"],
        threshold_info["weapon_margin"],
    )
    threshold_metrics = compute_macro_metrics(yv, pred_th)

    val_f1_argmaxs.append(argmax_metrics["f1"])
    val_f1_thresholds.append(threshold_metrics["f1"])

    scheduler.step(threshold_info["score"])
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {ep:02d}/{N_EPOCHS} | "
        f"loss={train_loss:.4f} | "
        f"val_f1_argmax={argmax_metrics['f1']:.4f} | "
        f"val_f1_threshold={threshold_metrics['f1']:.4f} | "
        f"score={threshold_info['score']:.4f} | "
        f"v_th={threshold_info['violence_th']:.2f} | "
        f"w_th={threshold_info['weapon_th']:.2f} | "
        f"lr={current_lr:.6f}"
    )

    if threshold_info["score"] > best_val_score:
        best_val_score = threshold_info["score"]
        best_epoch = ep
        best_thresholds = threshold_info
        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "window_size": int(window_size),
                "frame_feature_dim": int(frame_feature_dim),
                "num_classes": NUM_CLASSES,
                "hidden_size": HIDDEN,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "bidirectional": BIDIRECTIONAL,
                "seq_len": seq_len,
                "raw_seq_len": int(raw_seq_len),
                "class_names": CLASS_NAMES,
                "sample_label_rule": build_metadata.get(
                    "sample_label_rule",
                    "majority label among windows in sequence"
                ),
                "build_metadata": build_metadata,
                "best_val_score": float(best_val_score),
                "best_val_macroF1_threshold": float(threshold_metrics["f1"]),
                "best_val_macroF1_argmax": float(argmax_metrics["f1"]),
                "best_thresholds": best_thresholds,
                "epoch": ep,
                "model_type": "AttentionLSTM_majority_label_threshold_tuned",
            },
            best_path,
        )
        print("[INFO] Saved best model with thresholds and build metadata.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"[INFO] Early stopping at epoch {ep}.")
            break

print("[INFO] Finished training.")
print("[INFO] Best thresholds:", best_thresholds)

# =========================
# FINAL TEST
# =========================
checkpoint = torch.load(best_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

best_thresholds = checkpoint.get("best_thresholds", best_thresholds)

yt, pt = collect_probs(test_loader)
pred_test_argmax = pt.argmax(axis=1)
pred_test_threshold = predict_by_threshold(
    pt,
    best_thresholds["violence_th"],
    best_thresholds["weapon_th"],
    best_thresholds["violence_margin"],
    best_thresholds["weapon_margin"],
)

argmax_metrics = compute_macro_metrics(yt, pred_test_argmax)
threshold_metrics = compute_macro_metrics(yt, pred_test_threshold)

save_cm(yt, pred_test_argmax, "conf_matrix_test_argmax")
save_cm(yt, pred_test_threshold, "conf_matrix_test_threshold")

report_argmax = classification_report(
    yt, pred_test_argmax,
    labels=range(NUM_CLASSES),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)
report_threshold = classification_report(
    yt, pred_test_threshold,
    labels=range(NUM_CLASSES),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

with open(os.path.join(OUT_DIR, "classification_report_argmax.txt"), "w", encoding="utf-8") as f:
    f.write(report_argmax)
with open(os.path.join(OUT_DIR, "classification_report_threshold.txt"), "w", encoding="utf-8") as f:
    f.write(report_threshold)

print("\n[INFO] Test report - argmax")
print(report_argmax)
print("\n[INFO] Test report - threshold")
print(report_threshold)

pred_df = pd.DataFrame({
    "y_true": yt,
    "pred_argmax": pred_test_argmax,
    "pred_threshold": pred_test_threshold,
    "p_non_violence": pt[:, 0],
    "p_violence": pt[:, 1],
    "p_weapon": pt[:, 2],
})
pred_df.to_csv(os.path.join(OUT_DIR, "test_predictions_with_probs.csv"), index=False, encoding="utf-8-sig")

history_df = pd.DataFrame({
    "epoch": list(range(1, len(train_losses) + 1)),
    "train_loss": train_losses,
    "val_f1_argmax": val_f1_argmaxs,
    "val_f1_threshold": val_f1_thresholds,
})
history_df.to_csv(os.path.join(OUT_DIR, "history_threshold_train.csv"), index=False, encoding="utf-8-sig")

final_metrics = {
    "raw_train_shape": raw_train_shape,
    "raw_val_shape": raw_val_shape,
    "raw_test_shape": raw_test_shape,
    "adapted_train_shape": tuple(X_train.shape),
    "adapted_val_shape": tuple(X_val.shape),
    "adapted_test_shape": tuple(X_test.shape),
    "seq_len": int(seq_len),
    "window_size": int(window_size),
    "frame_feature_dim": int(frame_feature_dim),
    "input_dim": int(input_dim),
    "num_classes": int(NUM_CLASSES),
    "class_names": CLASS_NAMES,
    "sample_label_rule": build_metadata.get("sample_label_rule", "unknown"),
    "best_epoch": int(best_epoch),
    "best_val_score": float(best_val_score),
    "best_thresholds": best_thresholds,
    "test_argmax": argmax_metrics,
    "test_threshold": threshold_metrics,
    "train_config": {
        "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "use_focal_loss": USE_FOCAL_LOSS,
        "focal_gamma": FOCAL_GAMMA,
        "batch_size": BATCH_SIZE,
        "epochs_requested": N_EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "hidden": HIDDEN,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "bidirectional": BIDIRECTIONAL,
        "patience": PATIENCE,
    },
}
with open(os.path.join(OUT_DIR, "final_metrics_threshold_train.json"), "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, ensure_ascii=False, indent=2)

print(f"[INFO] Done. Saved artifacts in: {OUT_DIR}")
print("[INFO] Best thresholds + build metadata saved inside best_lstm.pth")
