
import os
import json
import ast
import re
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# =====================
# PATH CONFIG
# =====================
TRAIN_PATH = r"C:\Train_LSTM\inputs\train.csv"
VAL_PATH   = r"C:\Train_LSTM\inputs\val.csv"
TEST_PATH  = r"C:\Train_LSTM\inputs\test.csv"
SAVE_DIR   = r"C:\Train_LSTM\output\npy"

# =====================
# DATA CONFIG
# =====================
WINDOW_SIZE = 32
FRAME_STRIDE = 1

SEQ_LEN = 16
SEQ_STRIDE = 1


GROUP_COLS = ["videoID", "segment_id"]
FRAME_ORDER_COL = "Frame_number"

YOLO_ONEHOT_COL = "One-hot"
CONF_SCORE_COL = "conf-score"
GT_LABEL_COL = "check_video"

NUM_CLASSES = 3
CLASS_NAMES = ["non_violence", "violence", "weapon"]

CONF_LABEL_THRES = 0.75
VIOLENCE_MIN_SCORE = 28
WEAPON_MIN_SCORE   = 22
YOLO_BONUS_THRES   = 0.3
YOLO_BONUS_SCORE   = 0.2

if VIOLENCE_MIN_SCORE > WINDOW_SIZE:
    raise ValueError(f"VIOLENCE_MIN_RUN={VIOLENCE_MIN_SCORE} không được lớn hơn WINDOW_SIZE={WINDOW_SIZE}")
if WEAPON_MIN_SCORE > WINDOW_SIZE:
    raise ValueError(f"WEAPON_MIN_RUN={WEAPON_MIN_SCORE} không được lớn hơn WINDOW_SIZE={WINDOW_SIZE}")

CLASS_WEIGHT_BETA = 0.999
CLASS_WEIGHT_CLIP_MIN = 0.25
CLASS_WEIGHT_CLIP_MAX = 4.00

def max_consecutive_true(mask):
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def parse_vector_cell(x) -> np.ndarray:
    if isinstance(x, np.ndarray):        return x.astype(np.float32)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if pd.isna(x):
        return np.asarray([], dtype=np.float32)
    if not isinstance(x, str):
        return np.asarray([], dtype=np.float32)

    s = x.strip()
    try:
        v = ast.literal_eval(s)
        return np.asarray(v, dtype=np.float32)
    except Exception:
        pass

    s = s.strip("[]").replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return np.asarray([], dtype=np.float32)
    return np.fromstring(s, sep=" ").astype(np.float32)


def decide_window_label_from_gt_and_conf(
    frame_gt_labels: List[int],
    frame_conf_scores: List[np.ndarray],
) -> int:
    violence_score = 0.0
    weapon_score = 0.0

    for gt, conf_vec in zip(frame_gt_labels, frame_conf_scores):
        gt = int(gt)
        conf_vec = np.asarray(conf_vec, dtype=np.float32)

        if len(conf_vec) != NUM_CLASSES:
            continue

        # check_video là nhãn thật, nên được tính chính
        if gt == 1:
            violence_score += 1.0

            # YOLO đúng thì cộng thêm điểm phụ
            if float(conf_vec[1]) >= YOLO_BONUS_THRES:
                violence_score += YOLO_BONUS_SCORE

        elif gt == 2:
            weapon_score += 1.0

            if float(conf_vec[2]) >= YOLO_BONUS_THRES:
                weapon_score += YOLO_BONUS_SCORE

    # Ưu tiên weapon nếu đủ điểm
    if weapon_score >= WEAPON_MIN_SCORE:
        return 2
    if violence_score >= VIOLENCE_MIN_SCORE:
        return 1
    return 0


def decide_sequence_label_by_majority(seq_window_labels: np.ndarray) -> int:
    seq_window_labels = np.asarray(seq_window_labels, dtype=np.int64)
    counts = np.bincount(seq_window_labels, minlength=NUM_CLASSES)

    max_count = counts.max()

    # Tie-break theo mức độ nguy hiểm.
    if counts[2] == max_count:
        return 2
    if counts[1] == max_count:
        return 1
    return 0


def build_windows(
    frame_features: List[np.ndarray],
    frame_gt_labels: List[int],
    frame_conf_scores: List[np.ndarray],
    window_size: int = WINDOW_SIZE,
    stride: int = FRAME_STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tạo các window 32 frame và nhãn window."""
    Xw, yw = [], []
    n = len(frame_features)

    for i in range(0, n - window_size + 1, stride):
        feats = frame_features[i:i + window_size]
        labs = frame_gt_labels[i:i + window_size]
        confs = frame_conf_scores[i:i + window_size]

        Xw.append(np.stack(feats).astype(np.float32))
        yw.append(decide_window_label_from_gt_and_conf(labs, confs))

    if not Xw:
        feat_dim = len(frame_features[0]) if frame_features else 6
        return np.empty((0, window_size, feat_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.asarray(Xw, dtype=np.float32), np.asarray(yw, dtype=np.int64)


def build_sequences_from_windows(
    Xw: np.ndarray,
    yw: np.ndarray,
    seq_len: int = SEQ_LEN,
    seq_stride: int = SEQ_STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    n = len(Xw)

    for i in range(0, n - seq_len + 1, seq_stride):
        seq_x = Xw[i:i + seq_len].astype(np.float32)
        seq_y = yw[i:i + seq_len]

        Xs.append(seq_x)
        ys.append(int(seq_y[-1]))

    if not Xs:
        feat_dim = Xw.shape[-1] if Xw.ndim == 3 and Xw.shape[0] > 0 else 6
        win_size = Xw.shape[1] if Xw.ndim == 3 and Xw.shape[0] > 0 else WINDOW_SIZE
        return np.empty((0, seq_len, win_size, feat_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def process_csv_to_lstm_data(
    file_path: str,
    yolo_onehot_col: str = YOLO_ONEHOT_COL,
    conf_score_col: str = CONF_SCORE_COL,
    label_col: str = GT_LABEL_COL,
    group_cols: Optional[List[str]] = None,
    frame_order_col: str = FRAME_ORDER_COL,
    window_size: int = WINDOW_SIZE,
    frame_stride: int = FRAME_STRIDE,
    seq_len: int = SEQ_LEN,
    seq_stride: int = SEQ_STRIDE,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    required = [yolo_onehot_col, conf_score_col, label_col]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Thiếu cột bắt buộc: '{col}' trong {file_path}")

    if group_cols is None:
        auto_group = [c for c in ["videoID", "video_id"] if c in df.columns]
        group_cols = auto_group if auto_group else ["__all__"]

    # Nếu CSV chưa có segment_id thì báo rõ để tránh build sai sau khi split theo segment.
    for col in group_cols:
        if col not in df.columns and col != "__all__":
            raise KeyError(
                f"Thiếu cột group '{col}' trong {file_path}. "
                "Nếu đang dùng split theo segment, hãy chạy lại file split để tạo segment_id."
            )

    if "__all__" in group_cols:
        df["__all__"] = 0

    before_rows = len(df)

    df["yolo_onehot_vec"] = df[yolo_onehot_col].apply(parse_vector_cell)
    df["conf_score_vec"] = df[conf_score_col].apply(parse_vector_cell)

    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    df = df[df["yolo_onehot_vec"].map(lambda a: len(a) == NUM_CLASSES)].reset_index(drop=True)
    df = df[df["conf_score_vec"].map(lambda a: len(a) == NUM_CLASSES)].reset_index(drop=True)

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    df[label_col] = df[label_col].astype(int)
    df = df[df[label_col].isin([0, 1, 2])].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{file_path}: không còn dòng hợp lệ sau khi parse dữ liệu.")

    df["feature_vec"] = df.apply(
        lambda row: np.concatenate([row["yolo_onehot_vec"], row["conf_score_vec"]]).astype(np.float32),
        axis=1,
    )

    X_all, y_all = [], []
    usable_groups, total_windows, total_sequences = 0, 0, 0
    grouped = list(df.groupby(group_cols, sort=False))

    for _, g in grouped:
        if frame_order_col in g.columns:
            g = g.sort_values(frame_order_col, kind="stable")

        frame_features = g["feature_vec"].tolist()
        frame_gt = g[label_col].tolist()
        frame_conf = g["conf_score_vec"].tolist()

        # Cần tối thiểu WINDOW_SIZE + SEQ_LEN - 1 frame để tạo 1 sample.
        if len(frame_features) < window_size + seq_len - 1:
            continue

        Xw, yw = build_windows(
            frame_features,
            frame_gt,
            frame_conf,
            window_size=window_size,
            stride=frame_stride,
        )
        Xs, ys = build_sequences_from_windows(Xw, yw, seq_len=seq_len, seq_stride=seq_stride)

        if len(Xs) == 0:
            continue

        usable_groups += 1
        total_windows += len(Xw)
        total_sequences += len(Xs)
        X_all.append(Xs)
        y_all.append(ys)

    if not X_all:
        X = np.empty((0, seq_len, window_size, 6), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
    else:
        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)

    info = {
        "file_path": file_path,
        "rows_before_filter": int(before_rows),
        "rows_after_filter": int(len(df)),
        "num_groups": int(len(grouped)),
        "usable_groups": int(usable_groups),
        "total_windows": int(total_windows),
        "total_sequences": int(total_sequences),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "label_distribution": summarize_labels(y),
    }
    return X, y, info


def compute_effective_class_weights(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = np.ones(num_classes, dtype=np.float64)

    for c in range(num_classes):
        if counts[c] > 0:
            effective_num = 1.0 - np.power(CLASS_WEIGHT_BETA, counts[c])
            weights[c] = (1.0 - CLASS_WEIGHT_BETA) / max(effective_num, 1e-12)
        else:
            weights[c] = 0.0

    positive = weights[weights > 0]
    if len(positive):
        weights = weights / positive.mean()

    weights = np.clip(weights, CLASS_WEIGHT_CLIP_MIN, CLASS_WEIGHT_CLIP_MAX)
    return weights.astype(np.float32)


def compute_sample_weights(y: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    return np.asarray([class_weights[int(label)] for label in y], dtype=np.float32)


def summarize_labels(y: np.ndarray) -> dict:
    counts = Counter(map(int, y.tolist())) if len(y) else Counter()
    return {CLASS_NAMES[k]: int(counts.get(k, 0)) for k in range(NUM_CLASSES)}


def print_distribution(name: str, y: np.ndarray) -> None:
    dist = summarize_labels(y)
    total = max(int(len(y)), 1)
    print(f"\n[INFO] {name} distribution:")
    for cls_name, count in dist.items():
        print(f"  - {cls_name:13s}: {count:8d} ({count / total * 100:6.2f}%)")


def create_and_save_lstm_data(train_path: str, val_path: str, test_path: str, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    print("[INFO] Processing train...")
    X_train, y_train, info_train = process_csv_to_lstm_data(train_path, group_cols=GROUP_COLS)

    print("[INFO] Processing val...")
    X_val, y_val, info_val = process_csv_to_lstm_data(val_path, group_cols=GROUP_COLS)

    print("[INFO] Processing test...")
    X_test, y_test, info_test = process_csv_to_lstm_data(test_path, group_cols=GROUP_COLS)

    if len(y_train) == 0:
        raise ValueError("y_train rỗng. Kiểm tra lại CSV, videoID, segment_id, Frame_number, check_video.")

    class_weights = compute_effective_class_weights(y_train, num_classes=NUM_CLASSES)
    sample_weights = compute_sample_weights(y_train, class_weights)

    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val)
    np.save(os.path.join(save_dir, "y_val.npy"), y_val)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    np.save(os.path.join(save_dir, "class_weights.npy"), class_weights)
    np.save(os.path.join(save_dir, "sample_weights.npy"), sample_weights)

    metadata = {
        "feature_definition": "[onehot_yolo(3) + conf_score_yolo(3)]",
        "feature_dim": 6,
        "ground_truth_source": GT_LABEL_COL,
        "group_cols": GROUP_COLS,
        "window_size": WINDOW_SIZE,
        "frame_stride": FRAME_STRIDE,
        "seq_len": SEQ_LEN,
        "seq_stride": SEQ_STRIDE,
        "conf_label_threshold": CONF_LABEL_THRES,
        "yolo_bonus_threshold": YOLO_BONUS_THRES,
        "yolo_bonus_score": YOLO_BONUS_SCORE,
        "window_label_rule": {
            "weapon": f"GT score: check_video=2 gives 1.0 point, YOLO conf_weapon >= {YOLO_BONUS_THRES} gives +{YOLO_BONUS_SCORE}; total must be >= {WEAPON_MIN_SCORE}",
            "violence": f"GT score: check_video=1 gives 1.0 point, YOLO conf_violence >= {YOLO_BONUS_THRES} gives +{YOLO_BONUS_SCORE}; total must be >= {VIOLENCE_MIN_SCORE}",
            "priority": ["weapon", "violence", "non_violence"],
        },
        "sample_label_rule": "label of last window in SEQ_LEN windows",
        "class_names": CLASS_NAMES,
        "train_info": info_train,
        "val_info": info_val,
        "test_info": info_test,
        "train_label_distribution": summarize_labels(y_train),
        "val_label_distribution": summarize_labels(y_val),
        "test_label_distribution": summarize_labels(y_test),
        "class_weights": class_weights.tolist(),
        "class_weight_method": "effective_number_normalized_clipped",
        "class_weight_beta": CLASS_WEIGHT_BETA,
    }

    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== DONE BUILD DATA ===")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")
    print_distribution("Train", y_train)
    print_distribution("Val", y_val)
    print_distribution("Test", y_test)
    print("[INFO] class_weights:", class_weights.tolist())
    print(f"[INFO] Saved to: {save_dir}")


if __name__ == "__main__":
    create_and_save_lstm_data(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        test_path=TEST_PATH,
        save_dir=SAVE_DIR,
    )
