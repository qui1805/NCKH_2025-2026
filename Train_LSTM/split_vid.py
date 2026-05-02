

import os
import ast
import re
import random
import numpy as np
import pandas as pd

# =========================
# PATH CONFIG
# =========================
INPUT_CSV = r"C:\Train_LSTM\output\csv\input_yolo_detect.csv"
OUTPUT_DIR = r"C:\Train_LSTM\inputs"

# =========================
# SPLIT CONFIG
# =========================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.18
TEST_RATIO = 0.12

RANDOM_SEED = 42
N_TRIALS = 5000

# =========================
# COLUMN CONFIG
# =========================
VIDEO_COL = "videoID"
FRAME_COL = "Frame_number"
LABEL_COL = "check_video"
ONEHOT_COL = "One-hot"
CONF_COL = "conf-score"

# =========================
# LSTM DATA CONFIG
# =========================
SEGMENT_SIZE = 500

WINDOW_SIZE = 32
SEQ_LEN = 16
NUM_CLASSES = 3

CLASS_NAMES = ["non_violence", "violence", "weapon"]

VIOLENCE_MIN_SCORE = 28
WEAPON_MIN_SCORE = 22
YOLO_BONUS_THRES = 0.30
YOLO_BONUS_SCORE = 0.20

SAMPLE_LABEL_MODE = "last"

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO phải bằng 1.0")

if VIOLENCE_MIN_SCORE > WINDOW_SIZE * (1.0 + YOLO_BONUS_SCORE):
    raise ValueError("VIOLENCE_MIN_SCORE đang quá lớn so với WINDOW_SIZE và YOLO_BONUS_SCORE.")

if WEAPON_MIN_SCORE > WINDOW_SIZE * (1.0 + YOLO_BONUS_SCORE):
    raise ValueError("WEAPON_MIN_SCORE đang quá lớn so với WINDOW_SIZE và YOLO_BONUS_SCORE.")


# =========================
# UTILS
# =========================
def parse_vector_cell(x):
    """Parse ô CSV dạng '[1,0,0]' hoặc '1 0 0' thành np.ndarray."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if pd.isna(x):
        return np.asarray([], dtype=np.float32)
    if not isinstance(x, str):
        return np.asarray([], dtype=np.float32)

    s = x.strip()

    try:
        return np.asarray(ast.literal_eval(s), dtype=np.float32)
    except Exception:
        pass

    s = s.strip("[]").replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return np.asarray([], dtype=np.float32)

    return np.fromstring(s, sep=" ").astype(np.float32)


def decide_window_label(frame_gt_labels, frame_conf_scores):

    violence_score = 0.0
    weapon_score = 0.0

    for gt, conf_vec in zip(frame_gt_labels, frame_conf_scores):
        gt = int(gt)
        conf_vec = np.asarray(conf_vec, dtype=np.float32)

        if len(conf_vec) != NUM_CLASSES:
            continue

        # check_video là nhãn thật, được tính chính.
        if gt == 1:
            violence_score += 1.0

            # YOLO chỉ đóng vai trò điểm phụ để phản ánh tín hiệu detection.
            if float(conf_vec[1]) >= YOLO_BONUS_THRES:
                violence_score += YOLO_BONUS_SCORE

        elif gt == 2:
            weapon_score += 1.0

            if float(conf_vec[2]) >= YOLO_BONUS_THRES:
                weapon_score += YOLO_BONUS_SCORE

    if weapon_score >= WEAPON_MIN_SCORE:
        return 2

    if violence_score >= VIOLENCE_MIN_SCORE:
        return 1

    return 0


def decide_sequence_label(seq_window_labels):
   
    seq_window_labels = np.asarray(seq_window_labels, dtype=np.int64)

    if SAMPLE_LABEL_MODE == "last":
        return int(seq_window_labels[-1])

    if SAMPLE_LABEL_MODE == "majority":
        counts = np.bincount(seq_window_labels, minlength=NUM_CLASSES)
        max_count = counts.max()

        if counts[2] == max_count:
            return 2
        if counts[1] == max_count:
            return 1
        return 0

    raise ValueError("SAMPLE_LABEL_MODE chỉ được là 'last' hoặc 'majority'.")


def get_segment_sequence_distribution(g):
   
    if FRAME_COL in g.columns:
        g = g.sort_values(FRAME_COL, kind="stable")

    labels = g[LABEL_COL].tolist()
    confs = g["conf_vec"].tolist()

    n = len(g)

    # Cần tối thiểu WINDOW_SIZE + SEQ_LEN - 1 frame để tạo 1 sample.
    if n < WINDOW_SIZE + SEQ_LEN - 1:
        return {0: 0, 1: 0, 2: 0}, 0

    window_labels = []

    for i in range(0, n - WINDOW_SIZE + 1):
        labs = labels[i:i + WINDOW_SIZE]
        cf = confs[i:i + WINDOW_SIZE]
        window_labels.append(decide_window_label(labs, cf))

    seq_labels = []

    for i in range(0, len(window_labels) - SEQ_LEN + 1):
        seq_window_labels = window_labels[i:i + SEQ_LEN]
        seq_labels.append(decide_sequence_label(seq_window_labels))

    counts = {
        0: seq_labels.count(0),
        1: seq_labels.count(1),
        2: seq_labels.count(2),
    }

    return counts, len(seq_labels)


def print_sample_dist(name, segment_rows):
    total = sum(v["seq_total"] for v in segment_rows)

    print(f"\n[LSTM SAMPLE] {name}")
    print(f"Samples: {total}")

    for c, cls in enumerate(CLASS_NAMES):
        count = sum(v[f"class_{c}"] for v in segment_rows)
        percent = count / max(total, 1) * 100
        print(f"  - {cls:13s}: {count:8d} ({percent:6.2f}%)")


def print_frame_dist(name, split_df):
    total = len(split_df)
    counts = split_df[LABEL_COL].value_counts().to_dict()

    print(f"\n[FRAME] {name}")
    print(f"Rows: {total}")

    for c, cls in enumerate(CLASS_NAMES):
        count = counts.get(c, 0)
        percent = count / max(total, 1) * 100
        print(f"  - {cls:13s}: {count:8d} ({percent:6.2f}%)")


def evaluate_split(splits, target_seq, target_class):
   
    score = 0.0

    for split_name, items in splits.items():
        seq_total = sum(v["seq_total"] for v in items)

        # Cân bằng tổng sample.
        score += 2.0 * abs(seq_total - target_seq[split_name]) / max(target_seq[split_name], 1)

        # Cân bằng từng class.
        for c in [0, 1, 2]:
            count_c = sum(v[f"class_{c}"] for v in items)
            target_c = target_class[split_name][c]

            if c == 1:
                weight = 4.0   # violence quan trọng
            elif c == 2:
                weight = 3.0   # weapon quan trọng
            else:
                weight = 1.0

            score += weight * abs(count_c - target_c) / max(target_c, 1)

    return score


def assign_split_by_quota(items, target_seq):
  
    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    seq_count = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for item in items:
        need_ratio = {}

        for s in ["train", "val", "test"]:
            need_ratio[s] = (target_seq[s] - seq_count[s]) / max(target_seq[s], 1)

        candidates = sorted(
            ["train", "val", "test"],
            key=lambda s: need_ratio[s],
            reverse=True,
        )

        chosen = candidates[0]

        # Nếu split chưa vượt 105% target thì ưu tiên chọn split đó.
        for s in candidates:
            if seq_count[s] + item["seq_total"] <= target_seq[s] * 1.05:
                chosen = s
                break

        splits[chosen].append(item)
        seq_count[chosen] += item["seq_total"]

    return splits


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================
    # LOAD CSV
    # =========================
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    required_cols = [VIDEO_COL, FRAME_COL, LABEL_COL, ONEHOT_COL, CONF_COL]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc: {col}")

    df = df.dropna(subset=[VIDEO_COL, FRAME_COL, LABEL_COL, ONEHOT_COL, CONF_COL]).copy()

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df[FRAME_COL] = pd.to_numeric(df[FRAME_COL], errors="coerce")

    df = df.dropna(subset=[LABEL_COL, FRAME_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df[FRAME_COL] = df[FRAME_COL].astype(int)

    df = df[df[LABEL_COL].isin([0, 1, 2])].copy()

    df["onehot_vec"] = df[ONEHOT_COL].apply(parse_vector_cell)
    df["conf_vec"] = df[CONF_COL].apply(parse_vector_cell)

    df = df[df["onehot_vec"].map(lambda x: len(x) == NUM_CLASSES)].copy()
    df = df[df["conf_vec"].map(lambda x: len(x) == NUM_CLASSES)].copy()

    if df.empty:
        raise ValueError("CSV không còn dòng hợp lệ sau khi parse One-hot, conf-score và check_video.")

    # =========================
    # CREATE SEGMENT ID
    # =========================
    df = df.sort_values([VIDEO_COL, FRAME_COL], kind="stable").copy()

    # segment theo thứ tự frame trong từng video, không phụ thuộc Frame_number có bị nhảy số hay không.
    df["frame_index_in_video"] = df.groupby(VIDEO_COL).cumcount()
    df["segment_id"] = df["frame_index_in_video"] // SEGMENT_SIZE

    df["segment_key"] = (
        df[VIDEO_COL].astype(str)
        + "_seg_"
        + df["segment_id"].astype(str)
    )

    # =========================
    # BUILD SEGMENT PROFILE
    # =========================
    segment_profiles = []

    for seg_key, g in df.groupby("segment_key", sort=False):
        counts, seq_total = get_segment_sequence_distribution(g)

        if seq_total <= 0:
            continue

        video_id = g[VIDEO_COL].iloc[0]
        segment_id = int(g["segment_id"].iloc[0])

        segment_profiles.append({
            "segment_key": seg_key,
            "videoID": video_id,
            "segment_id": segment_id,
            "seq_total": int(seq_total),
            "class_0": int(counts[0]),
            "class_1": int(counts[1]),
            "class_2": int(counts[2]),
            "danger": int(counts[1] + counts[2]),
        })

    if len(segment_profiles) == 0:
        raise ValueError(
            "Không có segment nào đủ dài để tạo sequence LSTM. "
            "Hãy kiểm tra SEGMENT_SIZE, WINDOW_SIZE, SEQ_LEN hoặc dữ liệu video."
        )

    # =========================
    # TARGET DISTRIBUTION
    # =========================
    global_seq_total = sum(v["seq_total"] for v in segment_profiles)

    global_class_total = {
        c: sum(v[f"class_{c}"] for v in segment_profiles)
        for c in [0, 1, 2]
    }

    target_ratio = {
        "train": TRAIN_RATIO,
        "val": VAL_RATIO,
        "test": TEST_RATIO,
    }

    target_seq = {
        split: global_seq_total * ratio
        for split, ratio in target_ratio.items()
    }

    target_class = {
        split: {
            c: global_class_total[c] * ratio
            for c in [0, 1, 2]
        }
        for split, ratio in target_ratio.items()
    }

    # =========================
    # RANDOM SEARCH SPLIT
    # =========================
    best_score = float("inf")
    best_splits = None

    random.seed(RANDOM_SEED)

    for _ in range(N_TRIALS):
        items = segment_profiles[:]

        danger_items = [x for x in items if x["danger"] > 0]
        normal_items = [x for x in items if x["danger"] == 0]

        # Mỗi trial shuffle khác nhau để tìm split tốt hơn.
        random.shuffle(danger_items)
        random.shuffle(normal_items)

        # Đưa segment có nhãn nguy hiểm vào trước để dễ cân bằng lớp hiếm.
        danger_items = sorted(
            danger_items,
            key=lambda x: (x["danger"], x["seq_total"]),
            reverse=True,
        )

        items = danger_items + normal_items

        splits = assign_split_by_quota(items, target_seq)
        score = evaluate_split(splits, target_seq, target_class)

        if score < best_score:
            best_score = score
            best_splits = splits

    if best_splits is None:
        raise RuntimeError("Không tìm được split phù hợp.")

    print(f"[INFO] Best split score: {best_score:.6f}")

    # =========================
    # EXPORT CSV
    # =========================
    train_keys = [v["segment_key"] for v in best_splits["train"]]
    val_keys = [v["segment_key"] for v in best_splits["val"]]
    test_keys = [v["segment_key"] for v in best_splits["test"]]

    train_df = df[df["segment_key"].isin(train_keys)].copy()
    val_df = df[df["segment_key"].isin(val_keys)].copy()
    test_df = df[df["segment_key"].isin(test_keys)].copy()

    # Xóa cột phụ. Giữ lại segment_id vì file build/feet cần GROUP_COLS = ["videoID", "segment_id"].
    drop_cols = ["onehot_vec", "conf_vec", "frame_index_in_video", "segment_key"]

    for d in [train_df, val_df, test_df]:
        for col in drop_cols:
            if col in d.columns:
                d.drop(columns=[col], inplace=True)

    sort_cols = [VIDEO_COL, "segment_id", FRAME_COL]

    train_df = train_df.sort_values(sort_cols, kind="stable")
    val_df = val_df.sort_values(sort_cols, kind="stable")
    test_df = test_df.sort_values(sort_cols, kind="stable")

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    val_path = os.path.join(OUTPUT_DIR, "val.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    # Lưu thêm profile để kiểm tra lại split.
    profile_df = pd.DataFrame(segment_profiles)
    profile_path = os.path.join(OUTPUT_DIR, "segment_profiles.csv")
    profile_df.to_csv(profile_path, index=False, encoding="utf-8-sig")

    # =========================
    # REPORT
    # =========================
    print("\n=== SEGMENT-AWARE SPLIT DONE ===")
    print(f"Segment size      : {SEGMENT_SIZE} frames")
    print(f"Window size       : {WINDOW_SIZE}")
    print(f"Sequence length   : {SEQ_LEN}")
    print(f"Sample label mode : {SAMPLE_LABEL_MODE}")
    print(f"Tổng segment dùng : {len(segment_profiles)}")
    print(f"Tổng LSTM samples : {global_seq_total}")

    print("\n[GLOBAL LSTM SAMPLE]")
    for c, cls in enumerate(CLASS_NAMES):
        count = global_class_total[c]
        percent = count / max(global_seq_total, 1) * 100
        print(f"  - {cls:13s}: {count:8d} ({percent:6.2f}%)")

    print_sample_dist("TRAIN", best_splits["train"])
    print_sample_dist("VAL", best_splits["val"])
    print_sample_dist("TEST", best_splits["test"])

    print_frame_dist("TRAIN", train_df)
    print_frame_dist("VAL", val_df)
    print_frame_dist("TEST", test_df)

    print("\nSaved files:")
    print(train_path)
    print(val_path)
    print(test_path)
    print(profile_path)


if __name__ == "__main__":
    main()
