
import os
import cv2
import csv
import json

import torch
from ultralytics import YOLO

# =========================
# 1. CONFIG
# =========================

MODEL_PATH = r"C:\NCKH_chuan\VERSION 14\train_LSTM5\models\6m.pt"
VIDEO_DIR = r"C:\NCKH_chuan\VERSION 14\train_LSTM5\videos"
OUT_ROOT = r"C:\NCKH_chuan\VERSION 14\train_LSTM5\outputs\csv"

os.makedirs(OUT_ROOT, exist_ok=True)

CONF_THRES = 0.6
IOU_THRES = 0.50
IMGSZ = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DETECTION_LABELS = ["violence", "weapon"]
ACTION_LABELS = ["non_violence", "violence", "weapon"]

LABEL_TO_INDEX = {label: idx for idx, label in enumerate(DETECTION_LABELS)}
ACTION_TO_INDEX = {label: idx for idx, label in enumerate(ACTION_LABELS)}

PRIORITY_ORDER = ["weapon", "violence"]
MAX_LABEL_SLOTS = 2  

# =========================
# 2. LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)
NAMES = model.names if hasattr(model, "names") else {}

print("=== YOLO MODEL LOADED ===")
print("Using device:", DEVICE)
print("Model names:", NAMES)


# =========================
# 3. UTILS
# =========================
def normalize_label(label):
    return str(label).strip().lower()


def safe_json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False)


def create_multi_hot(labels):
    """
    Multi-hot theo DETECTION_LABELS:
    violence -> [1,0]
    weapon   -> [0,1]
    cả 2     -> [1,1]
    """
    vec = [0] * len(DETECTION_LABELS)
    for label in labels:
        label = normalize_label(label)
        if label in LABEL_TO_INDEX:
            vec[LABEL_TO_INDEX[label]] = 1
    return vec


def create_one_hot(frame_label):
    """
    One-hot theo ACTION_LABELS:
    non_violence -> [1,0,0]
    violence     -> [0,1,0]
    weapon       -> [0,0,1]
    """
    vec = [0] * len(ACTION_LABELS)
    if frame_label in ACTION_TO_INDEX:
        vec[ACTION_TO_INDEX[frame_label]] = 1
    return vec


def get_unique_labels(labels):
    """Lấy unique labels theo thứ tự cố định."""
    unique = []
    for cls in DETECTION_LABELS:
        if cls in labels:
            unique.append(cls)
    return unique


def to_fixed_slots(items, max_len=MAX_LABEL_SLOTS, fill_value=None):
    """Đưa list về đúng số ô cố định, ví dụ [None, None]."""
    items = list(items[:max_len])
    while len(items) < max_len:
        items.append(fill_value)
    return items


def build_frame_conf_scores(detections):

    violence_conf = 0.0
    weapon_conf = 0.0

    for det in detections:
        cls = normalize_label(det["class"])
        conf = float(det["conf"])

        if cls == "violence":
            violence_conf = max(violence_conf, conf)
        elif cls == "weapon":
            weapon_conf = max(weapon_conf, conf)

    non_violence_conf = 1.0 if len(detections) == 0 else 0.0
    return [round(non_violence_conf, 4), round(violence_conf, 4), round(weapon_conf, 4)]


def choose_final_label(current_conf_vector, unique_labels):
    if not unique_labels:
        return "non_violence"

    score_map = {
        "violence": current_conf_vector[1],
        "weapon": current_conf_vector[2],
    }

    max_score = max(score_map[label] for label in unique_labels)
    best_labels = [label for label in unique_labels if abs(score_map[label] - max_score) < 1e-9]

    if len(best_labels) == 1:
        return best_labels[0]

    for label in PRIORITY_ORDER:
        if label in best_labels:
            return label

    return best_labels[0]


def detect_yolo_frame(frame):
    results = model.predict(
        frame,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        verbose=False
    )

    detections = []
    if len(results) == 0:
        return detections

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.int().detach().cpu().numpy()

    for i in range(len(confs)):
        conf = float(confs[i])
        cls_id = int(clss[i])
        class_name = normalize_label(NAMES.get(cls_id, str(cls_id)))

        if class_name not in DETECTION_LABELS:
            continue

        detections.append({
            "class": class_name,
            "class_id": cls_id,
            "conf": round(conf, 4)
        })

    return detections


# =========================
# 4. PROCESS VIDEO
# =========================
def process_video(video_path, writer):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n=== Processing: {video_name} ===")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        time_in_video = frame_num / fps

        # 1. Detect
        detections = detect_yolo_frame(frame)

        # 2. Gom nhãn
        labels_all = [det["class"] for det in detections]
        unique_labels = get_unique_labels(labels_all)
        num_objects = len(detections)

        # 3. Chuẩn hóa để output giống ảnh
        labels_slots = to_fixed_slots(unique_labels, MAX_LABEL_SLOTS, None)
        all_detections_slots = to_fixed_slots(labels_all, MAX_LABEL_SLOTS, None)

        # 4. conf-score hiện tại: [non_violence, violence, weapon]
        conf_score = build_frame_conf_scores(detections)

        # 5. Multi-hot theo detection labels [violence, weapon]
        multi_hot = create_multi_hot(unique_labels)

        # 6. Quyết định LabelFusion (không còn smooth-score)
        final_label = choose_final_label(conf_score, unique_labels)
        one_hot = create_one_hot(final_label)

        # 7. Ghi CSV
        row = [
            video_name,
            round(time_in_video, 2),
            frame_num,
            safe_json_dumps(labels_slots),
            num_objects,
            safe_json_dumps(all_detections_slots),
            safe_json_dumps(conf_score),
            safe_json_dumps(multi_hot),
            final_label,
            safe_json_dumps(one_hot),
        ]
        writer.writerow(row)

        if frame_num % 15 == 0:
            print(
                f"Processed {frame_num}/{total_frames} | "
                f"Labels: {unique_labels if unique_labels else ['none']} | "
                f"Final: {final_label}"
            )

    cap.release()
    print(f"Completed: {video_name}")


# =========================
# 5. MAIN
# =========================
def main():
    exts = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")

    try:
        video_list = [
            os.path.join(VIDEO_DIR, f)
            for f in sorted(os.listdir(VIDEO_DIR))
            if f.endswith(exts) and os.path.isfile(os.path.join(VIDEO_DIR, f))
        ]
    except FileNotFoundError:
        print(f"Directory not found: {VIDEO_DIR}")
        return

    if not video_list:
        print(f"No videos found in: {VIDEO_DIR}")
        return

    print(f"Found {len(video_list)} video(s).")

    output_csv = os.path.join(OUT_ROOT, "all_videos_yolo_no_smooth.csv")

    if os.path.exists(output_csv):
        os.remove(output_csv)

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "videoID",
            "TimeInVideo",
            "Frame_number",
            "Labels",
            "NumObjects",
            "All_detections",
            "conf-score",
            "Multi-hot",
            "LabelFusion",
            "One-hot"
        ])

        for idx, video_path in enumerate(video_list, 1):
            print(f"\n[{idx}/{len(video_list)}] {os.path.basename(video_path)}")
            try:
                process_video(video_path, writer)
            except Exception as e:
                print(f"Error while processing {video_path}: {e}")
                import traceback
                traceback.print_exc()

    print("\n=== DONE ===")
    print("Output CSV:", output_csv)


if __name__ == "__main__":
    main()
