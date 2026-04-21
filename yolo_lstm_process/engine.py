# -*- coding: utf-8 -*-
"""
test.py
Test YOLO + LSTM on video
Pipeline:
    frame -> YOLO track -> extract 12 features -> sequence -> LSTM -> overlay result
"""

import os
import django
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, deque
from ultralytics import YOLO
from datetime import datetime
 
 
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.conf import settings
from monitoring.models import Event

from .ai_config import (
    YOLO_MODEL_PATH,
    LSTM_MODEL_PATH,
    VIDEO_PATH,
    OUTPUT_PATH,
    CONF_THRES,
    IMGSZ,
    SEQ_LEN,
    THRESH_LSTM,
    DEVICE,
)

# =========================
# LSTM MODEL
# =========================
class ViolenceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 dropout=0.3, bidirectional=False):
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
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            last_hidden = torch.cat([forward_last, backward_last], dim=1)
        else:
            last_hidden = h_n[-1]

        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits


# =========================
# LOAD MODELS
# =========================
def load_models():
    print(f"[INFO] Using device: {DEVICE}")

    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy YOLO model: {YOLO_MODEL_PATH}")

    if not os.path.exists(LSTM_MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy LSTM model: {LSTM_MODEL_PATH}")

    # Load YOLO
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_names = yolo_model.names
    print("[INFO] YOLO class names:", yolo_names)

    # Load LSTM checkpoint
    ckpt = torch.load(LSTM_MODEL_PATH, map_location=DEVICE)

    lstm_model = ViolenceLSTM(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        num_classes=ckpt["num_classes"],
        dropout=ckpt.get("dropout", 0.3),
        bidirectional=ckpt.get("bidirectional", False),
    ).to(DEVICE)

    lstm_model.load_state_dict(ckpt["model_state_dict"])
    lstm_model.eval()

    lstm_class_names = ckpt.get(
        "class_names",
        ["non_violence", "violence", "weapon"]
    )
    print("[INFO] LSTM class names:", lstm_class_names)

    return yolo_model, yolo_names, lstm_model, lstm_class_names


# =========================
# FEATURE EXTRACTION
# =========================
def extract_frame_features(result, frame_w, frame_h, yolo_names, track_history):
    """
    Trích xuất 12 đặc trưng:
    [num_obj_violence, num_obj_weapon,
     conf_max_violence, conf_max_weapon,
     mean_conf_violence, mean_conf_weapon,
     area_max_violence, area_max_weapon,
     mean_area_violence, mean_area_weapon,
     mean_speed, max_speed]
    """
    class_to_idx = {
        "violence": 0,
        "Weapon": 1,
        "weapon": 1,
    }

    num_classes = 2

    num_obj = [0] * num_classes
    conf_sum = [0.0] * num_classes
    conf_max = [0.0] * num_classes
    area_sum = [0.0] * num_classes
    area_max = [0.0] * num_classes
    speeds = []

    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros(12, dtype=np.float32)

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy()

    track_ids = None
    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        conf = float(confs[i])
        cls_id = int(clss[i])

        class_name = yolo_names.get(cls_id, str(cls_id))
        if class_name not in class_to_idx:
            continue

        idx = class_to_idx[class_name]

        num_obj[idx] += 1
        conf_sum[idx] += conf
        conf_max[idx] = max(conf_max[idx], conf)

        area = ((x2 - x1) * (y2 - y1)) / max(frame_w * frame_h, 1)
        area_sum[idx] += area
        area_max[idx] = max(area_max[idx], area)

        if track_ids is not None:
            tid = int(track_ids[i])
            cx = (x1 + x2) / 2.0 / frame_w
            cy = (y1 + y2) / 2.0 / frame_h

            track_history[tid].append((cx, cy))

            if len(track_history[tid]) >= 2:
                px, py = track_history[tid][-2]
                speed = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                speeds.append(speed)

    mean_conf = [
        conf_sum[i] / num_obj[i] if num_obj[i] > 0 else 0.0
        for i in range(num_classes)
    ]
    mean_area = [
        area_sum[i] / num_obj[i] if num_obj[i] > 0 else 0.0
        for i in range(num_classes)
    ]

    mean_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    max_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0

    feature = [
        num_obj[0], num_obj[1],
        conf_max[0], conf_max[1],
        mean_conf[0], mean_conf[1],
        area_max[0], area_max[1],
        mean_area[0], mean_area[1],
        mean_speed, max_speed
    ]
    return np.array(feature, dtype=np.float32)


# =========================
# DRAWING
# =========================
def draw_yolo_boxes(frame, result, yolo_names, lstm_label):
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    lstm_label = str(lstm_label).lower()

    if lstm_label not in ["violence", "weapon"]:
        return frame

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        conf = float(confs[i])
        cls_id = int(clss[i])

        yolo_label = str(yolo_names.get(cls_id, str(cls_id)))
        yolo_label_lower = yolo_label.lower()

        if lstm_label == "violence" and "violence" not in yolo_label_lower:
            continue
        if lstm_label == "weapon" and "weapon" not in yolo_label_lower:
            continue

        color = (0, 0, 255)
        text = f"{yolo_label} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame


def draw_lstm_status(frame, lstm_label, lstm_conf, fps_value, frame_idx):
    alarm = str(lstm_label).lower() in ["violence", "weapon"]

    status_color = (0, 0, 255) if alarm else (255, 0, 0)
    fixed_color = (255, 0, 0)

    text1 = f"LSTM Prediction: {lstm_label}"
    text2 = f"Confidence: {lstm_conf:.3f}"
    text3 = f"FPS: {fps_value:.2f}"
    text4 = f"Frame: {frame_idx}"

    cv2.putText(frame, text1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    cv2.putText(frame, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, text3, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fixed_color, 2)
    cv2.putText(frame, text4, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fixed_color, 2)

    return frame


# =========================
# PROCESS VIDEO
# =========================
def process_video(
    input_path,
    output_dir=None,
    show_window=False,   
    save_video=True
):
    os.makedirs(output_dir or OUTPUT_PATH, exist_ok=True)

    yolo_model, yolo_names, lstm_model, lstm_class_names = load_models()
    track_history = defaultdict(list)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    video_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = output_dir or OUTPUT_PATH
    save_path = os.path.join(output_dir, f"{video_name}_lstm_output.mp4")

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            print("[WARN] avc1 không hoạt động, chuyển sang mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError("Không thể tạo video output.")

    seq_buffer = deque(maxlen=SEQ_LEN)
    last_lstm_label = "violence"
    last_lstm_conf = 1.0

    frame_idx = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = yolo_model.track(
            frame,
            persist=True,
            conf=CONF_THRES,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False
        )
        result = results[0]

        feat = extract_frame_features(result, width, height, yolo_names, track_history)
        seq_buffer.append(feat)

        if len(seq_buffer) == SEQ_LEN:
            seq_np = np.array(seq_buffer, dtype=np.float32)
            seq_tensor = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = lstm_model(seq_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_label = lstm_class_names[pred_idx]
            pred_conf = float(probs[pred_idx])

            if pred_conf >= THRESH_LSTM:
                last_lstm_label = pred_label
                last_lstm_conf = pred_conf
            else:
                last_lstm_label = "non_violence"
                last_lstm_conf = pred_conf
        else:
            last_lstm_label = "waiting..."
            last_lstm_conf = 0.0

        now = time.time()
        fps_value = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame = draw_yolo_boxes(frame, result, yolo_names, last_lstm_label)
        frame = draw_lstm_status(frame, last_lstm_label, last_lstm_conf, fps_value, frame_idx)

        if show_window:
            cv2.imshow("YOLO + LSTM Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    return {
        "output_path": save_path if save_video else None,
        "final_label": last_lstm_label,
        "final_conf": last_lstm_conf,
        "total_frames": frame_idx,
    }


def generate_processed_frames(input_path):
    yolo_model, yolo_names, lstm_model, lstm_class_names = load_models()
    track_history = defaultdict(list)

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # buffer 3 giây trước
    pre_buffer = deque(maxlen=int(fps * 3))
    state = {
        "recording": False,
        "event_type": None,
        "confidence": 0,
        "frames": [],
        "post_frames": 0,

        "continuous_start": {
            "violence": None,
            "weapon": None
        },

        "last_trigger": {
            "violence": 0,
            "weapon": 0
        }
    }
    seq_buffer = deque(maxlen=SEQ_LEN)
    last_lstm_label = "waiting..."
    last_lstm_conf = 0.0

    frame_idx = 0
    prev_time = time.time()
    COOLDOWN = 60
    CONTINUOUS = 2

    def can_trigger(label, conf, now):
        if label not in ["violence", "weapon"]:
            return False

        if now - state["last_trigger"][label] < COOLDOWN:
            return False

        if conf >= 1.0:
            return True

        start = state["continuous_start"][label]

        if start is None:
            state["continuous_start"][label] = now
            return False

        if now - start >= CONTINUOUS:
            return True

        return False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pre_buffer.append(frame.copy())
            now = time.time()

            frame_idx += 1

            results = yolo_model.track(
                frame,
                persist=True,
                conf=CONF_THRES,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False
            )
            result = results[0]

            feat = extract_frame_features(result, width, height, yolo_names, track_history)
            seq_buffer.append(feat)

            if len(seq_buffer) == SEQ_LEN:
                seq_np = np.array(seq_buffer, dtype=np.float32)
                seq_tensor = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = lstm_model(seq_tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                pred_idx = int(np.argmax(probs))
                pred_label = lstm_class_names[pred_idx]
                pred_conf = float(probs[pred_idx])

                if pred_conf >= THRESH_LSTM:
                    last_lstm_label = pred_label
                    last_lstm_conf = pred_conf
                else:
                    last_lstm_label = "non_violence"
                    last_lstm_conf = pred_conf
            else:
                last_lstm_label = "waiting..."
                last_lstm_conf = 0.0

            now = time.time()
            fps_value = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            label = last_lstm_label
            conf = last_lstm_conf

            # reset nếu không phải event
            if label not in ["violence", "weapon"]:
                state["continuous_start"]["violence"] = None
                state["continuous_start"]["weapon"] = None

            # trigger event
            if not state["recording"]:
                if can_trigger(label, conf, now):

                    print("🔥 EVENT TRIGGER:", label, conf)

                    state["recording"] = True
                    state["event_type"] = label
                    state["confidence"] = conf

                    # lấy 3s trước
                    state["frames"] = list(pre_buffer)

                    # 5s sau
                    state["post_frames"] = int(fps * 5)

                    state["last_trigger"][label] = now

                    state["continuous_start"]["violence"] = None
                    state["continuous_start"]["weapon"] = None

            # đang ghi clip
            if state["recording"]:
                state["frames"].append(frame.copy())
                state["post_frames"] -= 1

                if state["post_frames"] <= 0:

                    print("💾 SAVE EVENT:", state["event_type"])

                    save_event_clip(
                        frames=state["frames"],
                        fps=fps,
                        width=width,
                        height=height,
                        label=state["event_type"],
                        conf=state["confidence"]
                    )

                    state["recording"] = False
                    state["frames"] = []
            frame = draw_yolo_boxes(frame, result, yolo_names, last_lstm_label)
            frame = draw_lstm_status(frame, last_lstm_label, last_lstm_conf, fps_value, frame_idx)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            jpg_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )
    finally:
        cap.release()


def save_event_clip(frames, fps, width, height, label, conf):
    clip_folder = os.path.join(settings.MEDIA_ROOT, "alerts", "clips")
    image_folder = os.path.join(settings.MEDIA_ROOT, "alerts", "images")
    os.makedirs(clip_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    clip_filename = f"{label}_{ts}.mp4"
    clip_path = os.path.join(clip_folder, clip_filename)

    image_filename = f"{label}_{ts}.jpg"
    image_path = os.path.join(image_folder, image_filename)

    # ưu tiên H264 cho browser
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[WARN] avc1 không hoạt động, chuyển sang mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError("Không thể tạo clip sự kiện")

    for f in frames:
        writer.write(f)

    writer.release()

    # lưu ảnh đại diện: lấy frame giữa clip
    if len(frames) > 0:
        mid_idx = len(frames) // 2
        cv2.imwrite(image_path, frames[mid_idx])

    Event.objects.create(
        event_type=label,
        confidence=conf,
        clip=f"alerts/clips/{clip_filename}",
        image=f"alerts/images/{image_filename}"
    ) 
# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Không tìm thấy video test: {VIDEO_PATH}")
        return

    result = process_video(
        input_path=VIDEO_PATH,
        output_dir=OUTPUT_PATH,
        show_window=True,
        save_video=True
    )

    print("✅ Done.")
    print("[RESULT]", result)


if __name__ == "__main__":
    main()