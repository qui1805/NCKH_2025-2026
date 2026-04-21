import cv2
import time
import torch
import numpy as np
from collections import deque, defaultdict

from ultralytics import YOLO

from config import *
from model_loader import ViolenceLSTM  # :contentReference[oaicite:0]{index=0}
from feature_extractor import extract_frame_features  # :contentReference[oaicite:1]{index=1}
from draw_utils import draw_yolo_boxes, draw_lstm_status  # :contentReference[oaicite:2]{index=2}
from clip_buffer import EventClipBuffer  # :contentReference[oaicite:3]{index=3}
from alert_service import send_gmail_alert  # :contentReference[oaicite:4]{index=4}
from video_utils import open_video  # :contentReference[oaicite:5]{index=5}


# ===== TRACK HISTORY (CHO SPEED) =====
track_history = defaultdict(lambda: deque(maxlen=10))


def load_models():
    # ===== YOLO =====
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.to(DEVICE)

    yolo_names = yolo_model.names

    # ===== LSTM =====
    ckpt = torch.load(LSTM_MODEL_PATH, map_location=DEVICE)

    input_size = ckpt.get("input_size", 12)
    hidden_size = ckpt.get("hidden_size", 64)
    num_layers = ckpt.get("num_layers", 2)
    num_classes = ckpt.get("num_classes", 3)
    bidirectional = ckpt.get("bidirectional", False)

    lstm_model = ViolenceLSTM(
        input_size, hidden_size, num_layers,
        num_classes, bidirectional=bidirectional
    ).to(DEVICE)

    lstm_model.load_state_dict(ckpt["model_state_dict"])
    lstm_model.eval()

    class_names = ckpt.get(
        "class_names",
        ["non_violence", "violence", "weapon"]
    )

    return yolo_model, yolo_names, lstm_model, class_names


def main_detect():
    global track_history
    track_history = defaultdict(list)

    print("🚀 START DETECT")
    print("Device:", DEVICE)

    # ===== LOAD =====
    yolo_model, yolo_names, lstm_model, lstm_class_names = load_models()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    # ===== BUFFER =====
    seq_buffer = deque(maxlen=SEQ_LEN)

    # ===== ALERT BUFFER =====
    from clip_buffer import EventClipBuffer
    from alert_service import send_gmail_alert

    clip_buffer = EventClipBuffer(fps)

    frame_idx = 0
    prev_time = time.time()

    # ===== THRESH =====
    THRESH_LSTM = 0.9   

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ===== YOLO TRACK =====
        results = yolo_model.track(
            frame,
            persist=True,
            conf=CONF_THRES,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        # ===== FEATURE =====
        feat = extract_frame_features(result, width, height, yolo_names)
        seq_buffer.append(feat)

        lstm_label = "waiting..."
        lstm_conf = 0.0

        # ===== LSTM =====
        if len(seq_buffer) == SEQ_LEN:
            seq_np = np.array(seq_buffer, dtype=np.float32)
            seq_tensor = torch.tensor(seq_np).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = lstm_model(seq_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx = int(np.argmax(probs))
            lstm_label = lstm_class_names[pred_idx]
            lstm_conf = float(probs[pred_idx])

            # ===== IN RA TERMINAL =====
            print(f"[SEQ_Frame {frame_idx}]"  f"LSTM Prediction: {lstm_label}"  f"Confidence: {lstm_conf:.4f}"  f"Probs: {probs}")

        # ===== ALERT =====
        clip_buffer.update(frame)

        if lstm_label in ["violence", "weapon"] and lstm_conf > THRESH_LSTM:
            if clip_buffer.can_trigger():
                print("⚠️ TRIGGER EVENT")
                clip_buffer.start_event(frame, lstm_label, lstm_conf)

        if clip_buffer.is_done():
            payload = clip_buffer.finalize_event()

            if payload:
                print("📦 EVENT SAVED:", payload)
                send_gmail_alert(payload)

        # ===== FPS =====
        now = time.time()
        fps_value = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # ===== DRAW =====
        frame = draw_yolo_boxes(frame, result, yolo_names, lstm_label)
        frame = draw_lstm_status(frame, lstm_label, lstm_conf, fps_value, frame_idx)

        cv2.imshow("YOLO + LSTM DETECT", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        
if __name__ == "__main__":
    main_detect()
