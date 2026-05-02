
import os
import cv2
import time
import numpy as np
import torch
import threading
import csv

from django.db import close_old_connections
from monitoring.models import Event
from .alert_service import send_event_telegram
from collections import deque
from datetime import datetime

from .model_loader import load_models
from .features import extract_frame_features
from .event_service import save_event_clip, create_event
from .drawing import draw_yolo_boxes, draw_lstm_status
from .statistics import build_action_statistics_from_timeline, save_lstm_statistics_csv
from .ai_config import (
    WINDOW_SIZE,
    SEQ_LEN,
    CONF_THRES,
    IMGSZ,
    DEVICE,
    FRAME_FEATURE_DIM,
    OUTPUT_PATH,
    VIOLENCE_THRES,
    WEAPON_THRES,
)


# =========================================================
# OPEN SOURCE
# =========================================================
def open_video_or_camera(source, is_camera=False):
   
    if is_camera:
        try:
            camera_index = int(source)
        except Exception:
            camera_index = 0

        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 25)

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)

    else:
        video_path = str(source)

        if not os.path.exists(video_path):
            raise RuntimeError(f"Video không tồn tại: {video_path}")

        # Video upload/file tuyệt đối không dùng CAP_DSHOW.
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn: {source}")

    return cap


# =========================================================
# MODEL / LSTM
# =========================================================
def _get_model_input_dim(lstm_model):
   
    try:
        return int(lstm_model.lstm.input_size)
    except Exception:
        return int(WINDOW_SIZE * FRAME_FEATURE_DIM)


def _predict_lstm(window_buffer, lstm_model, input_dim):

    seq_np = np.asarray(window_buffer, dtype=np.float32)

    if seq_np.shape[0] != SEQ_LEN:
        return "waiting...", 0.0, None

    seq_np = seq_np.reshape(1, SEQ_LEN, -1)

    if seq_np.shape[2] != int(input_dim):
        raise RuntimeError(
            f"Sai input_dim LSTM: web tạo {seq_np.shape[2]}, nhưng model cần {input_dim}. "
            f"Kiểm tra WINDOW_SIZE={WINDOW_SIZE}, FRAME_FEATURE_DIM={FRAME_FEATURE_DIM}."
        )

    x = torch.tensor(seq_np, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        logits = lstm_model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    p_non, p_vio, p_wea = map(float, probs[:3])

    violence_ok = p_vio >= VIOLENCE_THRES
    weapon_ok = p_wea >= WEAPON_THRES

    if weapon_ok and (not violence_ok or p_wea >= p_vio):
        return "weapon", p_wea, probs


    if violence_ok:
        return "violence", p_vio, probs


    return "non_violence", p_non, probs

# =========================================================
# YOLO / FEATURE
# =========================================================
def _run_yolo(yolo_model, frame):
  
    if FRAME_FEATURE_DIM == 6:
        return yolo_model(
            frame,
            conf=CONF_THRES,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False,
        )[0]

    return yolo_model.track(
        frame,
        persist=True,
        conf=CONF_THRES,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
    )[0]


def _extract_feature(result, width, height, yolo_names):
    feat = extract_frame_features(
        result=result,
        width=width,
        height=height,
        yolo_names=yolo_names,
    )

    feat = np.asarray(feat, dtype=np.float32)

    if feat.shape[0] != FRAME_FEATURE_DIM:
        raise RuntimeError(
            f"Feature sai kích thước. Model LSTM cần {FRAME_FEATURE_DIM} feature/frame, "
            f"nhưng đang nhận {feat.shape[0]}."
        )

    return feat


def _safe_extract_feature(result, width, height, yolo_names):
  
    try:
        return _extract_feature(result, width, height, yolo_names)
    except TypeError:
        feat = extract_frame_features(
            result=result,
            frame_width=width,
            frame_height=height,
            yolo_names=yolo_names,
        )
        feat = np.asarray(feat, dtype=np.float32)

        if feat.shape[0] != FRAME_FEATURE_DIM:
            raise RuntimeError(
                f"Feature sai kích thước. Model LSTM cần {FRAME_FEATURE_DIM} feature/frame, "
                f"nhưng đang nhận {feat.shape[0]}."
            )

        return feat


# =========================================================
# DRAW
# =========================================================
def _draw_lstm(frame, label, conf, fps_value, frame_idx):
   
    try:
        return draw_lstm_status(
            frame=frame,
            label=label,
            conf=conf,
            fps_value=fps_value,
            frame_idx=frame_idx,
        )
    except TypeError:
        return draw_lstm_status(frame, label, conf, fps_value, frame_idx)


def _draw_yolo(frame, result, yolo_names, label):
    try:
        return draw_yolo_boxes(frame, result, yolo_names, label)
    except TypeError:
        return draw_yolo_boxes(frame, result, yolo_names, label=label)


# =========================================================
# MAIN GENERATOR
# =========================================================

def _get_yolo_detect_text(result, yolo_names):
    if result.boxes is None or len(result.boxes) == 0:
        return "none"

    best_label = "none"
    best_conf = 0.0

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRES:
            continue

        cls_id = int(box.cls[0])

        if isinstance(yolo_names, dict):
            label = str(yolo_names.get(cls_id, f"class_{cls_id}"))
        else:
            label = str(yolo_names[int(cls_id)])

        if conf > best_conf:
            best_conf = conf
            best_label = label

    if best_label == "none":
        return "none"

    return f"{best_label}:{best_conf:.2f}"

def _generate_processed_frames(source, is_camera=False):
    yolo_model, lstm_model = load_models()

    yolo_names = getattr(yolo_model, "names", {})
    input_dim = _get_model_input_dim(lstm_model)

    cap = open_video_or_camera(source, is_camera=is_camera)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        width, height = 640, 480

    pre_buffer = deque(maxlen=max(1, int(fps * 3)))
    frame_buffer = deque(maxlen=WINDOW_SIZE)
    window_buffer = deque(maxlen=SEQ_LEN)

    state = {
        "recording": False,
        "event_type": None,
        "confidence": 0.0,
        "frames": [],
        "post_frames": 0,
        "event_obj": None,
        "continuous_start": {
            "violence": None,
            "weapon": None,
        },
        "last_trigger": 0.0,
    }

    last_lstm_label = "waiting..."
    last_lstm_conf = 0.0
    last_lstm_probs = None
    eval_stt = 0

    frame_idx = 0
    prev_time = time.perf_counter()
    fps_display = 0.0

    timeline_labels = []
    process_start_datetime = datetime.now()
    source_name = "camera_live" if is_camera else os.path.splitext(os.path.basename(str(source)))[0]

    eval_csv_path = os.path.join(OUTPUT_PATH, f"{source_name}_lstm_eval.csv")
    eval_csv_file = open(eval_csv_path, mode="w", newline="", encoding="utf-8-sig")
    eval_csv_writer = csv.writer(eval_csv_file)

    eval_csv_writer.writerow([
        "stt",
        "frame",
        "time_in_video_sec",
        "yolo_detect",
        "lstm_label",
        "lstm_conf",
        "p_non_violence",
        "p_violence",
        "p_weapon",
        "raw_lstm_label",
        "system_time",
    ])
    cooldown = 40
    continuous_violence = 2
    continuous_weapon = 0.7
    instant_alert_conf = 0.90

    def can_trigger(label, conf, now_time):
        if label not in ["violence", "weapon"]:
            return False

        if now_time - state["last_trigger"] < cooldown:
            return False

        # Nếu LSTM rất tự tin >= 0.90 thì cảnh báo ngay
        if conf >= instant_alert_conf:
            return True

        required_time = continuous_violence if label == "violence" else continuous_weapon
        start = state["continuous_start"][label]

        if start is None:
            state["continuous_start"][label] = now_time
            return False

        return (now_time - start) >= required_time

    try:
        fail_count = 0
        max_fail = 50

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                fail_count += 1

                if is_camera:
                    time.sleep(0.05)

                    if fail_count >= max_fail:
                        print("[WARN] Camera mất kết nối, thử reconnect...")
                        cap.release()
                        time.sleep(1)
                        cap = open_video_or_camera(source, is_camera=True)
                        fail_count = 0

                    continue

                break

            fail_count = 0
            frame_idx += 1
            pre_buffer.append(frame.copy())

            # FPS display
            now_perf = time.perf_counter()
            dt = now_perf - prev_time
            prev_time = now_perf

            if dt > 0:
                current_fps = 1.0 / dt
                fps_display = current_fps if fps_display <= 0 else (0.9 * fps_display + 0.1 * current_fps)

            # YOLO -> feature
            result = _run_yolo(yolo_model, frame)
            feat = _safe_extract_feature(result, width, height, yolo_names)
            frame_buffer.append(feat)

            # WINDOW_SIZE frame -> 1 window
            if len(frame_buffer) == WINDOW_SIZE:
                window_np = np.asarray(frame_buffer, dtype=np.float32)
                window_buffer.append(window_np)

            # SEQ_LEN window -> LSTM predict
            if len(window_buffer) == SEQ_LEN:
                last_lstm_label, last_lstm_conf, last_lstm_probs = _predict_lstm(
                    window_buffer=window_buffer,
                    lstm_model=lstm_model,
                    input_dim=input_dim,
                )
            if last_lstm_probs is not None:
                eval_stt += 1
                yolo_detect_text = _get_yolo_detect_text(result, yolo_names)

                raw_idx = int(np.argmax(last_lstm_probs))
                raw_label = ["non_violence", "violence", "weapon"][raw_idx]

                eval_csv_writer.writerow([
                    eval_stt,
                    frame_idx,
                    round(frame_idx / max(fps, 1e-8), 4),
                    yolo_detect_text,
                    last_lstm_label,
                    round(float(last_lstm_conf), 4),
                    round(float(last_lstm_probs[0]), 4),
                    round(float(last_lstm_probs[1]), 4),
                    round(float(last_lstm_probs[2]), 4),
                    raw_label,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ])

            else:
                last_lstm_label = "waiting..."
                last_lstm_conf = 0.0
                last_lstm_probs = None
                

                if last_lstm_probs is not None:
                    eval_stt += 1
                    yolo_detect_text = _get_yolo_detect_text(result, yolo_names)

                    raw_idx = int(np.argmax(last_lstm_probs))
                    raw_label = ["non_violence", "violence", "weapon"][raw_idx]

                    eval_csv_writer.writerow([
                        eval_stt,
                        frame_idx,
                        round(frame_idx / max(fps, 1e-8), 4),
                        yolo_detect_text,
                        last_lstm_label,
                        round(float(last_lstm_conf), 4),
                        round(float(last_lstm_probs[0]), 4),
                        round(float(last_lstm_probs[1]), 4),
                        round(float(last_lstm_probs[2]), 4),
                        raw_label,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ])
            timeline_labels.append(last_lstm_label)

            # Event trigger
            now_time = time.time()
            label = last_lstm_label
            conf = last_lstm_conf

            if label == "violence":
                state["continuous_start"]["weapon"] = None
            elif label == "weapon":
                state["continuous_start"]["violence"] = None
            else:
                state["continuous_start"]["violence"] = None
                state["continuous_start"]["weapon"] = None

            if not state["recording"] and can_trigger(label, conf, now_time):
                print("[INFO] EVENT TRIGGER:", label, conf)

                state["recording"] = True
                state["event_type"] = label
                state["confidence"] = conf

                event = create_event(frame, label, conf)
                state["event_obj"] = event

                state["frames"] = list(pre_buffer)
                state["post_frames"] = int(fps * 5)
                state["last_trigger"] = now_time

                state["continuous_start"]["violence"] = None
                state["continuous_start"]["weapon"] = None

            if state["recording"]:
                state["frames"].append(frame.copy())
                state["post_frames"] -= 1

                if state["post_frames"] <= 0:
                    print("[INFO] SAVE EVENT:", state["event_type"])

                    clip_path = save_event_clip(
                        frames=state["frames"],
                        fps=fps,
                        width=width,
                        height=height,
                        label=state["event_type"],
                    )

                    if state["event_obj"] is not None:
                        state["event_obj"].clip = clip_path
                        state["event_obj"].save(update_fields=["clip"])

                        event_id = state["event_obj"].id

                        threading.Thread(
                            target=send_event_email_async,
                            args=(event_id,),
                            daemon=True
                        ).start()

                        threading.Thread(
                            target=send_event_telegram_async,
                            args=(event_id,),
                            daemon=True
                        ).start()

                        state["event_obj"] = None

                    state["recording"] = False
                    state["frames"] = []

            # Draw
            frame = _draw_yolo(frame, result, yolo_names, last_lstm_label)
            frame = _draw_lstm(frame, last_lstm_label, last_lstm_conf, fps_display, frame_idx)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    finally:
        cap.release()

        try:
            eval_csv_file.close()
            print(f"[INFO] Đã lưu CSV eval giống realtime: {eval_csv_path}")
        except Exception as e:
            print("[WARN] Không thể đóng CSV eval:", e)

        if timeline_labels:
            try:
                stat_rows = build_action_statistics_from_timeline(
                    timeline_labels=timeline_labels,
                    fps=fps,
                    start_datetime=process_start_datetime,
                )

                save_lstm_statistics_csv(
                    stat_rows=stat_rows,
                    output_dir=OUTPUT_PATH,
                    video_name=source_name,
                )

                print(f"[INFO] Đã lưu CSV thống kê: {source_name}_lstm_statistics.csv")

            except Exception as e:
                print(f"[WARN] Không thể lưu CSV thống kê: {e}")


def generate_processed_frames(input_path):
    return _generate_processed_frames(input_path, is_camera=False)


def generate_processed_frames_camera(camera_index=1):
    """
    Webcam ngoài CyberTrack H7 của bạn đang là index 1.
    Nếu đổi camera, sửa camera_index tại đây.
    """
    return _generate_processed_frames(camera_index, is_camera=True)

def send_event_email_async(event_id):
    try:
        close_old_connections()
        event = Event.objects.get(id=event_id)
        #send_event_email(event)
    except Exception as e:
        print("[EMAIL ERROR]", e)
    finally:
        close_old_connections()


def send_event_telegram_async(event_id):
    try:
        close_old_connections()
        event = Event.objects.get(id=event_id)
        send_event_telegram(event)
    except Exception as e:
        print("[TELEGRAM ERROR]", e)
    finally:
        close_old_connections()