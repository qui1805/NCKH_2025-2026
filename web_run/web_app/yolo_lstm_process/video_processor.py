

import os
import cv2
import time
import numpy as np
import torch
from collections import deque
from datetime import datetime

from .features import extract_frame_features
from .model_loader import load_models
from .drawing import draw_lstm_status, draw_yolo_boxes
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

def _get_model_input_dim(lstm_model):
    try:
        return int(lstm_model.lstm.input_size)
    except Exception:
        return int(WINDOW_SIZE * FRAME_FEATURE_DIM)

def _predict_lstm(window_buffer, lstm_model, input_dim):
    seq_np = np.asarray(window_buffer, dtype=np.float32)
    if seq_np.shape[0] != SEQ_LEN:
        return 'waiting...', 0.0

    seq_np = seq_np.reshape(1, SEQ_LEN, -1)
    if seq_np.shape[2] != int(input_dim):
        raise RuntimeError(
            f'Sai input_dim LSTM: web tạo {seq_np.shape[2]}, nhưng model cần {input_dim}. '
            f'Kiểm tra WINDOW_SIZE={WINDOW_SIZE}, FRAME_FEATURE_DIM={FRAME_FEATURE_DIM}.'
        )

    x = torch.tensor(seq_np, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = lstm_model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    p_non, p_vio, p_wea = map(float, probs[:3])
    violence_ok = p_vio >= VIOLENCE_THRES
    weapon_ok = p_wea >= WEAPON_THRES

    if weapon_ok and (not violence_ok or p_wea >= p_vio):
        return 'weapon', p_wea
    if violence_ok:
        return 'violence', p_vio
    return 'non_violence', p_non

def _run_yolo(yolo_model, frame):
    if FRAME_FEATURE_DIM == 6:
        return yolo_model(frame, conf=CONF_THRES, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
    return yolo_model.track(frame, persist=True, conf=CONF_THRES, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]


def _extract_feature(result, width, height, yolo_names):
    feat = extract_frame_features(result=result, frame_width=width, frame_height=height, yolo_names=yolo_names)
    feat = np.asarray(feat, dtype=np.float32)
    if feat.shape[0] != FRAME_FEATURE_DIM:
        raise RuntimeError(
            f'Feature sai kích thước. Model LSTM cần {FRAME_FEATURE_DIM} feature/frame, '
            f'nhưng đang nhận {feat.shape[0]}.'
        )
    return feat


def _draw_lstm(frame, label, conf, fps_value, frame_idx):
    try:
        return draw_lstm_status(frame=frame, label=label, conf=conf, fps_value=fps_value, frame_idx=frame_idx)
    except TypeError:
        return draw_lstm_status(frame, label, conf, fps_value, frame_idx)


def _safe_video_writer(save_path, fps, size):
    for codec in ['avc1', 'mp4v', 'XVID', 'MJPG']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(save_path, fourcc, fps, size)
        if writer.isOpened():
            print(f'[INFO] VideoWriter codec: {codec}')
            return writer
        writer.release()
    raise RuntimeError('Không thể tạo video output.')


def process_video(input_path, output_dir=None, show_window=False, save_video=True):
    output_dir = output_dir or OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)

    yolo_model, lstm_model = load_models()
    yolo_names = yolo_model.names
    input_dim = _get_model_input_dim(lstm_model)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Không mở được video: {input_path}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        width, height = 640, 480

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    video_name = os.path.splitext(os.path.basename(str(input_path)))[0]
    save_path = os.path.join(str(output_dir), f'{video_name}_lstm_output.mp4')

    writer = _safe_video_writer(save_path, fps, (width, height)) if save_video else None

    frame_buffer = deque(maxlen=WINDOW_SIZE)
    window_buffer = deque(maxlen=SEQ_LEN)
    last_lstm_label = 'waiting...'
    last_lstm_conf = 0.0
    frame_idx = 0
    prev_time = time.perf_counter()
    fps_display = 0.0
    timeline_labels = []
    process_start_datetime = datetime.now()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            now_perf = time.perf_counter()
            dt = now_perf - prev_time
            prev_time = now_perf
            if dt > 0:
                current_fps = 1.0 / dt
                fps_display = current_fps if fps_display <= 0 else (0.9 * fps_display + 0.1 * current_fps)

            result = _run_yolo(yolo_model, frame)
            feat = _extract_feature(result, width, height, yolo_names)
            frame_buffer.append(feat)

            if len(frame_buffer) == WINDOW_SIZE:
                window_np = np.asarray(frame_buffer, dtype=np.float32)
                window_buffer.append(window_np)

            if len(window_buffer) == SEQ_LEN:
                last_lstm_label, last_lstm_conf = _predict_lstm(window_buffer, lstm_model, input_dim)
            else:
                last_lstm_label = 'waiting...'
                last_lstm_conf = 0.0

            timeline_labels.append(last_lstm_label)

            frame = draw_yolo_boxes(frame, result, yolo_names, last_lstm_label)
            frame = _draw_lstm(frame, last_lstm_label, last_lstm_conf, fps_display, frame_idx)

            if show_window:
                cv2.imshow('YOLO + LSTM Test', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

            if writer is not None:
                writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    stat_rows = build_action_statistics_from_timeline(
        timeline_labels=timeline_labels,
        fps=fps,
        start_datetime=process_start_datetime,
    )
    csv_path = save_lstm_statistics_csv(stat_rows, output_dir, video_name)

    return {
        'output_path': save_path if save_video else None,
        'csv_path': csv_path,
        'final_label': last_lstm_label,
        'final_conf': last_lstm_conf,
        'total_frames': frame_idx,
        'statistics_count': len(stat_rows),
    }
