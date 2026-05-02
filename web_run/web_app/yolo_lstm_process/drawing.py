import cv2
from datetime import datetime

from .ai_config import CONF_THRES
from .features import yolo_cls_to_lstm_cls, get_yolo_name


COLOR_NON = (0, 255, 0)       
COLOR_VIOLENCE = (0, 255, 255)
COLOR_WEAPON = (0, 0, 255)     
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def get_color_by_label(label: str):
    label = str(label).lower()
    if label == "violence":
        return COLOR_VIOLENCE
    if label == "weapon":
        return COLOR_WEAPON
    if label == "non_violence":
        return COLOR_NON
    return COLOR_WHITE


def get_bg_color_by_label(label: str):
    label = str(label).lower()
    if label == "violence":
        return (0, 80, 80)
    if label == "weapon":
        return (0, 0, 90)
    if label == "non_violence":
        return (0, 70, 0)
    return (50, 50, 50)


def draw_text_with_bg(
    frame,
    text: str,
    org: tuple,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.65,
    color: tuple = COLOR_WHITE,
    thickness: int = 2,
    bg_color: tuple = COLOR_BLACK,
    alpha: float = 0.55,
    pad: int = 6,
):
    x, y = org
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x1 = max(0, x - pad)
    y1 = max(0, y - th - pad)
    x2 = min(frame.shape[1] - 1, x + tw + pad)
    y2 = min(frame.shape[0] - 1, y + baseline + pad)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, text, org, font, scale, COLOR_BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_yolo_boxes(frame, result, yolo_names, lstm_label):
    
    lstm_label = str(lstm_label).lower()
    if lstm_label not in ["violence", "weapon"]:
        return frame
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    target_cls = 1 if lstm_label == "violence" else 2
    color = get_color_by_label(lstm_label)

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < CONF_THRES:
            continue

        lstm_cls = yolo_cls_to_lstm_cls(cls_id, yolo_names)
        if lstm_cls != target_cls:
            continue

        yolo_label = get_yolo_name(yolo_names, cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{yolo_label} {conf:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_lstm_status(frame, lstm_label, lstm_conf, fps_value, frame_idx):
  
    h, _ = frame.shape[:2]
    label = str(lstm_label)
    conf = 0.0 if lstm_conf is None else float(lstm_conf)

    label_color = get_color_by_label(label)
    bg_color = get_bg_color_by_label(label)

    draw_text_with_bg(
        frame,
        f"LSTM: {label} {conf:.2f}",
        (20, 42),
        scale=0.82,
        color=label_color,
        thickness=2,
        bg_color=bg_color,
        alpha=0.70,
        pad=8,
    )

    draw_text_with_bg(
        frame,
        f"Frame: {frame_idx}",
        (20, 82),
        scale=0.62,
        color=COLOR_WHITE,
        thickness=2,
        bg_color=COLOR_BLACK,
        alpha=0.45,
        pad=5,
    )

    draw_text_with_bg(
        frame,
        f"FPS: {fps_value:.2f}",
        (20, 118),
        scale=0.62,
        color=COLOR_WHITE,
        thickness=2,
        bg_color=COLOR_BLACK,
        alpha=0.45,
        pad=5,
    )

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw_text_with_bg(
        frame,
        now_text,
        (20, h - 20),
        scale=0.58,
        color=COLOR_WHITE,
        thickness=2,
        bg_color=COLOR_BLACK,
        alpha=0.45,
        pad=5,
    )
    return frame
