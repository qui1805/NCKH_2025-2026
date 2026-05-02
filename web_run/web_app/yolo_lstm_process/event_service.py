import os
import cv2
from datetime import datetime

from config import settings
from monitoring.models import Event



VALID_EVENT_LABELS = ["violence", "weapon"]

EVENT_CONF_THRES = 0.7


def save_event_image(frame, label):
    filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    rel_path = os.path.join("alerts", "images", filename)
    abs_path = os.path.join(settings.MEDIA_ROOT, rel_path)

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    ok = cv2.imwrite(abs_path, frame)
    if not ok:
        raise RuntimeError("Không thể lưu ảnh sự kiện")

    return rel_path.replace("\\", "/")


def save_event_clip(frames, fps, width, height, label):
    if not frames:
        raise RuntimeError("Không có frame để lưu clip sự kiện")

    if fps <= 0:
        fps = 25.0

    clip_folder = os.path.join(settings.MEDIA_ROOT, "alerts", "clips")
    os.makedirs(clip_folder, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    clip_filename = f"{label}_{ts}.mp4"
    clip_path = os.path.join(clip_folder, clip_filename)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[WARN] avc1 không hoạt động, chuyển sang mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError("Không thể tạo clip sự kiện")

    for frame in frames:
        writer.write(frame)

    writer.release()

    rel_path = os.path.join("alerts", "clips", clip_filename)
    return rel_path.replace("\\", "/")


def create_event(frame, label, conf):
    label = str(label).lower()

    if label not in VALID_EVENT_LABELS:
        print(f"[INFO] Bỏ qua event không nguy hiểm: {label}")
        return None

    if conf < EVENT_CONF_THRES:
        print(f"[INFO] Bỏ qua event do confidence thấp: {conf:.2f}")
        return None

    image_path = save_event_image(frame, label)

    event = Event.objects.create(
        event_type=label,
        confidence=float(conf),
        image=image_path,
        clip=None,
    )

    return event