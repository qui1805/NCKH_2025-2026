# feature_extractor.py
import numpy as np
from collections import defaultdict, deque

from config import YOLO_CLASS_NAMES, YOLO_CLASS_TO_IDX

# Khởi tạo biến dùng để lưu lịch sử track
track_history = defaultdict(lambda: deque(maxlen=10))

def reset_track_history():
    global track_history
    track_history = defaultdict(lambda: deque(maxlen=10))
    
def extract_frame_features(result, frame_w, frame_h, yolo_names):
    global track_history

    # FIX CỨNG mapping đúng với lúc train
    class_to_idx = {
        "violence": 0,
        "Weapon": 1,
        "weapon": 1
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

        # ===== BASIC =====
        num_obj[idx] += 1
        conf_sum[idx] += conf
        conf_max[idx] = max(conf_max[idx], conf)

        area = ((x2 - x1) * (y2 - y1)) / max(frame_w * frame_h, 1)
        area_sum[idx] += area
        area_max[idx] = max(area_max[idx], area)

        # ===== SPEED =====
        if track_ids is not None:
            tid = int(track_ids[i])
            cx = (x1 + x2) / 2 / frame_w
            cy = (y1 + y2) / 2 / frame_h

            track_history[tid].append((cx, cy))

            if len(track_history[tid]) >= 2:
                px, py = track_history[tid][-2]
                speed = np.sqrt((cx - px)**2 + (cy - py)**2)
                speeds.append(speed)

    mean_conf = [
        conf_sum[i] / num_obj[i] if num_obj[i] > 0 else 0.0
        for i in range(num_classes)
    ]

    mean_area = [
        area_sum[i] / num_obj[i] if num_obj[i] > 0 else 0.0
        for i in range(num_classes)
    ]

    mean_speed = np.mean(speeds) if len(speeds) > 0 else 0.0
    max_speed = np.max(speeds) if len(speeds) > 0 else 0.0

    feature = [
        num_obj[0], num_obj[1],
        conf_max[0], conf_max[1],
        mean_conf[0], mean_conf[1],
        area_max[0], area_max[1],
        mean_area[0], mean_area[1],
        mean_speed,
        max_speed
    ]

    return np.array(feature, dtype=np.float32)
