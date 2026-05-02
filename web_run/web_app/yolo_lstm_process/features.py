import numpy as np

from .ai_config import CONF_THRES, FRAME_FEATURE_DIM


prev_main_bbox = None


def normalize_name(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace('-', '_').replace(' ', '_')
    return name

def get_yolo_name(yolo_names, cls_id: int) -> str:
    try:
        if isinstance(yolo_names, dict):
            return str(yolo_names.get(int(cls_id), f'class_{cls_id}'))
        return str(yolo_names[int(cls_id)])
    except Exception:
        return f'class_{cls_id}'

def yolo_cls_to_lstm_cls(yolo_cls: int, yolo_names=None):
   
    name = normalize_name(get_yolo_name(yolo_names, yolo_cls)) if yolo_names is not None else ''

    weapon_keywords = ['weapon']
    violence_keywords = ['violence']
    non_keywords = ['non_violence']

    if any(k in name for k in weapon_keywords):
        return 2
    if any(k in name for k in violence_keywords):
        return 1
    if any(k in name for k in non_keywords):
        return 0

    # Fallback cho YOLO 2 lớp: [violence, weapon]
    if int(yolo_cls) == 0:
        return 1
    if int(yolo_cls) == 1:
        return 2
    if int(yolo_cls) == 2:
        return 2
    return None


def _bbox_to_normalized_xywh(box, frame_w: int, frame_h: int):
    x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(float)
    bw = max(0.0, (x2 - x1) / max(frame_w, 1))
    bh = max(0.0, (y2 - y1) / max(frame_h, 1))
    cx = ((x1 + x2) / 2.0) / max(frame_w, 1)
    cy = ((y1 + y2) / 2.0) / max(frame_h, 1)
    area = bw * bh
    return float(cx), float(cy), float(bw), float(bh), float(area)


def _min_center_distance(centers):
    if len(centers) < 2:
        return 1.0
    min_dist = 1.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            min_dist = min(min_dist, float(np.sqrt(dx * dx + dy * dy)))
    return float(min_dist)


def extract_feature_6(result, yolo_names=None) -> np.ndarray:
    """YOLO result -> 6 dim: [onehot(3) + conf(3)]."""
    onehot = np.zeros(3, dtype=np.float32)
    confs = np.zeros(3, dtype=np.float32)

    if result.boxes is None or len(result.boxes) == 0:
        onehot[0] = 1.0
        confs[0] = 1.0
        return np.concatenate([onehot, confs]).astype(np.float32)

    for box in result.boxes:
        yolo_cls = int(box.cls[0])
        score = float(box.conf[0])

        if score < CONF_THRES:
            continue

        lstm_cls = yolo_cls_to_lstm_cls(yolo_cls, yolo_names)
        if lstm_cls is None or not (0 <= lstm_cls < 3):
            continue

        onehot[lstm_cls] = 1.0
        confs[lstm_cls] = max(confs[lstm_cls], score)

    # Không có detection hợp lệ sau lọc conf -> non nhưng conf=0 để không ép chắc chắn an toàn.
    if float(confs.max()) <= 0.0:
        onehot[0] = 1.0
        confs[0] = 1.0

    return np.concatenate([onehot, confs]).astype(np.float32)


def extract_feature_18(result, frame_shape, yolo_names=None) -> np.ndarray:
    """Giữ tương thích nếu quay lại model 18 feature/frame."""
    global prev_main_bbox

    frame_h, frame_w = frame_shape[:2]
    onehot = np.zeros(3, dtype=np.float32)
    confs = np.zeros(3, dtype=np.float32)

    centers = []
    valid_dets = []
    num_objects = 0
    num_violence = 0
    num_weapon = 0

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            yolo_cls = int(box.cls[0])
            score = float(box.conf[0])
            if score < CONF_THRES:
                continue

            lstm_cls = yolo_cls_to_lstm_cls(yolo_cls, yolo_names)
            if lstm_cls is None or not (0 <= lstm_cls < 3):
                continue

            cx, cy, bw, bh, area = _bbox_to_normalized_xywh(box, frame_w, frame_h)
            onehot[lstm_cls] = 1.0
            confs[lstm_cls] = max(confs[lstm_cls], score)
            centers.append((cx, cy))
            valid_dets.append({
                'lstm_cls': lstm_cls,
                'conf': score,
                'cx': cx,
                'cy': cy,
                'w': bw,
                'h': bh,
                'area': area,
            })
            num_objects += 1
            if lstm_cls == 1:
                num_violence += 1
            elif lstm_cls == 2:
                num_weapon += 1

    if num_objects == 0:
        onehot[0] = 1.0
        confs[0] = 0.0
        cx, cy, bw, bh, area = 0.0, 0.0, 0.0, 0.0, 0.0
        dx, dy, speed = 0.0, 0.0, 0.0
        min_dist = 1.0
        prev_main_bbox = None
    else:
        danger = [d for d in valid_dets if d['lstm_cls'] in (1, 2)]
        candidates = danger if danger else valid_dets
        main = max(candidates, key=lambda d: d['conf'])
        cx, cy, bw, bh, area = main['cx'], main['cy'], main['w'], main['h'], main['area']

        if prev_main_bbox is None:
            dx, dy, speed = 0.0, 0.0, 0.0
        else:
            dx = float(cx - prev_main_bbox[0])
            dy = float(cy - prev_main_bbox[1])
            speed = float(np.sqrt(dx * dx + dy * dy))
        prev_main_bbox = (cx, cy)
        min_dist = _min_center_distance(centers)

    context = np.asarray([
        cx, cy, bw, bh, area,
        dx, dy, speed,
        min_dist,
        min(num_objects, 10) / 10.0,
        min(num_violence, 10) / 10.0,
        min(num_weapon, 10) / 10.0,
    ], dtype=np.float32)

    return np.concatenate([onehot, confs, context]).astype(np.float32)


def extract_frame_features(
    result,
    width=None,
    height=None,
    yolo_names=None,
    track_history=None,
    frame_width=None,
    frame_height=None,
):
    if frame_width is not None:
        width = frame_width
    if frame_height is not None:
        height = frame_height

    if FRAME_FEATURE_DIM == 6:
        feat = extract_feature_6(result, yolo_names=yolo_names)
    elif FRAME_FEATURE_DIM == 18:
        if width is None or height is None:
            raise ValueError('Cần width/height khi FRAME_FEATURE_DIM=18')
        feat = extract_feature_18(result, (int(height), int(width), 3), yolo_names=yolo_names)
    else:
        raise ValueError(f'FRAME_FEATURE_DIM không hỗ trợ: {FRAME_FEATURE_DIM}')

    if feat.shape[0] != FRAME_FEATURE_DIM:
        raise ValueError(f'Feature sai chiều: {feat.shape[0]}, cần {FRAME_FEATURE_DIM}')
    return feat.astype(np.float32)


