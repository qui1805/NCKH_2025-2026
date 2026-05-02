
import os
import csv
import cv2
import torch
import numpy as np
from datetime import datetime
from collections import deque, Counter
import time
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
YOLO_MODEL_PATH = r"C:\Train_LSTM\models\yolo.pt"
LSTM_MODEL_PATH = r"C:\Train_LSTM\models\best_lstm.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE = 32
SEQ_LEN = 16
FRAME_FEATURE_DIM = 6

LSTM_CLASS_NAMES = ["non_violence", "violence", "weapon"]
NUM_CLASSES = 3
CONF_THRES = 0.75
VIOLENCE_THRES = 0.8
WEAPON_THRES = 0.7

DEBUG_YOLO_EVERY = 0
SAVE_DIR = r"C:\Train_LSTM\output\videos"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# LOAD YOLO
# =========================
print(f"[INFO] Device: {DEVICE}")
print("[INFO] Loading YOLO...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("[INFO] YOLO names:", yolo_model.names)


# =========================
# CLASS NAME MAPPING
# =========================
def normalize_name(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace("-", "_").replace(" ", "_")
    return name

def yolo_cls_to_lstm_cls(cls_id: int):
    try:
        raw_name = yolo_model.names[int(cls_id)]
    except Exception:
        return None

    name = normalize_name(raw_name)

    weapon_keywords = ["weapon"]
    violence_keywords = ["violence"]
    non_keywords = ["non_violence"]

    if any(k in name for k in weapon_keywords):
        return 2
    if any(k in name for k in violence_keywords):
        return 1
    if any(k in name for k in non_keywords):
        return 0
    return None


def get_yolo_label_name(cls_id: int) -> str:
    try:
        return str(yolo_model.names[int(cls_id)])
    except Exception:
        return f"class_{cls_id}"

def get_yolo_detect_text(results) -> str:
    if results.boxes is None or len(results.boxes) == 0:
        return "none"

    best_label = "none"
    best_conf = 0.0

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRES:
            continue

        cls_id = int(box.cls[0])
        label = get_yolo_label_name(cls_id)

        if conf > best_conf:
            best_conf = conf
            best_label = label

    if best_label == "none":
        return "none"

    return f"{best_label}:{best_conf:.2f}"

def get_color_by_lstm_cls(lstm_cls: int):
    if lstm_cls == 1:
        return (0, 255, 255)
    if lstm_cls == 2:
        return (0, 0, 255)
    if lstm_cls == 0:
        return (0, 255, 0)
    return (255, 255, 255)


def draw_text_with_bg(
    frame,
    text: str,
    org: tuple,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.65,
    color: tuple = (255, 255, 255),
    thickness: int = 2,
    bg_color: tuple = (0, 0, 0),
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
    cv2.putText(frame, text, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def get_lstm_status_bg_color(lstm_cls: int):
    if lstm_cls == 1:
        return (0, 80, 80)
    if lstm_cls == 2:
        return (0, 0, 90)
    if lstm_cls == 0:
        return (0, 70, 0)
    return (40, 40, 40)


# =========================
# MODEL DEFINITIONS
# =========================
class AttentionPooling(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.score = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.score(x).squeeze(-1)
        attn = torch.softmax(attn, dim=1)
        return torch.sum(x * attn.unsqueeze(-1), dim=1)


class ActionLSTMAttention(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_size: int,
                 num_layers: int, dropout: float, bidirectional: bool):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.attn = AttentionPooling(out_dim)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(out_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(out_dim, hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = self.attn(out)
        return self.head(pooled)


class ActionLSTMLegacy(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_size: int,
                 num_layers: int, dropout: float, bidirectional: bool):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(out_dim, hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# =========================
# LOAD LSTM CHECKPOINT
# =========================
print("[INFO] Loading LSTM checkpoint...")
checkpoint = torch.load(LSTM_MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint["model_state_dict"]

input_dim = int(checkpoint.get("input_dim", WINDOW_SIZE * FRAME_FEATURE_DIM))
hidden_size = int(checkpoint.get("hidden_size", 128))
num_layers = int(checkpoint.get("num_layers", 2))
bidirectional = bool(checkpoint.get("bidirectional", True))
num_classes = int(checkpoint.get("num_classes", NUM_CLASSES))
dropout = float(checkpoint.get("dropout", 0.3))

SEQ_LEN = int(checkpoint.get("seq_len", SEQ_LEN))
FRAME_FEATURE_DIM = int(checkpoint.get("frame_feature_dim", FRAME_FEATURE_DIM))

if "window_size" in checkpoint:
    WINDOW_SIZE = int(checkpoint["window_size"])
elif input_dim % FRAME_FEATURE_DIM == 0:
    WINDOW_SIZE = int(input_dim // FRAME_FEATURE_DIM)
else:
    raise ValueError(f"Không suy ra được WINDOW_SIZE: input_dim={input_dim}, FRAME_FEATURE_DIM={FRAME_FEATURE_DIM}")

ckpt_class_names = checkpoint.get("class_names", LSTM_CLASS_NAMES)
if len(ckpt_class_names) == NUM_CLASSES:
    LSTM_CLASS_NAMES = ckpt_class_names

use_attention = any(k.startswith("attn.") for k in state_dict.keys())
if use_attention:
    print("[INFO] Model type: Attention LSTM")
    lstm_model = ActionLSTMAttention(input_dim, num_classes, hidden_size, num_layers, dropout, bidirectional).to(DEVICE)
else:
    print("[INFO] Model type: Legacy LSTM")
    lstm_model = ActionLSTMLegacy(input_dim, num_classes, hidden_size, num_layers, dropout, bidirectional).to(DEVICE)

lstm_model.load_state_dict(state_dict, strict=True)
lstm_model.eval()

print(f"[INFO] LSTM class names: {LSTM_CLASS_NAMES}")
print(f"[INFO] Runtime shape: WINDOW_SIZE={WINDOW_SIZE}, SEQ_LEN={SEQ_LEN}, FRAME_FEATURE_DIM={FRAME_FEATURE_DIM}")
print("[INFO] Realtime rule: YOLO only creates 6D features; LSTM decides label by sequence.")
print(f"[INFO] Prob gate: violence >= {VIOLENCE_THRES:.2f}, weapon >= {WEAPON_THRES:.2f}")

# =========================
# BUFFERS
# =========================
frame_buffer = deque(maxlen=WINDOW_SIZE)
window_buffer = deque(maxlen=SEQ_LEN)
#window_label_buffer = deque(maxlen=SEQ_LEN)
#prob_smooth_buffer = deque(maxlen=SMOOTH_QUEUE_LEN)


# =========================
# FEATURE + RULE GATE
# =========================
def extract_feature(results) -> np.ndarray:
    onehot = np.zeros(3, dtype=np.float32)
    confs = np.zeros(3, dtype=np.float32)

    if results.boxes is None:
        onehot[0] = 1.0
        confs[0] = 1.0
        return np.concatenate([onehot, confs]).astype(np.float32)

    has_valid = False
    for box in results.boxes:
        yolo_cls = int(box.cls[0])
        score = float(box.conf[0])
        if score < CONF_THRES:
            continue

        lstm_cls = yolo_cls_to_lstm_cls(yolo_cls)
        if lstm_cls is None or not (0 <= lstm_cls < 3):
            continue

        has_valid = True
        onehot[lstm_cls] = 1.0
        confs[lstm_cls] = max(confs[lstm_cls], score)

    if not has_valid:
        onehot[0] = 1.0
        confs[0] = 1.0

    return np.concatenate([onehot, confs]).astype(np.float32)

def final_label_by_probability(prob: np.ndarray) -> int:
    p_non, p_vio, p_wea = map(float, prob[:3])

    if p_wea >= WEAPON_THRES and p_wea >= p_vio:
        return 2

    if p_vio >= VIOLENCE_THRES:
        return 1

    return 0


def build_window_from_frame_buffer() -> np.ndarray:
    return np.asarray(frame_buffer, dtype=np.float32)


def predict_lstm_sequence():
    seq = np.asarray(window_buffer, dtype=np.float32)
    seq = seq.reshape(1, SEQ_LEN, -1)

    if seq.shape[2] != input_dim:
        raise ValueError(f"Sai input_dim: realtime tạo {seq.shape[2]}, checkpoint cần {input_dim}")

    x = torch.tensor(seq, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        logits = lstm_model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred = final_label_by_probability(probs)
    return pred, probs


def get_display_conf(display_pred: int, display_prob: np.ndarray) -> float:
    if display_prob is None or len(display_prob) < 3:
        return 0.0
    if display_pred == 1:
        return float(display_prob[1])
    if display_pred == 2:
        return float(display_prob[2])
    return float(display_prob[0])


# =========================
# DRAW + CSV
# =========================
def safe_video_writer(output_path: str, fps: float, size):
    for codec in ["mp4v", "XVID", "MJPG"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        if writer.isOpened():
            print(f"[INFO] VideoWriter codec: {codec}")
            return writer
        writer.release()
    print("[WARN] Không tạo được VideoWriter. Sẽ chạy không ghi video.")
    return None


def create_eval_csv(output_path: str):
    csv_path = os.path.splitext(output_path)[0] + "_lstm_eval.csv"
    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
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
    return csv_path, csv_file, csv_writer


def update_label_counter(counter: dict, label_name: str):
    counter[label_name] = counter.get(label_name, 0) + 1


def get_final_label_from_counter(counter: dict) -> str:
    if not counter:
        return "no_prediction"
    priority = {"weapon": 3, "violence": 2, "non_violence": 1}
    return max(counter.items(), key=lambda x: (x[1], priority.get(x[0], 0)))[0]


def draw_lstm_info(frame, frame_id, fps_value, display_pred, display_prob, ready_sequence):
    if ready_sequence:
        label = LSTM_CLASS_NAMES[display_pred]
        conf = get_display_conf(display_pred, display_prob)
        display_text = f"LSTM: {label} {conf:.2f}"
        display_color = get_color_by_lstm_cls(display_pred)
        bg_color = get_lstm_status_bg_color(display_pred)
    else:
        display_text = "LSTM: waiting 0.00"
        display_color = (255, 255, 255)
        bg_color = (50, 50, 50)

    draw_text_with_bg(frame, display_text, (20, 42), scale=0.75, color=display_color, thickness=2, bg_color=bg_color, alpha=0.70, pad=8)
    draw_text_with_bg(frame, f"Frame: {frame_id}", (20, 82), scale=0.62, color=(255, 255, 255), thickness=2, bg_color=(0, 0, 0), alpha=0.45, pad=5)
    draw_text_with_bg(frame, f"FPS: {fps_value:.2f}", (20, 118), scale=0.62, color=(255, 255, 255), thickness=2, bg_color=(0, 0, 0), alpha=0.45, pad=5)


def draw_yolo_boxes_normal(frame, results):
    if results.boxes is None:
        return

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        yolo_cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < CONF_THRES:
            continue

        lstm_cls = yolo_cls_to_lstm_cls(yolo_cls)
        color = get_color_by_lstm_cls(lstm_cls) if lstm_cls is not None else (255, 255, 255)
        yolo_name = get_yolo_label_name(yolo_cls)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{yolo_name} {conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# =========================
# MAIN LOOP
# =========================
def run_video(source=0, save_output=True, output_name=None):
    global frame_buffer, window_buffer

    frame_buffer.clear()
    window_buffer.clear()
    #window_label_buffer.clear()
    #prob_smooth_buffer.clear()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được nguồn video: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 20.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        width, height = 640, 480

    writer = None
    output_path = None
    csv_path = None
    csv_file = None
    csv_writer = None
    eval_stt = 0
    label_counter = {name: 0 for name in LSTM_CLASS_NAMES}

    if save_output:
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"record_{timestamp}.mp4"
        output_path = os.path.join(SAVE_DIR, output_name)
        writer = safe_video_writer(output_path, fps, (width, height))
        if writer is not None:
            print(f"[INFO] Đang ghi video ra: {output_path}")
        csv_path, csv_file, csv_writer = create_eval_csv(output_path)
        print(f"[INFO] Đang ghi CSV đánh giá nhãn: {csv_path}")

    frame_id = 0
    sequence_pred = 0
    sequence_prob = None
    has_sequence_prediction = False

    prev_time = time.perf_counter()
    fps_display = 0.0

    print("[INFO] Nhấn ESC để thoát.")
    print("[INFO] Realtime: YOLO tạo feature 6D -> 32 frame/window -> 16 window/sample -> LSTM dự đoán.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            now_perf = time.perf_counter()
            dt = now_perf - prev_time
            prev_time = now_perf
            if dt > 0:
                current_fps = 1.0 / dt
                fps_display = current_fps if fps_display <= 0 else (0.9 * fps_display + 0.1 * current_fps)

            results = yolo_model(frame, verbose=False)[0]
            yolo_detect_text = get_yolo_detect_text(results)
            feat = extract_feature(results)
            frame_buffer.append(feat)

            if len(frame_buffer) == WINDOW_SIZE:
                window = build_window_from_frame_buffer()
                window_buffer.append(window)

            if len(window_buffer) == SEQ_LEN:
                sequence_pred, sequence_prob = predict_lstm_sequence()
                has_sequence_prediction = True


                if csv_writer is not None and sequence_prob is not None:
                    eval_stt += 1
                    label_name = LSTM_CLASS_NAMES[sequence_pred]
                    conf = get_display_conf(sequence_pred, sequence_prob)
                    update_label_counter(label_counter, label_name)

                    csv_writer.writerow([
                        eval_stt,
                        frame_id,
                        round(frame_id / max(fps, 1e-8), 4),
                        yolo_detect_text,
                        label_name,
                        round(float(conf), 4),
                        round(float(sequence_prob[0]), 4),
                        round(float(sequence_prob[1]), 4),
                        round(float(sequence_prob[2]), 4),
                        LSTM_CLASS_NAMES[int(np.argmax(sequence_prob))],
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ])

            # Chỉ hiện box YOLO khi LSTM đã dự đoán ra violence hoặc weapon
            if has_sequence_prediction and sequence_pred in [1, 2]:
                draw_yolo_boxes_normal(frame, results)

            draw_lstm_info(frame, frame_id, fps_display, sequence_pred, sequence_prob, has_sequence_prediction)

            now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            draw_text_with_bg(frame, now_text, (20, height - 20), scale=0.58, color=(255, 255, 255), thickness=2, bg_color=(0, 0, 0), alpha=0.45, pad=5)

            if writer is not None:
                writer.write(frame)

            cv2.imshow("YOLO feature + LSTM structure-gated output", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        if csv_writer is not None:
            final_label = get_final_label_from_counter(label_counter)
            final_count = label_counter.get(final_label, 0)
            total_predictions = sum(label_counter.values())

            csv_writer.writerow([])
            csv_writer.writerow(["SUMMARY"])
            csv_writer.writerow(["total_frames", frame_id])
            csv_writer.writerow(["total_lstm_predictions", total_predictions])
            csv_writer.writerow(["final_label_by_counter", final_label])
            csv_writer.writerow(["final_label_count", final_count])
            for name in LSTM_CLASS_NAMES:
                csv_writer.writerow([f"count_{name}", label_counter.get(name, 0)])

        if csv_file is not None:
            csv_file.close()

    if output_path is not None and writer is not None:
        print(f"[INFO] Đã lưu video tại: {output_path}")
    if csv_path is not None:
        print(f"[INFO] Đã lưu CSV đánh giá nhãn tại: {csv_path}")


print("[MODE] Predict mode: structure_rule + LSTM_probability_gate")

# =========================
# RUN
# =========================
if __name__ == "__main__":

    # 1) Webcam + record
    #run_video(source=1, save_output=True)

    # 2) Video file + record
    run_video(source=r"C:\Train_LSTM\videos\non.avi", save_output=True,)

    # 3) Webcam không record
    # run_video(source=0, save_output=False)


