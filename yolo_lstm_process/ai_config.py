import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

YOLO_MODEL_PATH = PROJECT_ROOT / "weights" / "best.pt"
LSTM_MODEL_PATH = PROJECT_ROOT / "weights" / "best_lstmnew.pth"

VIDEO_PATH = PROJECT_ROOT / "media" / "test_inputs" / "test_2.mp4"
OUTPUT_PATH = PROJECT_ROOT / "media" / "test_outputs"

CONF_THRES = 0.7
IMGSZ = 640
SEQ_LEN = 64
THRESH_LSTM = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_CLASS_NAMES = ["violence", "Weapon"]
YOLO_CLASS_TO_IDX = {name: i for i, name in enumerate(YOLO_CLASS_NAMES)}