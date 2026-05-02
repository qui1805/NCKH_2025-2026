from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

YOLO_MODEL_PATH = PROJECT_ROOT / "weights" / "best.pt"
LSTM_MODEL_PATH = PROJECT_ROOT / "weights" / "best_lstm.pth"

WINDOW_SIZE = 32
SEQ_LEN = 16
FRAME_FEATURE_DIM = 6
LSTM_INPUT_DIM = WINDOW_SIZE * FRAME_FEATURE_DIM

LSTM_CLASS_NAMES = ["non_violence", "violence", "weapon"]
NUM_CLASSES = 3

CONF_THRES = 0.75
IMGSZ = 640, 480
VIOLENCE_THRES = 0.7
WEAPON_THRES = 0.7

LSTM_ALERT_PROB_THRES = 0.75

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_PATH = PROJECT_ROOT / "media" / "test_outputs"

TELEGRAM_BOT_TOKEN = "8458378313:AAFYUfemCZYsIr-5vKvKqlmhuFN9oDnnpy8"
TELEGRAM_CHAT_ID = "6201286783"