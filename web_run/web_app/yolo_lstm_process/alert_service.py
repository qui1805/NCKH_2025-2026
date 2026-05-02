from django.utils import timezone
import os
import requests

from .ai_config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def send_telegram_message(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML"
        }

        response = requests.post(url, data=data, timeout=10)
        print("[TELEGRAM MESSAGE]", response.status_code, response.text)

    except Exception as e:
        print("[TELEGRAM MESSAGE ERROR]", e)


def send_telegram_photo(image_path, caption=""):
    try:
        if not image_path or not os.path.exists(image_path):
            print("[WARN] Không tìm thấy ảnh:", image_path)
            return False

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

        with open(image_path, "rb") as img:
            files = {"photo": img}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption,
                "parse_mode": "HTML"
            }

            response = requests.post(url, files=files, data=data, timeout=20)
            print("[TELEGRAM PHOTO]", response.status_code, response.text)

            return response.status_code == 200

    except Exception as e:
        print("[TELEGRAM PHOTO ERROR]", e)
        return False


def send_telegram_video(video_path, caption=""):
    try:
        if not video_path or not os.path.exists(video_path):
            print("[WARN] Không tìm thấy video:", video_path)
            return False

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"

        with open(video_path, "rb") as vid:
            files = {"video": vid}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption,
                "parse_mode": "HTML"
            }

            response = requests.post(url, files=files, data=data, timeout=60)
            print("[TELEGRAM VIDEO]", response.status_code, response.text)

            return response.status_code == 200

    except Exception as e:
        print("[TELEGRAM VIDEO ERROR]", e)
        return False


def send_event_telegram(event):
    if event is None:
        print("[WARN] event = None, không gửi Telegram")
        return

    try:
        local_time = timezone.localtime(event.timestamp)
        time_str = local_time.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        time_str = "Không xác định"

    event_type = getattr(event, "event_type", "Không xác định")
    confidence = getattr(event, "confidence", 0)

    caption = f"""
🚨 <b>[CẢNH BÁO] - Phát hiện hành vi nguy hiểm</b>

THÔNG BÁO CẢNH BÁO TỪ HỆ THỐNG GIÁM SÁT

Hệ thống đã phát hiện một sự kiện bất thường cần được kiểm tra.

Hành động: {event_type}
Độ tin cậy: {confidence:.2f}
Thời gian: {time_str}

Tệp đính kèm:
- Ảnh cảnh báo
- Đoạn video liên quan
""".strip()

    try:
        image_sent = False

        if getattr(event, "image", None):
            image_path = event.image.path
            if os.path.exists(image_path):
                image_sent = send_telegram_photo(image_path, caption)

        if not image_sent:
            send_telegram_message(caption)

        if getattr(event, "clip", None):
            clip_path = event.clip.path
            if os.path.exists(clip_path):
                send_telegram_video(
                    clip_path,
                    caption="🎥 <b>Đoạn video liên quan đến sự kiện</b>"
                )

        print("✅ Đã gửi Telegram cảnh báo!")

    except Exception as e:
        print("[TELEGRAM EVENT ERROR]", e)