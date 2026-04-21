# alert_service.py
import smtplib
import os
import mimetypes
from email.message import EmailMessage


# ===== CONFIG GMAIL =====
SENDER_EMAIL = "vanqui01805@gmail.com"
APP_PASSWORD = "euqe lmke nhrk mjey"
RECEIVER_EMAIL = "6451020052@st.utc2.edu.vn"


def send_gmail_alert(payload):
    """
    payload từ clip_buffer:
    {
        camera_id,
        label,
        confidence,
        timestamp,
        snapshot_path,
        video_path
    }
    """

    subject = f"[CẢNH BÁO] {payload['label']} - {payload['camera_id']}"

    body = f"""
CẢNH BÁO HỆ THỐNG

Camera: {payload['camera_id']}
Nhãn: {payload['label']}
Độ tin cậy: {payload['confidence']:.3f}
Thời gian: {payload['timestamp']}

Đã đính kèm:
- Snapshot
- Video
"""

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.set_content(body)

    # ===== attach file =====
    files = [
        payload.get("snapshot_path"),
        payload.get("video_path")
    ]

    for file_path in files:
        if file_path and os.path.exists(file_path):
            ctype, encoding = mimetypes.guess_type(file_path)
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"

            maintype, subtype = ctype.split("/", 1)

            with open(file_path, "rb") as f:
                msg.add_attachment(
                    f.read(),
                    maintype=maintype,
                    subtype=subtype,
                    filename=os.path.basename(file_path)
                )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)

        print("📧 Gmail sent successfully!")

    except Exception as e:
        print("❌ Gmail error:", e)