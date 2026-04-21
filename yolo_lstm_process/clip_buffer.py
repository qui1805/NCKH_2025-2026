# clip_buffer.py
import os
import cv2
from collections import deque
from datetime import datetime


class EventClipBuffer:
    """
    Quản lý buffer video để lưu:
    - pre-event: N giây trước sự kiện
    - post-event: M giây sau sự kiện
    """

    def __init__(self, fps, pre_seconds=3, post_seconds=7, save_dir=r"D:\NCKH_YOLO+LSTM\Version10\my_model\alerts"):
        self.fps = max(float(fps), 1.0)
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.save_dir = save_dir

        self.pre_maxlen = int(self.fps * self.pre_seconds)
        self.post_maxlen = int(self.fps * self.post_seconds)

        self.pre_buffer = deque(maxlen=self.pre_maxlen)

        self.is_recording_post = False
        self.post_frames = []
        self.event_snapshot = None
        self.event_label = None
        self.event_conf = 0.0
        self.event_time_str = None

        os.makedirs(self.save_dir, exist_ok=True)

    def update(self, frame):
        """
        Gọi ở mỗi frame để luôn giữ 5 giây gần nhất
        """
        self.pre_buffer.append(frame.copy())

        if self.is_recording_post:
            self.post_frames.append(frame.copy())

    def can_trigger(self):
        """
        Chỉ trigger khi chưa đang ghi post-event
        """
        return not self.is_recording_post

    def start_event(self, frame, label, conf):
        """
        Bắt đầu 1 sự kiện:
        - chụp snapshot
        - copy pre_buffer
        - bắt đầu thu post-event
        """
        if self.is_recording_post:
            return False

        self.is_recording_post = True
        self.post_frames = []
        self.event_snapshot = frame.copy()
        self.event_label = str(label)
        self.event_conf = float(conf)
        self.event_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # copy pre-event frames tại thời điểm trigger
        self.event_pre_frames = list(self.pre_buffer)

        return True

    def is_done(self):
        """
        Kiểm tra đã thu đủ post-event chưa
        """
        if not self.is_recording_post:
            return False
        return len(self.post_frames) >= self.post_maxlen

    def finalize_event(self, camera_id="CAMERA_HANHLANG_03DN"):
        """
        Khi đã đủ 10 giây sau:
        - lưu snapshot
        - ghép pre + post thành mp4
        - trả payload
        """
        if not self.is_recording_post:
            return None

        image_path = os.path.join(
            self.save_dir,
            f"{self.event_time_str}_{self.event_label}.jpg"
        )

        video_path = os.path.join(
            self.save_dir,
            f"{self.event_time_str}_{self.event_label}.mp4"
        )

        # lưu ảnh
        cv2.imwrite(image_path, self.event_snapshot)

        # ghép clip
        all_frames = self.event_pre_frames + self.post_frames
        video_saved = self._save_clip(all_frames, video_path)

        payload = {
            "camera_id": camera_id,
            "label": self.event_label,
            "confidence": self.event_conf,
            "timestamp": self.event_time_str,
            "snapshot_path": image_path,
            "video_path": video_path,
            "video_saved": bool(video_saved),
            "pre_frames": len(self.event_pre_frames),
            "post_frames": len(self.post_frames),
            "total_frames": len(all_frames),
        }

        self._reset_event_state()
        return payload

    def _save_clip(self, frames, out_path):
        if not frames:
            return False

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))

        if not writer.isOpened():
            return False

        for fr in frames:
            writer.write(fr)

        writer.release()
        return True

    def _reset_event_state(self):
        self.is_recording_post = False
        self.post_frames = []
        self.event_snapshot = None
        self.event_label = None
        self.event_conf = 0.0
        self.event_time_str = None
        self.event_pre_frames = []

    def get_status_text(self):
        """
        Dùng để vẽ debug lên màn hình nếu muốn
        """
        if self.is_recording_post:
            return f"Recording post-event: {len(self.post_frames)}/{self.post_maxlen}"
        return "Idle"