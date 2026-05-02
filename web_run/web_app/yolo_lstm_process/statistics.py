import os
import csv
from datetime import datetime, timedelta


VALID_ACTIONS = ["violence", "weapon"]


def save_lstm_statistics_csv(stat_rows, output_dir, video_name="video"):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{video_name}_lstm_statistics.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["STT", "Hành động", "Ngày xảy ra", "Giờ xảy ra"])

        for row in stat_rows:
            writer.writerow([
                row["stt"],
                row["action"],
                row["date"],
                row["time_range"],
            ])

    return csv_path


def build_action_statistics_from_timeline(timeline_labels, fps, start_datetime=None):
    if start_datetime is None:
        start_datetime = datetime.now()

    if fps <= 0:
        fps = 25.0

    stat_rows = []
    current_label = None
    start_idx = None
    stt = 1

    def is_action(label):
        return label in VALID_ACTIONS

    def append_row(label, start_frame_idx, end_frame_idx, stt_value):
        start_sec = start_frame_idx / fps
        end_sec = end_frame_idx / fps

        start_dt = start_datetime + timedelta(seconds=start_sec)
        end_dt = start_datetime + timedelta(seconds=end_sec)

        stat_rows.append({
            "stt": stt_value,
            "action": label,
            "date": start_dt.strftime("%Y-%m-%d"),
            "time_range": f"{start_dt.strftime('%H:%M:%S')} - {end_dt.strftime('%H:%M:%S')}",
        })

    for i, label in enumerate(timeline_labels):
        label = str(label).lower()

        if not is_action(label):
            if current_label is not None:
                append_row(current_label, start_idx, i - 1, stt)
                stt += 1
                current_label = None
                start_idx = None
            continue

        if current_label is None:
            current_label = label
            start_idx = i
        elif current_label != label:
            append_row(current_label, start_idx, i - 1, stt)
            stt += 1
            current_label = label
            start_idx = i

    if current_label is not None:
        append_row(current_label, start_idx, len(timeline_labels) - 1, stt)

    return stat_rows