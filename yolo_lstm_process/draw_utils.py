import cv2

def get_color(label_name):
    label_name = str(label_name).lower()
    if "violence" in label_name or "weapon" in label_name:
        return (0, 0, 255)   # đỏ (BGR)
    return (255, 0, 0)       # xanh dương (BGR)


def draw_yolo_boxes(frame, result, yolo_names, lstm_label):
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    lstm_label = str(lstm_label).lower()

    # chỉ vẽ khi LSTM chốt violence hoặc weapon
    if lstm_label not in ["violence", "weapon"]:
        return frame

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy()

    track_ids = None
    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        conf = float(confs[i])
        cls_id = int(clss[i])

        yolo_label = str(yolo_names.get(cls_id, str(cls_id)))
        yolo_label_lower = yolo_label.lower()

        # chỉ vẽ bbox đúng với nhãn LSTM hiện tại
        if lstm_label == "violence" and "violence" not in yolo_label_lower:
            continue
        if lstm_label == "weapon" and "weapon" not in yolo_label_lower:
            continue

        color = (0, 0, 255)  # đỏ hết khi có cảnh báo

        if track_ids is not None:
            text = f"ID:{int(track_ids[i])} {yolo_label} {conf:.2f}"
        else:
            text = f"ID:- {yolo_label} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame


def draw_lstm_status(frame, lstm_label, lstm_conf, fps_value, frame_idx):
    alarm = str(lstm_label).lower() in ["violence", "weapon"]

    status_color = (0, 0, 255) if alarm else (255, 0, 0)   # đỏ nếu alarm, không thì xanh dương
    fixed_color = (255, 0, 0)                               # FPS + Frame luôn xanh dương

    text1 = f"LSTM Prediction: {lstm_label}"
    text2 = f"Confidence: {lstm_conf:.3f}"
    text3 = f"FPS: {fps_value:.2f}"
    text4 = f"Frame: {frame_idx}"

    cv2.putText(frame, text1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    cv2.putText(frame, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, text3, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fixed_color, 2)
    cv2.putText(frame, text4, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fixed_color, 2)

    return frame