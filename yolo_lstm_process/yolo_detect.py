import os
import sys
import argparse
import glob
import time
import torch

import torch.nn as nn
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

SEQ_LEN = 16
MAX_OBJ = 3
seq_buffer = deque(maxlen=SEQ_LEN)

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()


# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

import torch.nn as nn

class ActionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
INPUT_SIZE = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 3

lstm_model = ActionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
device = "cuda" if torch.cuda.is_available() else "cpu"
lstm_model.load_state_dict(torch.load(r"D:\NCKH_YOLO+LSTM\Version7\my_model\best_lstm.pth", map_location=device))
lstm_model.to(device)
lstm_model.eval()

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

# Set bounding box colors 
bbox_colors = [
    (0, 0, 255),    # Đỏ (Red)
    (0, 255, 255),  # Vàng (Yellow)
    (255, 0, 0)     # Xanh dương (Blue)
]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop
while True:
    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder': # If source is image or image folder, load the image using its filename
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video': # If source is a video, load next frame from video file
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb': # If source is a USB camera, grab frame from camera
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # ===== LẤY FEATURE CHO LSTM (CHỈ LẤY 1 OBJECT) =====
    if len(detections) > 0:
        det = detections[0]

        xyxy = det.xyxy[0].cpu().numpy()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(det.cls[0].item())

        h, w, _ = frame.shape
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h

        frame_features = [x_center, y_center, classidx]
    else:
        frame_features = [0, 0, 0]
    print("One frame feature:", frame_features)

    seq_buffer.append(frame_features)

    # Calculate and draw framerate (if using video, USB, or Picamera source)
    if source_type == 'video' or source_type == 'usb':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate

    # default
    action_text = "No detect"

    if len(seq_buffer) == SEQ_LEN:
        input_seq = np.array(seq_buffer)
        print("DEBUG shape:", input_seq.shape)  

        input_seq = np.expand_dims(input_seq, axis=0)

        with torch.no_grad():
            lstm_out = lstm_model(torch.tensor(input_seq).float().to(device))
            pred = torch.argmax(lstm_out, dim=1).item()
        print("Sequence:\n", input_seq)
        if pred == 1:
            action_text = "Violence"
        
        elif pred == 2:
            action_text = "Weapon"

    # set màu theo action
    if action_text == "Violence":
        box_color = (0, 0, 255)
    elif action_text == "Weapon":
        box_color = (255, 0, 0)
    else:
        box_color = (0, 255, 255)
    for det in detections:
        conf = det.conf[0].item()
        if conf < min_thresh:
            continue
        
        xyxy = det.xyxy[0].cpu().numpy()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(det.cls[0].item())
        label = labels[classidx]

        #  màu theo class
        if label.lower() == "violence":
            color = (0, 0, 255)      # đỏ
        elif label.lower() == "weapon":
            color = (255, 0, 0)      # xanh dương
        else:
            color = (0, 255, 0)      # xanh lá

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        text = f"{label}: {conf:.2f}"
        cv2.putText(frame, text, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f'Actions: {action_text}', (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
    cv2.imshow('YOLO detection results',frame) # Display image
    if record: recorder.write(frame)

    # If inferencing on individual images, wait for user keypress before moving to next image. Otherwise, wait 5ms before moving to next frame.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)
    
    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()