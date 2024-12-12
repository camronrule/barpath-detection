''' Model training ::
from ultralytics import YOLO

# Build a new model
model = YOLO("yolo11m.yaml")


# Train the model
train_results = model.train(
    data="data/CV.v15i.yolov11/data.yaml",  # path to dataset YAML
    epochs=1000,  # number of training epochs
    imgsz=640,  # training image size
    device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)
'''

import os

from ultralytics import YOLO
import cv2

model_path = os.path.join('..', 'runs', 'detect', 'train5', 'weights', 'best.pt')
model = YOLO(model_path)

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'IMG_7472.MOV')

cap = cv2.VideoCapture(video_path)

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video processing completed.")
        break

    # Perform detection
    results = model(frame)

    # Annotate frame with detections
    annotated_frame = results[0].plot()  # Adjust based on your YOLO framework

    # Display the frame
    cv2.imshow("YOLOv11 Detection", annotated_frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
