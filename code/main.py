'''
from ultralytics import YOLO

# Build a new model
model = YOLO("yolo11n.yaml")

# Train the model
train_results = model.train(
    data="data/CV.v15i.yolov11/data.yaml",  # path to dataset YAML
    epochs=1000,  # number of training epochs
    device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)
'''
import cv2
from ultralytics import YOLO  # Replace with your YOLO module if different

# Load your model
model = YOLO("../runs/detect/train/weights/best.pt")

# Define the video input (change 'video.mp4' to your video file or use 0 for a webcam)
video_path = "data/videos/IMG_7472.MOV"  # Replace with 0 for webcam
video_path_out = '{}_out.mp4'.format(video_path)
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
h,w, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (w,h))

threshold = 0.5

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1,y1,x2,y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1), int(x2), int(y2)), (0,255,0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 3, cv2.LINE_AA)
    
    out.write(frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()


# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()