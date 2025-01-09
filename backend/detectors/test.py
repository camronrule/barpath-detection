from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

import supervision as sv                     # annotate video

# Load the YOLO11 model
model = YOLO("detection-best.pt")

# Open the video file
video_path = "IMG_6650.MOV"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Define the video input (change 'video.mp4' to your video file or use 0 for a webcam)
video_path_out = 'out.mp4'

ret, frame = cap.read()
h, w, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(
    *'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (w, h))

count = 1

# Loop through the video frames
while ret:

    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, verbose=False, classes=[1])

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        if (count > 100 and count < 150):
            track.append((float(x), float(y)))  # x, y center point

        if (len(track) > 0):
            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(
                230, 230, 230), thickness=10)

    out.write(annotated_frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()
    count += 1


# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
