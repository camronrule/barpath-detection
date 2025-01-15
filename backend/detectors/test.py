from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

"""200 Frames:
    0-25: Rep 0. No rep yet. 
    25 - 125: Rep 1
    125 - 200: Rep 2
    
"""


def get_rep_num(count: int):
    # return the current rep number
    if count < 25:
        return 0
    if 25 <= count <= 125:
        return 1
    if count > 125:
        return 2


def draw_line(count: int) -> bool:
    # if line should be drawn during this phase
    """Return if the current rep is in a phase that should be drawn

    i.e., not in RACKING / UNRACKING / RACKED

    Args:
        count (int): Placeholder

    Returns:
        bool: Whether or not the current rep is in a phase that should be drawn
    """
    if 25 < count < 201:
        return True
    return False


def is_active(rep_num: int, count: int) -> bool:
    """Return if rep: rep_num is currently being done in the video

    Args:
        rep_num (int): The rep being referenced
        count (int): Placeholder

    Returns:
        bool: True if rep is active
    """
    if rep_num == 0 and count < 25:
        return True
    if rep_num == 1 and 25 <= count <= 125:
        return True
    if rep_num == 2 and count > 125:
        return True
    return False


# Load the YOLO11 model
model = YOLO("detection-best.pt")

# Open the video file
video_path = "IMG_6650.MOV"
cap = cv2.VideoCapture(video_path)

# Store the track history
# reps_history = {tracker_id : [rep_num (int): [(x: (float), y: (float)), (x,y), (x,y), ...]]}
reps_history = defaultdict(lambda: defaultdict(list))


# Define the video input (change 'video.mp4' to your video file or use 0 for a webcam)
video_path_out = 'out.mp4'

ret, frame = cap.read()
h, w, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(
    *'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (w, h))

count = 1

# Loop through the video frames
while ret:

    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, verbose=False, classes=[1])

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Visualize the results on the frame
    # TODO replace this with supervision bounding box,
    # which I like better
    annotated_frame = results[0].plot(boxes=False)

    # Plot the tracks of each individual detection
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        reps_by_id = reps_history[track_id]

        current_rep = get_rep_num(count)
        rep = reps_by_id[current_rep]
        if draw_line(count):
            rep.append((float(x), float(y)))

        if (len(rep) > 0):  # if we have reps to track
            for rep_num in reps_by_id:
                if (len(reps_by_id[rep_num]) > 0):
                    points = np.hstack(reps_by_id[rep_num]).astype(np.int32).reshape(
                        (-1, 1, 2))  # all points of this rep of this id
                    # color based on if active
                    # active -> lime green,
                    # inactive -> gray
                    color = (50, 205, 50) if is_active(
                        rep_num, count) else (173, 173, 173)
                    cv2.polylines(annotated_frame, [
                        points], isClosed=False, color=color, thickness=10, lineType=cv2.LINE_AA)

        """
        Maintain frames associated with a rep
        
        reps = {tracker_id : [rep_num (int): (x: (float), y: (float))]}
        rep = reps[tracker_id] # [rep_num (int): (x: (float), y: (float))]
        
        current_rep = get_rep_num()
        
        if draw_line():
            rep[current_rep].append((float(x), float(y)))
        
        if (len(rep) > 0): # if we have reps to track
            for rep_num, coord in rep.items():
                points = np.hstack(coord).astype(np.int32).reshape((-1,1,2)) # all points of this rep of this id
                is_active(rep_num) ? color = red : color = gray              # color based on if active
                cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=10)                
        """

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
