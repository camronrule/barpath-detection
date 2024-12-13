from collections import defaultdict, deque
import supervision as sv
from ultralytics import YOLO
from sys import maxsize

'''
TODO:
- differentiate eccentric vs concentric (for max speed)
- identify top and bottom of rep
- graph velocity
- different color line for each rep
- 
'''

model_path = "../runs/detect/train/weights/best.pt"
video_path = "data/videos/IMG_6860.MOV"
video_path_out = '{}_out.mp4'.format(video_path)

model = YOLO("../runs/detect/train/weights/best.pt")

video_info = sv.VideoInfo.from_video_path(video_path=video_path)
frame_generator = sv.get_video_frames_generator(source_path=video_path)

thickness = sv.calculate_optimal_line_thickness(
    resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)


tracker = sv.ByteTrack(frame_rate=video_info.fps)
smoother = sv.DetectionsSmoother()

box_annotator = sv.BoxCornerAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
trace_annotator = sv.TraceAnnotator(trace_length = maxsize) # length = maxsize, draw for the whole video

coordinates = defaultdict(lambda: deque(maxlen=2)) # keep only two positions to get speed
plate_size_meters = 0.45 # plate diameter in meters

ema_speeds = defaultdict(lambda: None) # use exponential floating avg to smooth speed values
alpha = 0.5   # smoothing factor (0 < alpha <= 1)

max_speed = max_smoothed_speed = 0

tracker_speeds = {} # keep speeds per each tracker_id, to plot later

# dont post every single speed, to make speed readable
update_interval = 6 # update speed after this many frames
frame_counter = defaultdict(lambda: 0) # holds the frame buffer count for each tracker_id
display_speed = defaultdict(lambda: 0) # holds the current speed shown on screen for each tracker_id

with sv.VideoSink(video_path_out, video_info=video_info) as sink:
    for frame_idx, frame in enumerate(frame_generator): # frame_idx = index of frame
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        # filter detections
        detections = detections[detections.confidence > 0.5]
        detections = detections[detections.class_id == 1] # filter to only barbell
        labels = []

        if len(detections) > 0:

            # get plate size for scaling
            coords = detections.xyxy[0]
            plate_size_pixels = (abs(coords[3]-coords[1]) + abs(coords[2] - coords[1]))/2

            scaling_factor = plate_size_meters / plate_size_pixels

            id = detections.tracker_id[0]

            plate_center = detections.get_anchors_coordinates(sv.Position.CENTER)
            coordinates[id].append(plate_center[0]) # get the y coord

            if len(coordinates[id]) == 2:
                #calculate euclidian distance since last frame
                (x1, y1) = coordinates[id][0]
                (x2, y2) = coordinates[id][1]

                pixel_displacement = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

                distance_meters = pixel_displacement * scaling_factor
                time_seconds = 1 / video_info.fps
                speed_mps = distance_meters / time_seconds

                if speed_mps > max_speed: max_speed = speed_mps

                # ema smoothing
                previous_ema = ema_speeds[id]
                if previous_ema is None:
                    ema_speeds[id] = speed_mps
                else:
                    ema_speeds[id] = \
                        alpha * speed_mps + (1 - alpha) * previous_ema
                smoothed_speed = ema_speeds[id]
                if smoothed_speed > max_smoothed_speed: max_smoothed_speed = smoothed_speed
                
                frame_counter[id] += 1
                if frame_counter[id] >= update_interval:
                    #display this speed
                    display_speed[id] = smoothed_speed
                    frame_counter[id] = 0

                 # Add speed to the tracker-specific list
                if id not in tracker_speeds:
                    tracker_speeds[id] = []
                tracker_speeds[id].append((frame_idx / video_info.fps, smoothed_speed))

                labels.append(f"{display_speed[id]:.3f} m/s")

            else:
                labels.append("") # edge case -> first label, not able to calculate speed

        annotated_frame = box_annotator.annotate(frame.copy(), detections)
        annotated_frame = trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)

        sink.write_frame(annotated_frame)

print(f"Max speed found: {max_speed:.3f} m/s ({max_smoothed_speed:.3f} m/s smoothed)")

import matplotlib.pyplot as plt

# Plot speeds for each tracker
plt.figure(figsize=(10, 6))
for tracker_id, speeds in tracker_speeds.items():
    times, values = zip(*speeds)  # Separate time and speed values
    plt.plot(times, values, label=f"Tracker {tracker_id}")

plt.title("Speed Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Speed (m/s)")
plt.legend()
plt.grid()
plt.show()
