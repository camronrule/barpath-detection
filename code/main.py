from collections import defaultdict, deque
import pandas as pd
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
video_path = "data/videos/IMG_6723.MOV"
video_path_out = '{}_out.mp4'.format(video_path)
output_csv_path = '{}_out.csv'.format(video_path)

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

# dont post every single speed, to make speed readable
update_interval = 6 # update speed after this many frames
frame_counter = defaultdict(lambda: 0) # holds the frame buffer count for each tracker_id
display_speed = defaultdict(lambda: 0) # holds the current speed shown on screen for each tracker_id

# for outputting data to csv
frame_index = []
x_norm = []
y_norm = []
delta_x_out = []
delta_y_out = []
speed = []
velocity = []
velocity_x = []
velocity_y = []
acceleration = [] 

with sv.VideoSink(video_path_out, video_info=video_info) as sink:
    for frame_idx, frame in enumerate(frame_generator): # frame_idx = index of frame

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        # filter detections
        detections = detections[detections.confidence > 0.5] # only confident detection
        detections = detections[detections.class_id == 1] # only barbell
        labels = [] # labels that will be written to this frame

        if len(detections) > 0: # if a barbell is detected

            # get plate size for scaling
            coords = detections.xyxy[0] # [x1, y1, x2, y2]
            plate_size_pixels = (abs(coords[3]-coords[1]) + abs(coords[2] - coords[1]))/2 # should be a square, just average height and width as a safeguard

            scaling_factor = plate_size_meters / plate_size_pixels

            id = detections.tracker_id[0] # id of this detection (barbell)

            plate_center = detections.get_anchors_coordinates(sv.Position.CENTER)[0] # [x, y]
            coordinates[id].append(plate_center) # keep x,y of the barbell to calculate displacement

            if len(coordinates[id]) == 2: # if there is a previous frame

                #calculate euclidian distance since last the last frame with previous coordinates
                (x1, y1) = coordinates[id][0]
                (x2, y2) = coordinates[id][1]

                # change in x, y (in pixels)
                delta_x = (x2-x1)
                delta_y = (y2-y1)
                pixel_displacement = (delta_x ** 2 + delta_y ** 2) ** 0.5 # total displacement of x, y combined

                # change in x, y (in meters)
                delta_x_meters = delta_x * scaling_factor
                delta_y_meters = delta_y * scaling_factor
                distance_meters = pixel_displacement * scaling_factor

                time_seconds = 1 / video_info.fps # time that has passed since last frame, in seconds
                speed_mps = distance_meters / time_seconds # meters per second

                # apply EMA smoothing to speeds (which also applies for velocity, acceleration)
                previous_ema = ema_speeds[id]
                if previous_ema is None:
                    ema_speeds[id] = speed_mps
                else:
                    ema_speeds[id] = \
                        alpha * speed_mps + (1 - alpha) * previous_ema
                smoothed_speed = ema_speeds[id]

                # if we have reached the throttle point (update_interval),
                # then it is time to update the speed displayed on screen
                frame_counter[id] += 1
                if frame_counter[id] >= update_interval:
                    #display this speed
                    display_speed[id] = smoothed_speed
                    frame_counter[id] = 0 # reset throttle count

                # calculate velocity from displacement
                # velocity = change in distance / change in time
                # meters per second
                delta_t = 1 / video_info.fps
                v_x = delta_x_meters / delta_t
                v_y = delta_y_meters / delta_t

                # calculate acceleration from velocity
                # acceleration = change in velocity / change in time
                # meters per second squared
                if frame_idx > 1: # need to be on the third frame to calc acceleration (velocity not available until second frame)
                    a_x = (v_x - velocity_x[-1]) / delta_t
                    a_y = (v_y - velocity_y[-1]) / delta_t
                    a_total = (a_x ** 2 + a_y ** 2) ** 0.5
                    acceleration.append(a_total)
                else:
                    acceleration.append(None) # to ensure that all data is same length

                # add to output data
                frame_index.append(frame_idx)
                x_norm.append(plate_center[0] / video_info.width) # x_norm = x / width
                y_norm.append(plate_center[1] / video_info.height)
                delta_x_out.append(delta_x_meters)
                delta_y_out.append(delta_y_meters)
                velocity_x.append(v_x)
                velocity_y.append(v_y)
                speed.append(speed_mps)

                labels.append(f"{display_speed[id]:.3f} m/s") # show speed

            else:
                labels.append("") # edge case -> first label, not able to calculate speed

                # append empty data
                frame_index.append(frame_idx)
                x_norm.append(None) # x / width = x_norm
                y_norm.append(None)
                delta_x_out.append(None)
                delta_y_out.append(None)
                velocity_x.append(None)
                velocity_y.append(None)
                speed.append(None)
                acceleration.append(None)

        annotated_frame = box_annotator.annotate(frame.copy(), detections)
        annotated_frame = trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)

        sink.write_frame(annotated_frame)

df = pd.DataFrame({
    "Frame": frame_index,
    "X_normalized": x_norm,
    "Y_normalized": y_norm,
    "Delta_X": delta_x_out,
    "Delta Y": delta_y_out,
    "Speed": speed,
    "Velocity_X": velocity_x,
    "Velocity_Y": velocity_y,
    "Acceleration": acceleration
})

df.to_csv(output_csv_path, index=False)