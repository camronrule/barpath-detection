from collections import defaultdict, deque
import cv2
import pandas as pd
import supervision as sv
from enum import Enum
from ultralytics import YOLO
from sys import maxsize


'''
TODO:
- differentiate eccentric vs concentric (for max speed)
- identify top and bottom of rep
- different color line for each rep
- 
'''

REP_COMPLETED = False        # whether or not a rep has been completed
PHASE_BUFFER_LENGTH = 5      # number of frames to wait until start detecting phase, so we can gather data

LAST_FRAME_PHASE_CHANGED = 0     # keep track of the index of last frame where phase changed
AVG_VELOCITY_OVER_FRAMES = 10    # take the mean of X,Y velocity over so many frames to take care of spikes
FRAMES_BETWEEN_PHASE_CHANGE = 10 # minimum length of a phase before a new phase can be detected

phase_data = {
    "RACKED": {"x": [], "y": []},
    "UNRACKING": {"x": [], "y": []},
    "TOP": {"x": [], "y": []},
    "ECCENTRIC": {"x": [], "y": []},
    "BOTTOM": {"x": [], "y": []},
    "CONCENTRIC": {"x": [], "y": []},
    "RACKING": {"x": [], "y": []}
}


def detectPhase(phase):
    global REP_COMPLETED, LAST_FRAME_PHASE_CHANGED, \
    PHASE_BUFFER_LENGTH, AVG_VELOCITY_OVER_FRAMES,  \
    FRAMES_BETWEEN_PHASE_CHANGE

    # not enough data, return passed in phase
    if frame_index[-1] < PHASE_BUFFER_LENGTH:
        return phase
    
    # store x,y norm values for each phase
    # then we can verify that changes in velocity are changes in position as well
    phase_data[phase.name]["x"].append(x_norm[-1])
    phase_data[phase.name]["y"].append(y_norm[-1])
    
    # only change phase every few frames
    if frame_index[-1] - LAST_FRAME_PHASE_CHANGED < FRAMES_BETWEEN_PHASE_CHANGE:
        print(LAST_FRAME_PHASE_CHANGED)
        return phase 
    
    phase_holder = phase
    
    v_x = sum(velocity_x[-AVG_VELOCITY_OVER_FRAMES:]) / len(velocity_x[-AVG_VELOCITY_OVER_FRAMES:])
    v_y = sum(velocity_y[-AVG_VELOCITY_OVER_FRAMES:]) / len(velocity_y[-AVG_VELOCITY_OVER_FRAMES:])
    print(f"{phase.name}: {v_x:.3f}, {v_y:.3f}")

    pos_x_norm = sum(phase_data[phase.name]["x"]) / len(phase_data[phase.name]["x"])
    pos_y_norm = sum(phase_data[phase.name]["y"]) / len(phase_data[phase.name]["y"])

    match (phase):
        case BarbellPhase.RACKED:
            if v_y < -0.05 and not REP_COMPLETED: # moving => assume in process of unracking
                phase = BarbellPhase.UNRACKING
            # check to see if video starts unracked already, therefore we skip 
            #if abs(v_x < 0.01) and v_y > 0.0:
                #phase = BarbellPhase.CONCENTRIC

        case BarbellPhase.UNRACKING:
            # no movement in x direction, and x position is different from the rack position
            print(abs(x_norm[-1] - phase_data[BarbellPhase.RACKED.name]["x"][-1]))
            if (abs(v_x) < 0.001) and (abs(x_norm[-1] - phase_data[BarbellPhase.RACKED.name]["x"][-1]) > 0.05): 
                phase = BarbellPhase.TOP

        case BarbellPhase.TOP: # if coming from a previous rep -> redraw line
            if (v_y > 0.05) and (abs(y_norm[-1] - pos_y_norm) > 0.1): 
                phase = BarbellPhase.ECCENTRIC
            # moving in x direction, and rep has been done
            elif (abs(v_x) > 0.05) and REP_COMPLETED: # TODO check that x is moving closer to rack
                phase = BarbellPhase.RACKING
        
        case BarbellPhase.ECCENTRIC:
            if v_y < 0.03: # no longer moving down => in the hole
                phase = BarbellPhase.BOTTOM

        case BarbellPhase.BOTTOM:
            if v_y < -0.08:  # y velocity is negative when going up
                phase = BarbellPhase.CONCENTRIC

        case BarbellPhase.CONCENTRIC:
            if v_y > -0.01: # y vel is negative when going up
                phase = BarbellPhase.TOP
                REP_COMPLETED = True
        
        case BarbellPhase.RACKING:     
            if abs(v_x) < 0.03 and abs(v_y) < 0.03:
                phase = BarbellPhase.RACKED

    if phase_holder is not phase:
        LAST_FRAME_PHASE_CHANGED = frame_index[-1]
    return phase

# default -> RACKED. 
# start drawing rep at TOP phase
class BarbellPhase(Enum):
    RACKED = 1
    UNRACKING = 2
    TOP = 3
    ECCENTRIC = 4
    BOTTOM = 5
    CONCENTRIC = 6
    RACKING = 7


model_path = "../runs/detect/train/weights/best.pt"
video_path = "data/videos/IMG_6860.MOV"
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
alpha = 0.2   # smoothing factor (0 < alpha <= 1). Larger values -> less smoothing

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
velocity_x = []
velocity_y = []
acceleration = [] 

# use cv2 to write data on video
cap = cv2.VideoCapture(video_path)

# iterate through frames with Supervision
with sv.VideoSink(video_path_out, video_info=video_info) as sink:

    ret, frame = cap.read()

    phase = BarbellPhase.RACKED

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

                # calculate velocity from displacement
                # velocity = change in distance / change in time
                # meters per second
                delta_t = 1 / video_info.fps
                v_x = delta_x_meters / delta_t
                v_y = delta_y_meters / delta_t

                # calculate acceleration from velocity
                # acceleration = change in velocity / change in time
                # meters per second squared
                a_x = a_y = 0
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

                labels.append(f"speed: {display_speed[id]:.2f}") # show speed
 
                # if we have reached the throttle point (update_interval),
                # then it is time to update the speed displayed on screen
                frame_counter[id] += 1
                if frame_counter[id] >= update_interval:
                    #display this speed
                    display_speed[id] = smoothed_speed
                    frame_counter[id] = 0 # reset throttle count

                phase = detectPhase(phase) # calculate after x,y_norm have been set

                if (frame_idx > 5):
                        
                    overlay_text_1 = f"speed: {display_speed[id]:.2f}"
                    overlay_text_2 = f"v_x: {(sum(velocity_x[-5:]) / len(velocity_x[-5:])):+.2f}"
                    overlay_text_3 = f"v_y: {(sum(velocity_y[-5:]) / len(velocity_y[-5:])):+.2f}"
                    overlay_text_4 = f"acc: {(sum(acceleration[-5:]) / len(acceleration[-5:])):+.2f}"
                    overlay_text_5 = f"phase: {phase.name}"

                    cv2.putText(frame, overlay_text_1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        text_scale*2, (0, 255, 0), thickness, cv2.LINE_8)
                    cv2.putText(frame, overlay_text_2, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                        text_scale*2, (0, 255, 0), thickness, cv2.LINE_8)
                    cv2.putText(frame, overlay_text_3, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                        text_scale*2, (0, 255, 0), thickness, cv2.LINE_8)
                    cv2.putText(frame, overlay_text_4, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 
                        text_scale*2, (0, 255, 0), thickness, cv2.LINE_8)
                    cv2.putText(frame, overlay_text_5, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 
                        text_scale*2, (0, 255, 0), thickness, cv2.LINE_8)
                    

            else:
                labels.append("") # edge case -> first label, not able to calculate speed

                # append empty data


        annotated_frame = box_annotator.annotate(frame.copy(), detections)
        annotated_frame = trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)

        sink.write_frame(annotated_frame)

cap.release()

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