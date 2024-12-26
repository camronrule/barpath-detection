from typing import Tuple
from BarbellPhase import BarbellPhase        # BarbellPhase type
from enum import Enum                        # BarbellPhase type


from collections import defaultdict, deque   # hold lists of data for each tracker_id
import cv2                                   # write data to video
import pandas as pd                          # create output csv
import supervision as sv                     # annotate video
from sys import maxsize                      # length of trace annotation
from statistics import fmean                 # to calculate averages in get_mean()



class BarbellTracker:

    alpha = 0.2                              # EMA smoothing factor (0 < alpha <= 1). Larger values -> less smoothing

    SPEED_UPDATE_INTERVAL = 6
    REP_COMPLETED = False
    PHASE_BUFFER_LENGTH = 5
    LAST_FRAME_PHASE_CHANGED = 0
    AVG_VELOCITY_OVER_FRAMES = 7
    FRAMES_BETWEEN_PHASE_CHANGE = 10

    PLATE_SIZE_METERS = 0.45                 # plate diameter in meters
    
    
    def __init__(self):
        self.coordinates = defaultdict(lambda: deque(maxlen=2))
        self.ema_speeds = defaultdict(lambda: None)
        self.frame_counter = defaultdict(lambda: 0)
        self.display_speed = defaultdict(lambda: 0)

        self.frame_indices = []
        self.x_norms = []
        self.y_norms = []
        self.delta_x_outs = []
        self.delta_y_outs = []
        self.speeds = []
        self.velocities_x = []
        self.velocities_y = []
        self.accelerations = []
        
        self.phase_data = {
            "RACKED": {"x": [], "y": []},
            "UNRACKING": {"x": [], "y": []},
            "TOP": {"x": [], "y": []},
            "ECCENTRIC": {"x": [], "y": []},
            "BOTTOM": {"x": [], "y": []},
            "CONCENTRIC": {"x": [], "y": []},
            "RACKING": {"x": [], "y": []}
        }
        self.phase = BarbellPhase.RACKED # initial phase = RACKED


    def get_mean(self, data, length=None):
        if length == 0 or len(data) == 0:
            return 0
        elif length > len(data):
            length = len(data)
        elif length is None:
            length = len(data)
        return fmean(data[-length:])
    
    def update(self, frame_idx, detections, video_info) -> Tuple[list, BarbellPhase, float, float]:
        labels = [] # will hold either BB speed or "", to be appended to the bounding box

        if len(detections) > 0: # if a detection was found

            # get plate size for scaling
            coords = detections.xyxy[0] # [x1, y1, x2, y2]
            # bounding box should be a square, so we can 
            # average the height and width as an safeguard
            plate_size_pixels = self.calculate_plate_size(coords)
            scaling_factor = self.calculate_scaling_factor(plate_size_pixels) # meters per pixel   

            id = detections.tracker_id[0] # id of this detection (barbell)

            plate_center = detections.get_anchors_coordinates(sv.Position.CENTER)[0] # [x, y]
            self.update_coordinates(id, plate_center) # update x,y coordinates for displacement calculation

            # edge case -> first label
            # not able to calculate values
            if len(self.coordinates[id]) < 2:
                labels.append("") # no speed to display

            # if there was a previous frame, then we 
            # we can compute speed, velocity, acceleration
            elif len(self.coordinates[id]) == 2: 

                pixel_displacement, delta_x, delta_y = self.calculate_displacement(id)
                meters_displacement, delta_x_meters, delta_y_meters = self.convert_to_meters(pixel_displacement, delta_x, delta_y, scaling_factor)

                time_seconds = 1 / video_info.fps # time that has passed since last frame, in seconds
                speed_mps = meters_displacement / time_seconds # meters per second

                # apply EMA smoothing to speeds 
                # (which bubbles down to velocity, acceleration)
                previous_ema = self.ema_speeds[id]
                if previous_ema is None:
                    self.ema_speeds[id] = speed_mps
                else:
                    self.ema_speeds[id] = self.alpha * speed_mps + (1 - self.alpha) * previous_ema
                smoothed_speed = self.ema_speeds[id]

                # calculate velocity from displacement
                # velocity = change in distance / change in time
                # unit: meters per second
                delta_t = 1 / video_info.fps
                v_x = delta_x_meters / delta_t
                v_y = delta_y_meters / delta_t

                # calculate acceleration from velocity
                # acceleration = change in velocity / change in time
                # unit: meters per second squared
                a_x = a_y = 0
                # need to be on the third frame to calculate
                # acceleration (since velocity isn't available 
                # until the second frame)
                if frame_idx > 1:
                    a_x = (v_x - self.velocities_x[-1]) / delta_t
                    a_y = (v_y - self.velocities_y[-1]) / delta_t
                    a_total = (a_x ** 2 + a_y ** 2) ** 0.5
                    self.accelerations.append(a_total)
                else: # to ensure that all data is same length
                    self.accelerations.append(None)

                # update internal data
                self.frame_indices.append(frame_idx)
                self.x_norms.append(plate_center[0] / video_info.width)
                self.y_norms.append(plate_center[1] / video_info.height)
                self.delta_x_outs.append(delta_x_meters)
                self.delta_y_outs.append(delta_y_meters)
                self.velocities_x.append(v_x)
                self.velocities_y.append(v_y)
                self.speeds.append(speed_mps)

                # decide if we need to update the displayed speed value
                # we only update every SPEED_UPDATE_INTERVAL frames
                labels.append(f"speed: {self.display_speed[id]:.2f}") 
                self.update_speed_throttle(id, smoothed_speed)

                self.phase = self.detect_phase(self.phase)

                # end elif len(coordinates[id]) == 2
            # end if detections > 0

        formatted_v_x = f"{self.get_mean(self.velocities_x, self.AVG_VELOCITY_OVER_FRAMES):+.2f}"
        formatted_v_y = f"{self.get_mean(self.velocities_y, self.AVG_VELOCITY_OVER_FRAMES):+.2f}"

        values = {
            "Phase": self.phase.name,
            "v_x": formatted_v_x,
            "v_y": formatted_v_y
        }

        return labels, values


    def calculate_plate_size(self, coords):
            return (abs(coords[3] - coords[1]) + abs(coords[2] - coords[1])) / 2
    
    def calculate_scaling_factor(self, plate_size_pixels):
        return self.PLATE_SIZE_METERS / plate_size_pixels
    
    def update_coordinates(self, id, plate_center):
        self.coordinates[id].append(plate_center)
    
    def calculate_displacement(self, id):
        #calculate euclidian distance since last the last frame with previous coordinates
        (x1, y1) = self.coordinates[id][0]
        (x2, y2) = self.coordinates[id][1]
        # change in x, y (in pixels)
        delta_x = (x2 - x1)
        delta_y = (y2 - y1)
        # total displacement of x, y combined
        pixel_displacement = (delta_x ** 2 + delta_y ** 2) ** 0.5
        return pixel_displacement, delta_x, delta_y
    
    # change in x, y (in meters)
    def convert_to_meters(self, pixel_displacement, delta_x, delta_y, scaling_factor):
        delta_x_meters = delta_x * scaling_factor
        delta_y_meters = delta_y * scaling_factor
        meters_displacement = pixel_displacement * scaling_factor
        return meters_displacement, delta_x_meters, delta_y_meters
    
    def update_speed_throttle(self, id, smoothed_speed):
        self.frame_counter[id] += 1
        if self.frame_counter[id] >= self.SPEED_UPDATE_INTERVAL:
            #display this speed
            self.display_speed[id] = smoothed_speed
            self.frame_counter[id] = 0 # reset throttle count


    def detect_phase(self, phase):
        if self.frame_indices[-1] < self.PHASE_BUFFER_LENGTH:
            return phase

        self.phase_data[phase.name]["x"].append(self.x_norms[-1])
        self.phase_data[phase.name]["y"].append(self.y_norms[-1])

        if self.frame_indices[-1] - self.LAST_FRAME_PHASE_CHANGED < self.FRAMES_BETWEEN_PHASE_CHANGE:
            return phase

        phase_holder = phase
        v_x = self.get_mean(self.velocities_x, self.AVG_VELOCITY_OVER_FRAMES)
        v_y = self.get_mean(self.velocities_y, self.AVG_VELOCITY_OVER_FRAMES)

        pos_x_norm = fmean(self.phase_data[phase.name]["x"])
        pos_y_norm = fmean(self.phase_data[phase.name]["y"])

        match (phase):
            case BarbellPhase.RACKED:
                if ((v_y < -0.05) or (abs(v_x) > 0.05)) and not self.REP_COMPLETED:
                    phase = BarbellPhase.UNRACKING
            case BarbellPhase.UNRACKING:
                if (abs(v_x) < 0.005) and (abs(self.x_norms[-1] - self.phase_data[BarbellPhase.RACKED.name]["x"][-1]) > 0.005):
                    phase = BarbellPhase.TOP
            case BarbellPhase.TOP:
                if (v_y > 0.04) and (abs(self.y_norms[-1] - self.phase_data[BarbellPhase.TOP.name]["y"][1]) > 0.01):
                    phase = BarbellPhase.ECCENTRIC
                elif (abs(v_x) > 0.05) and self.REP_COMPLETED:
                    phase = BarbellPhase.RACKING
            case BarbellPhase.ECCENTRIC:
                if v_y < 0.03:
                    phase = BarbellPhase.BOTTOM
            case BarbellPhase.BOTTOM:
                if v_y < -0.08:
                    phase = BarbellPhase.CONCENTRIC
            case BarbellPhase.CONCENTRIC:
                if v_y > -0.01:
                    phase = BarbellPhase.TOP
                    self.REP_COMPLETED = True
            case BarbellPhase.RACKING:
                if abs(v_x) < 0.03 and abs(v_y) < 0.03:
                    phase = BarbellPhase.RACKED

        if phase_holder is not phase:
            self.LAST_FRAME_PHASE_CHANGED = self.frame_indices[-1]

        return phase
