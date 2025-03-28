from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from json import loads

# hold lists of data for each tracker_id
from collections import defaultdict, deque
import supervision as sv                     # annotate video

from .PhaseDetector import PhaseDetector
from .BarbellData import BarbellData
from .BarbellPhase import BarbellPhase


class BarbellTracker:
    """Track a barbell from its coordinates and estimate speed, velocity, and acceleration
    """

    # EMA smoothing factor (0 < alpha <= 1). Larger values -> less smoothing
    alpha = 0.2

    # ** VARIABLES USED IN PHASE DETECTION **
    # number of frames to buffer before phase detection
    PHASE_BUFFER_LENGTH = 5
    LAST_FRAME_PHASE_CHANGED = 0             # frame index of the last phase change
    # number of frames to average velocity over
    AVG_VELOCITY_OVER_FRAMES = 7
    # number of frames to wait before changing phase
    FRAMES_BETWEEN_PHASE_CHANGE = 10

    # update displayed speed every SPEED_UPDATE_INTERVAL frames
    SPEED_UPDATE_INTERVAL = 6
    PLATE_SIZE_METERS = 0.45                 # plate diameter in meters

    def __init__(self):
        """Initialize the BarbellTracker class with default values and empty data
        """
        # does not hold recognizeable data for the user,
        # but is used to calculate other data and decide
        # when to update the displayed speed
        self.__coordinates = defaultdict(              # keep only last two coord positions to get speed
            lambda: deque(maxlen=2))
        # use exponential floating avg to smooth speed values
        self.__ema_speeds = defaultdict(lambda: None)
        # buffer count for updating displayed speed, for each ID
        self.__frame_counter = defaultdict(lambda: 0)
        # displayed speed value, for each ID
        self.__display_speed = defaultdict(lambda: 0)

        self.data = BarbellData()

    def set_lift_type(self, lift: str) -> None:
        assert lift in ["Squat", "Bench", "Deadlift"], "Unknown lift type"
        self.data.set_lift_type(lift)

    def get_lift_type(self) -> str:
        return self.data.get_lift_type()

    def update(self, frame_idx: int, detections: sv.Detections, video_info: sv.VideoInfo) -> Tuple[list, BarbellPhase, float, float]:
        """Update barbell data using the coordinates of the barbell in the current frame

        Args:
            frame_idx (int): Index of the current frame
            detections (sv.Detections): Detection data in the current frame from Supervision
            video_info (sv.VideoInfo): Video information from Supervision

        Returns:
            Tuple[list, list]: Tuple of the barbell label, and a list of any other data to be written
        """
        labels = []  # will hold either BB speed or "", to be appended to the bounding box

        if not detections.empty():  # if a detection was found

            # get plate size for scaling
            coords = detections.xyxy[0]  # [x1, y1, x2, y2]
            # bounding box should be a square, so we can
            # average the height and width as an safeguard
            plate_size_pixels = self.__calculate_plate_size(coords)
            scaling_factor = self.__calculate_scaling_factor(
                plate_size_pixels)  # meters per pixel

            id = detections.tracker_id[0]  # id of this detection (barbell)

            """TODO - update rep num if necessary
            """

            plate_center = detections.get_anchors_coordinates(sv.Position.CENTER)[
                0]  # [x, y]
            # update x,y coordinates for displacement calculation
            self.__update_coordinates(id, plate_center)

            # edge case -> first label
            # not able to calculate values
            if len(self.__coordinates[id]) < 2:
                labels.append("")  # no speed to display

            # if there was a previous frame, then we
            # we can compute speed, velocity, acceleration
            elif len(self.__coordinates[id]) == 2:

                self.__add_rep_to_data(id, plate_center)

                pixel_displacement, delta_x, delta_y = self.__calculate_displacement(
                    id)
                meters_displacement, delta_x_meters, delta_y_meters = self.__convert_to_meters(
                    pixel_displacement, delta_x, delta_y, scaling_factor)

                # time that has passed since last frame, in seconds
                time_seconds = 1 / video_info.fps
                speed_mps = meters_displacement / time_seconds  # meters per second

                # apply EMA smoothing to speeds
                # (does not apply to velocity, acceleration)
                previous_ema = self.__ema_speeds[id]
                if previous_ema is None:
                    self.__ema_speeds[id] = speed_mps
                else:
                    self.__ema_speeds[id] = self.alpha * \
                        speed_mps + (1 - self.alpha) * previous_ema
                smoothed_speed = self.__ema_speeds[id]

                # calculate velocity from displacement
                # velocity = change in distance / change in time
                # unit: meters per second
                delta_t = 1 / video_info.fps
                v_x = delta_x_meters / delta_t
                v_y = delta_y_meters / delta_t

                v_total = (v_x ** 2 + v_y ** 2) ** 0.5

                # calculate acceleration from velocity
                # acceleration = change in velocity / change in time
                # unit: meters per second squared
                a_x = a_y = 0
                # only calculate acceleration if we have previous velocity data
                if len(self.data.velocities_x) > 0 and len(self.data.velocities_y) > 0:
                    a_x = (v_x - self.data.velocities_x[-1]) / delta_t
                    a_y = (v_y - self.data.velocities_y[-1]) / delta_t
                    a_total = (a_x ** 2 + a_y ** 2) ** 0.5
                    self.data.accelerations.append(a_total)
                else:  # to ensure that all data is same length
                    self.data.accelerations.append(None)

                # update internal data
                self.data.frame_indices.append(frame_idx)
                self.data.x_norms.append(plate_center[0] / video_info.width)
                self.data.y_norms.append(plate_center[1] / video_info.height)
                self.data.delta_x_outs.append(delta_x_meters)
                self.data.delta_y_outs.append(delta_y_meters)
                self.data.velocities_x.append(v_x)
                self.data.velocities_y.append(v_y)
                self.data.total_velocities.append(v_total)
                self.data.speeds.append(speed_mps)

                # decide if we need to update the displayed speed value
                # we only update every SPEED_UPDATE_INTERVAL frames
                labels.append(f"speed: {self.__display_speed[id]:.2f}")
                self.__update_speed_throttle(id, smoothed_speed)

                self.data.phase = self.__detect_phase(self.data.phase)

                # end elif len(coordinates[id]) == 2
            # end if detections > 0

        formatted_v_x = f"{self.data.get_mean(
            self.data.velocities_x, self.data.AVG_VELOCITY_OVER_FRAMES):+.2f}"
        formatted_v_y = f"{self.data.get_mean(
            self.data.velocities_y, self.data.AVG_VELOCITY_OVER_FRAMES):+.2f}"

        values = {
            "Phase": self.data.phase.name,
            "v_x": formatted_v_x,
            "v_y": formatted_v_y,
            "rep_num": self.data.get_rep_num()
        }

        return labels, values

    def __calculate_plate_size(self, coords):
        return (abs(coords[3] - coords[1]) + abs(coords[2] - coords[1])) / 2

    def __calculate_scaling_factor(self, plate_size_pixels: float) -> float:
        """Calculate the scaling factor to convert pixels to meters

        Args:
            plate_size_pixels (float): The size of the kilo plate in pixels

        Returns:
            float: The scaling factor to convert pixels to meters
        """
        return self.PLATE_SIZE_METERS / plate_size_pixels

    def __update_coordinates(self, id: int, plate_center: list) -> None:
        """Update internal coordinates list with the center coords of this barbell

        Args:
            id (int): The ID of the tracked barbell
            plate_center (list): The x,y coordinates of the center of the barbell
        """
        self.__coordinates[id].append(plate_center)

    def __calculate_displacement(self, id: int) -> Tuple[float, float, float]:
        """Calculate the pixel displacement of the barbell from the last frame

        Args:
            id (int): The ID of the tracked barbell

        Returns:
            Tuple[float, float, float]: Total, x, and y displacement in pixels
        """
        # calculate euclidian distance since last the last frame with previous coordinates
        (x1, y1) = self.__coordinates[id][0]
        (x2, y2) = self.__coordinates[id][1]
        # change in x, y (in pixels)
        delta_x = (x2 - x1)
        delta_y = (y2 - y1)
        # total displacement of x, y combined
        pixel_displacement = (delta_x ** 2 + delta_y ** 2) ** 0.5
        return pixel_displacement, delta_x, delta_y

    # change in x, y (in meters)
    def __convert_to_meters(self, pixel_displacement: float, delta_x: float, delta_y: float, scaling_factor: float) -> Tuple[float, float, float]:
        """Convert displacement from pixels to meters

        Args:
            pixel_displacement (float): Euclidian distance between two points in pixels
            delta_x (float): Displacement in x direction in pixels
            delta_y (float): Displacement in y direction in pixels
            scaling_factor (float): Meters per pixel

        Returns:
            Tuple[float, float, float]: Total, x, and y displacement in meters
        """
        delta_x_meters = delta_x * scaling_factor
        delta_y_meters = delta_y * scaling_factor
        meters_displacement = pixel_displacement * scaling_factor
        return meters_displacement, delta_x_meters, delta_y_meters

    def __update_speed_throttle(self, id: int, smoothed_speed: float) -> None:
        """Update the displayed speed value every SPEED_UPDATE_INTERVAL frames

        Args:
            id (_type_): ID of the barbell being tracked
            smoothed_speed (_type_): The estimated smoothed speed of this barbell
        """
        self.__frame_counter[id] += 1
        if self.__frame_counter[id] >= self.SPEED_UPDATE_INTERVAL:
            # display this speed
            self.__display_speed[id] = smoothed_speed
            self.__frame_counter[id] = 0  # reset throttle count

    def __add_rep_to_data(self, id: int, center_coords: List[float]) -> None:
        """Add rep to the data.reps_history arr, if the bar should be tracked in this frame

        Args:
            center_coords List[float]: The center coordinates of the detection

        """
        current_rep = self.data.get_rep_num()
        reps = self.data.reps_history[id]
        rep = reps[current_rep]

        if self.__should_draw_line():
            self.data.reps_by_frame.append(current_rep)
            rep.append((float(center_coords[0]), float(center_coords[1])))

        else:
            self.data.reps_by_frame.append(None)  # keep index = frame

    def __should_draw_line(self) -> bool:
        """Return if the current rep is in a phase that should be drawn

        i.e., not in RACKING / UNRACKING / RACKED

        Returns:
            bool: Whether or not the current rep is in a phase that should be drawn
        """
        phase = self.data.phase
        lift = self.data.get_lift_type()

        match lift:
            case "Deadlift":
                return phase in [BarbellPhase.CONCENTRIC,
                                 BarbellPhase.ECCENTRIC,
                                 BarbellPhase.TOP,
                                 BarbellPhase.BOTTOM]
            case "Bench" | "Squat":
                return phase in [BarbellPhase.CONCENTRIC,
                                 BarbellPhase.ECCENTRIC,
                                 BarbellPhase.TOP,
                                 BarbellPhase.BOTTOM]

    def __rep_is_active(self, rep_num: int) -> bool:
        """Determine if the specified rep is currently being done

        Args:
            rep_num (int): The rep that is being referenced

        Returns:
            bool: True if the rep was still being done during the last frame, false otherwise
        """
        return self.data.rep_num == rep_num

    def write_trace_annotation(self, frame: np.ndarray) -> np.ndarray:
        """Trace the barbell path through the rep history

        Args:
            frame (np.ndarray): The frame to be written on

        Returns:
            np.ndarray: The input frame with trace annotations applied
        """
        # for each barbell ID
        for tracker_id in self.data.reps_history:
            reps_by_id = self.data.reps_history[tracker_id]

            # for the coords of each rep for this id
            for rep_num in reps_by_id:
                rep = reps_by_id[rep_num]

                # if there are any coords to write
                if len(rep) > 0:

                    # all coords from when this rep passed __should_draw_line()
                    points = np.hstack(rep).astype(np.int32).reshape(
                        (-1, 1, 2))

                    # trace color is based on if that rep is active:
                    # active -> lime green, inactive -> gray
                    color = (50, 205, 50) if self.__rep_is_active(
                        rep_num) else (173, 173, 173)

                    cv2.polylines(frame, [
                        points], isClosed=False, color=color, thickness=10, lineType=cv2.LINE_AA)
        return frame

    def __detect_phase(self, phase):
        """From the current phase, detect the next phase based on the current velocity and position

        Args:
            phase (BarbellPhase): The current phase of the barbell

        Returns:
            BarbellPhase: The phase that the barbell is in after this frame
        """

        # not enough data, return passed in phase
        if self.data.frame_indices[-1] < self.PHASE_BUFFER_LENGTH:
            return phase

        # store x,y norm values for each phase
        # then we can verify that changes in velocity are changes in position as well
        self.data.phase_data[phase.name]["x"].append(self.data.x_norms[-1])
        self.data.phase_data[phase.name]["y"].append(self.data.y_norms[-1])

        # require that a phase change must be at least
        # FRAMES_BETWEEN_PHASE_CHANGE frames apart
        if self.data.frame_indices[-1] - self.LAST_FRAME_PHASE_CHANGED < self.FRAMES_BETWEEN_PHASE_CHANGE:
            return phase

        phase_holder = phase  # save current phase for comparison later

        # DETECT PHASE HERE
        detector = PhaseDetector(self.get_lift_type())
        phase = detector.detect(phase, self.data)

        # if we detected a new phase, then update
        # the variable holding the last frame changed
        # to the most recent frame
        if phase_holder is not phase:
            self.LAST_FRAME_PHASE_CHANGED = self.data.frame_indices[-1]

        return phase

    def get_json_from_data(self) -> List[Dict[str, Any]]:
        return self.data.get_json_from_data()

    def get_reps_history(self):
        return self.data.get_reps_history()
