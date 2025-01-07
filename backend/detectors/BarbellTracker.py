from typing import Any, Dict, List, Tuple

import pandas as pd
from json import loads, dumps
from detectors.BarbellPhase import BarbellPhase        # BarbellPhase type

# hold lists of data for each tracker_id
from collections import defaultdict, deque
import supervision as sv                     # annotate video
# to calculate averages in get_mean()
from statistics import fmean


class BarbellTracker:
    """Track a barbell from its coordinates and estimate speed, velocity, and acceleration
    """

    # EMA smoothing factor (0 < alpha <= 1). Larger values -> less smoothing
    alpha = 0.2

    # ** VARIABLES USED IN PHASE DETECTION **
    REP_COMPLETED = False                    # whether or not a rep has been completed
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

        # hold specific barbell data for each frame
        # used to calculate other data (e.g. velocity, acceleration),
        # to detect the barbell phase, or is sent to the barbell detection
        # class to be annotated on the video
        self.frame_indices = []  # index of frame
        self.x_norms = []       # normalized x position of barbell center
        self.y_norms = []       # normalized y position of barbell center
        # change in x position (in meters) since last frame
        self.delta_x_outs = []
        # change in y position (in meters) since last frame
        self.delta_y_outs = []
        # smoothed speed of barbell (in meters per second)
        self.speeds = []
        # velocity in x direction (in meters per second)
        self.velocities_x = []
        # velocity in y direction (in meters per second)
        self.velocities_y = []
        self.accelerations = []  # acceleration (in meters per second squared)

        # x,y norm values for each frame that we are in each position
        # *edge case*: values are only updated in frames that a barbell is detected
        self.phase_data = {
            "RACKED": {"x": [], "y": []},
            "UNRACKING": {"x": [], "y": []},
            "TOP": {"x": [], "y": []},
            "ECCENTRIC": {"x": [], "y": []},
            "BOTTOM": {"x": [], "y": []},
            "CONCENTRIC": {"x": [], "y": []},
            "RACKING": {"x": [], "y": []}
        }
        self.phase = BarbellPhase.RACKED  # initial phase = RACKED

    def get_mean(self, data: list, length: int = None) -> float:
        """Get average of the last 'length' elements in 'data'

        Args:
            data (list): List of numerical data to average
            length (int, optional): Number of elements to average over. Defaults to len(data).

        Returns:
            int: The average of the last 'length' elements in 'data'
        """
        if length == 0 or len(data) == 0:
            return 0
        elif length > len(data):
            length = len(data)
        elif length is None:
            length = len(data)
        return fmean(data[-length:])

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

        if len(detections) > 0:  # if a detection was found

            # get plate size for scaling
            coords = detections.xyxy[0]  # [x1, y1, x2, y2]
            # bounding box should be a square, so we can
            # average the height and width as an safeguard
            plate_size_pixels = self.__calculate_plate_size(coords)
            scaling_factor = self.__calculate_scaling_factor(
                plate_size_pixels)  # meters per pixel

            id = detections.tracker_id[0]  # id of this detection (barbell)

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

                # calculate acceleration from velocity
                # acceleration = change in velocity / change in time
                # unit: meters per second squared
                a_x = a_y = 0
                # only calculate acceleration if we have previous velocity data
                if len(self.velocities_x) > 0 and len(self.velocities_y) > 0:
                    a_x = (v_x - self.velocities_x[-1]) / delta_t
                    a_y = (v_y - self.velocities_y[-1]) / delta_t
                    a_total = (a_x ** 2 + a_y ** 2) ** 0.5
                    self.accelerations.append(a_total)
                else:  # to ensure that all data is same length
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
                labels.append(f"speed: {self.__display_speed[id]:.2f}")
                self.__update_speed_throttle(id, smoothed_speed)

                self.phase = self.__detect_phase(self.phase)

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
        """Update coordinates list with the center coords of this barbell

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

    def __detect_phase(self, phase):
        """From the current phase, detect the next phase based on the current velocity and position

        Args:
            phase (BarbellPhase): The current phase of the barbell

        Returns:
            BarbellPhase: The phase that the barbell is in after this frame
        """

        # not enough data, return passed in phase
        if self.frame_indices[-1] < self.PHASE_BUFFER_LENGTH:
            return phase

        # store x,y norm values for each phase
        # then we can verify that changes in velocity are changes in position as well
        self.phase_data[phase.name]["x"].append(self.x_norms[-1])
        self.phase_data[phase.name]["y"].append(self.y_norms[-1])

        # require that a phase change must be at least
        # FRAMES_BETWEEN_PHASE_CHANGE frames apart
        if self.frame_indices[-1] - self.LAST_FRAME_PHASE_CHANGED < self.FRAMES_BETWEEN_PHASE_CHANGE:
            return phase

        phase_holder = phase  # save current phase for comparison later

        v_x = self.get_mean(self.velocities_x, self.AVG_VELOCITY_OVER_FRAMES)
        v_y = self.get_mean(self.velocities_y, self.AVG_VELOCITY_OVER_FRAMES)

        match (phase):

            case BarbellPhase.RACKED:
                # we have x or y movement and rep has not been done yet
                # (unracking only happens before the reps start) -> unracking
                if ((v_y < -0.05) or (abs(v_x) > 0.05)) and not self.REP_COMPLETED:
                    phase = BarbellPhase.UNRACKING
                # check to see if video starts unracked already, therefore we skip
                # if abs(v_x < 0.01) and v_y > 0.0:
                    # phase = BarbellPhase.CONCENTRIC

            case BarbellPhase.UNRACKING:
                # no movement in x direction, and x position is
                # different from the x of the start of the rack position
                if (abs(v_x) < 0.005) and (abs(self.x_norms[-1] - self.phase_data[BarbellPhase.RACKED.name]["x"][-1]) > 0.005):
                    phase = BarbellPhase.TOP
                # elif we missed the 'TOP' -> if large y movement, go strait to concentric

            case BarbellPhase.TOP:  # if coming from a previous rep -> redraw line
                # if moving down in the y direction, and y_norm has changed since the beginning of TOP phase
                if (v_y > 0.04) and (abs(self.y_norms[-1] - self.phase_data[BarbellPhase.TOP.name]["y"][1]) > 0.01):
                    phase = BarbellPhase.ECCENTRIC
                # moving in x direction, and rep has been done
                # TODO check that x is moving closer to rack
                elif (abs(v_x) > 0.05) and self.REP_COMPLETED:
                    phase = BarbellPhase.RACKING

            case BarbellPhase.ECCENTRIC:
                # no longer moving down => in the hole
                if v_y < 0.03:
                    phase = BarbellPhase.BOTTOM

            case BarbellPhase.BOTTOM:
                # y velocity is negative when going up
                if v_y < -0.08:
                    phase = BarbellPhase.CONCENTRIC

            case BarbellPhase.CONCENTRIC:
                # y vel is negative when going up
                if v_y > -0.01:
                    phase = BarbellPhase.TOP
                    self.REP_COMPLETED = True

            case BarbellPhase.RACKING:
                if abs(v_x) < 0.03 and abs(v_y) < 0.03:
                    phase = BarbellPhase.RACKED

        # if we detected a new phase, then update
        # the variable holding the last frame changed
        # to the most recent frame
        if phase_holder is not phase:
            self.LAST_FRAME_PHASE_CHANGED = self.frame_indices[-1]

        return phase

    def get_json_from_data(self) -> List[Dict[str, Any]]:
        df = pd.DataFrame({
            "Frame": self.frame_indices,
            "X_normalized": self.x_norms,
            "Y_normalized": self.y_norms,
            "Delta_X": self.delta_x_outs,
            "Delta Y": self.delta_y_outs,
            "Speed": self.speeds,
            "Velocity_X": self.velocities_x,
            "Velocity_Y": self.velocities_y,
            "Acceleration": self.accelerations
        })
        return loads(df.to_json(orient='records'))
