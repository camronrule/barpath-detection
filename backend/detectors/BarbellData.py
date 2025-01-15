from collections import defaultdict
from json import loads
from typing import Any, Dict, List

import pandas as pd
from .BarbellPhase import BarbellPhase
from statistics import fmean


class BarbellData:

    REP_COMPLETED = False

    # number of frames to average velocity over
    AVG_VELOCITY_OVER_FRAMES = 7

    def __init__(self):
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

        self.phase = BarbellPhase.RACKED
        self.lift = None  # not known yet, will be updated before barbell tracking starts

        # for each tracker_id, store the x,y coordinates of each rep
        # reps_history = {tracker_id : [rep_num (int): [(x: (float), y: (float)), (x,y), (x,y), ...]]}
        self.reps_history = defaultdict(lambda: defaultdict(list))
        self.rep_num = 0

    def get_v_x(self) -> float:
        """Get the average of the barbell's X velocity over a constant number of frames

        Returns:
            float: The average X velocity over AVG_VELOCITY_OVER_FRAMES frames
        """
        return self.get_mean(self.velocities_x, self.AVG_VELOCITY_OVER_FRAMES)

    def get_v_y(self) -> float:
        """Get the average of the barbell's Y velocity over a constant number of frames

        Returns:
            float: The average Y velocity over AVG_VELOCITY_OVER_FRAMES frames
        """
        return self.get_mean(self.velocities_y, self.AVG_VELOCITY_OVER_FRAMES)

    def increment_rep_num(self) -> None:
        self.rep_num += 1

    def get_rep_num(self) -> int:
        """Get the most recent rep that was started

        Returns:
            int: The rep
        """
        return self.rep_num

    def set_lift_type(self, lift: str) -> None:
        """Set the lift type from the lift classifier

        Args:
            lift (str): The lift
        """
        assert lift in ["Squat", "Bench", "Deadlift"], "Unknown lift type"
        self.lift = lift

    def get_lift_type(self) -> str:
        """Return the lift type

        "Squat", "Bench", or "Deadlift". 

        Returns:
            str: _description_
        """
        assert self.lift != None, "BarbellData initialized before lift has been classified."
        return self.lift

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

    def get_json_from_data(self) -> List[Dict[str, Any]]:
        df = pd.DataFrame({
            "Frame": self.frame_indices,
            # identify lift type
            "Lift": [self.lift] * len(self.frame_indices),
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
