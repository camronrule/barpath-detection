import asyncio
from datetime import timedelta
import tempfile
import time
from typing import Tuple
import cv2                                   # write data to video
import pandas as pd                          # create output csv
import supervision as sv                     # annotate video
from ultralytics import YOLO                 # detection, classification model
from sys import maxsize                      # length of trace annotation

import os                                    # set PATH, CONF_THRESH environment vars
import numpy as np                           # img to/from byte array
import logging

from detectors.BarbellTracker import BarbellTracker   # BarbellTracker class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("BarbellDetection")


class YoloV11BarbellDetection:
    """A class to detect barbells in a video using YOLO v11 and track them using the BarbellTracker class

    Works most effectively with videos that have the entire barbell plate (specifically a kilo plate)
    visible for the entire duration of the video."""

    PATH = os.environ.get("YOLO_WEIGHTS_PATH",
                          "detectors/best.pt")       # Path to a model
    # Confidence threshold
    CONF_THRESH = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.60"))

    def __init__(self, video_path_in: str, video_path_out: str) -> None:
        """Initialize a YOLO v11 image detection model with a video

        Args:
            video_path (str): Path to the input video
        """

        self.model = self.__load_model()

        self.video_path_in = video_path_in
        self.video_path_out = video_path_out

        self.__setup_supervision()
        self.__barbell_tracker = self.__setup_barbell_tracker()

    def __load_model(self) -> YOLO:
        """Load YOLO v11 model from path

        Returns:
            YOLO: The loaded YOLO v11 detection model
        """
        logger.info(f"Loading model from {YoloV11BarbellDetection.PATH}")
        model = YOLO(YoloV11BarbellDetection.PATH)
        logger.info("Model loaded successfully")
        return model

    def __setup_supervision(self) -> None:
        """Setup the Supervision library for video annotation"""

        logger.info(f"Setting up Supervision library for video annotation")

        # utilities for processing video frames
        self.__video_info = sv.VideoInfo.from_video_path(
            video_path=self.video_path_in)
        self.__frame_generator = sv.get_video_frames_generator(
            source_path=self.video_path_in)

        # variables for writing to video, dependedent on video specifications
        self.__thickness = sv.calculate_optimal_line_thickness(
            resolution_wh=self.__video_info.resolution_wh)
        self.__text_scale = sv.calculate_optimal_text_scale(
            resolution_wh=self.__video_info.resolution_wh)

        # Supverision classes for smooth and simple video annotation
        self.__sv_tracker = sv.ByteTrack(
            frame_rate=self.__video_info.fps)  # add history line to barball
        # smooth bounding boxes while moving
        self.__smoother = sv.DetectionsSmoother()
        self.__box_annotator = sv.BoxCornerAnnotator(
            thickness=self.__thickness)  # bounding box = corners
        self.__label_annotator = sv.LabelAnnotator(
            text_scale=self.__text_scale, text_thickness=self.__thickness)  # writes the speed label to the barbell
        # length = maxsize, draw for the whole video
        self.__trace_annotator = sv.TraceAnnotator(trace_length=maxsize)

    def __setup_barbell_tracker(self) -> BarbellTracker:
        """Setup the BarbellTracker class for tracking and phase detection"""
        logger.info(
            "Setting up BarbellTracker for tracking and phase detection")
        return BarbellTracker()

    def __update_sv(self, results: list) -> sv.Detections:
        """Update Supervision library with new detections, and filter results

        Args:
            results (list): List of results from the model

        Returns:
            sv.Detections: Filtered detections output from Supervision
        """

        detections = sv.Detections.from_ultralytics(results)
        detections = self.__sv_tracker.update_with_detections(detections)
        detections = self.__smoother.update_with_detections(detections)

        # filter to only "barbell" - not "bar"
        detections = detections[detections.class_id == 1]

        return detections

    # async
    def __call__(self, frame: np.ndarray) -> sv.Detections:
        """Analyze a single frame and return results found

        Args:
            frame (np.ndarray): Frame to be analyzed

        Returns:
            sv.Detections: Supervision formatted detections in a frame
        """

        result = self.model(
            [frame],
            conf=YoloV11BarbellDetection.CONF_THRESH,
            verbose=False)[0]
        # update sv_tracker, smoother with the results
        detections = self.__update_sv(result)
        return detections

    def __write_to_frame(self, detections: sv.Detections, frame: np.ndarray, label: list, formatted_strs: list) -> np.ndarray:
        """Plots bounding boxes, labels, and data on frame from model detections and custom tracking

        Args:
            detections (sv.Detections): Supvervision formatted detections in a frame
            frame (np.ndarray): The unannotated frame which will be drawn on
            label (list): The label that will br drawn on the barbell (e.g. speed)
            formatted_strs (list): List of any other data that will be drawn on the frame

        Returns:
            np.ndarray: The frame with the annotations applied
        """

        annotated_frame = self.__box_annotator.annotate(
            frame.copy(), detections)
        annotated_frame = self.__trace_annotator.annotate(
            annotated_frame, detections)
        annotated_frame = self.__label_annotator.annotate(
            annotated_frame, detections, labels=label)

        for idx, text in enumerate(formatted_strs):
            annotated_frame = cv2.putText(img=annotated_frame,
                                          text=text,
                                          org=(30, 150 + 100 * idx),
                                          fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                          color=(0, 0, 255),
                                          fontScale=self.__text_scale*2,
                                          thickness=self.__thickness,
                                          lineType=cv2.LINE_AA)  # use LINE_8 for quicker writing
        return annotated_frame

    async def process_video(self):
        """
        """
        start = time.time()
        logger.info(f"Video processing starting")
        await asyncio.to_thread(self._process_video_in_thread)
        end = time.time()
        logger.info(
            f"Video processing finished took {str(timedelta(seconds=end-start))} to finish")

    def _process_video_in_thread(self):  # -> Tuple[str, str]:
        """Processes a video frame by frame, annotating the frames with the data from the custom barbell tracker

        Returns: 
            Tuple[str, str]: The output video path and the output JSON str
        """

        barbell_tracker = self.__barbell_tracker   # custom barbell tracker
        # temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        with sv.VideoSink(self.video_path_out, self.__video_info) as sink:

            # get detections by calling the model on each frame
            # get velocity data by passing detections to the barbell tracker
            # format velocty data into formatted_strings, then append to
            # frame and write to output video
            for frame_idx, frame in enumerate(self.__frame_generator):
                detections = self(frame)
                label, values = barbell_tracker.update(
                    frame_idx, detections, self.__video_info)

                formatted_strings = [
                    f"{key}: {value}" for key, value in values.items()]
                annotated_frame = self.__write_to_frame(
                    detections, frame, label, formatted_strings)

                sink.write_frame(annotated_frame)

        # return self.video_path_out, barbell_tracker.get_json_from_data()
        return


'''def main() -> str:
    input_video_path = "../../data/videos/IMG_6527.MOV"
    output_video_path = "../../data/videos/IMG_6527_OUT.MOV"

    # Initialize the object detector
    detector = YoloV11BarbellDetection(input_video_path, output_video_path)

    # Process a video and save the annotated output
    data = detector.process_video()
    return data


if __name__ == "__main__":
    main()
'''
