import asyncio
from collections import defaultdict
from datetime import timedelta
import sys
import time
import cv2                                   # write data to video
import pandas as pd                          # create output csv
from json import loads, dumps                # handle JSON data
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

# Different possible responses for video state
STATE_PROCESSING = "Processing"
STATE_ERROR = "Error"
STATE_FINISHED = "Finished"


class YoloV11BarbellDetection:
    """A class to detect barbells in a video using YOLO v11 and track them using the BarbellTracker class

    Works most effectively with videos that have the entire barbell plate (specifically a kilo plate)
    visible for the entire duration of the video."""

    DETECTOR_PATH = os.environ.get("DETECTOR_WEIGHTS_PATH",
                                   "detectors/detection-best.pt")
    CLASSIFIER_PATH = os.environ.get(
        "CLASSIFIER_WEIGHTS_PATH", "detectors/classification-best.pt")
    # Confidence threshold
    CONF_THRESH = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.60"))

    def __init__(self) -> None:
        """Initialize a YOLO v11 image detection model
        """
        self.barbell_detector = self.__load_detector()
        self.lift_classifier = self.__load_classifier()
        self.data = {}
        self.reps = {}
        # video_id : {"state": str, "progress": float}
        self.status = defaultdict(
            lambda: {"state": STATE_PROCESSING, "progress": "0.0"})
        self.speeds = {'preprocess': [], 'inference': [], 'postprocess': []}

    def init_video(self, video_path_in: str, video_path_out: str, video_id: int) -> None:
        """Take in a video for the YOLO v11 model

        Args:
            video_path_in (str): Path to the input video
            video_path_out (str): Path to where the output video will be saved
            video_id (int): ID of the video
        """
        assert video_id not in self.data, "Video ID already exists"

        self.video_path_in = video_path_in
        self.video_path_out = video_path_out
        self.video_id = video_id

        self.__setup_supervision()
        self.__barbell_tracker = self.__setup_barbell_tracker()

        # show that we are not done processing this video yet
        self.data[video_id] = "N/A"
        self.reps[video_id] = "N/A"

        self.update_progress(video_id, 0)
        self.update_state(video_id, STATE_PROCESSING)

    def __load_classifier(self) -> YOLO:
        """Load YOLO v11 lift classifier from path

        Returns:
            YOLO: The loaded YOLO v11 classification model
        """
        logger.info(
            f"Loading classification model from {YoloV11BarbellDetection.CLASSIFIER_PATH}")
        model = YOLO(YoloV11BarbellDetection.CLASSIFIER_PATH)
        logger.info("Classification model loaded successfully")
        return model

    def __load_detector(self) -> YOLO:
        """Load YOLO v11 barbell detector from path

        Returns:
            YOLO: The loaded YOLO v11 detection model
        """
        logger.info(
            f"Loading detection model from {YoloV11BarbellDetection.DETECTOR_PATH}")
        model = YOLO(YoloV11BarbellDetection.DETECTOR_PATH)
        logger.info("Detection model loaded successfully")
        return model

    def __setup_supervision(self) -> None:
        """Setup the Supervision library for video annotation"""

        logger.info(f"Setting up Supervision library for video annotation")

        # utilities for processing video frames
        self.__video_info = sv.VideoInfo.from_video_path(
            video_path=self.video_path_in)

        # variables for writing to video, dependedent on video specifications
        self.__thickness = sv.calculate_optimal_line_thickness(
            resolution_wh=self.__video_info.resolution_wh)
        self.__text_scale = sv.calculate_optimal_text_scale(
            resolution_wh=self.__video_info.resolution_wh)

        # Supverision classes for smooth and simple video annotation
        # smooth bounding boxes while moving
        self.__smoother = sv.DetectionsSmoother()
        self.__box_annotator = sv.BoxCornerAnnotator(
            thickness=self.__thickness)  # bounding box = corners
        self.__label_annotator = sv.LabelAnnotator(
            # writes the speed label to the barbell
            text_scale=self.__text_scale, text_thickness=self.__thickness)
        # length = maxsize, draw for the whole video
        self.__trace_annotator = sv.TraceAnnotator(trace_length=1)

    def __setup_barbell_tracker(self) -> BarbellTracker:
        """Setup the BarbellTracker class for tracking and phase detection"""
        logger.info(
            "Setting up BarbellTracker for tracking and phase detection")
        return BarbellTracker()

    def __update_sv(self, results: list) -> sv.Detections:
        """Update Supervision library with new detections

        Args:
            results (list): List of results from the model

        Returns:
            sv.Detections: Filtered detections output from Supervision
        """

        detections = sv.Detections.from_ultralytics(results)
        detections = self.__smoother.update_with_detections(detections)

        return detections

    def update_progress(self, video_id: int, progress: float) -> None:
        self.status[video_id]["progress"] = progress

    def update_state(self, video_id: int, state: str) -> None:
        self.status[video_id]["state"] = state

    def get_progress(self, video_id: int) -> float:
        return self.status[video_id]["progress"]

    def get_state(self, video_id: int) -> str:
        return self.status[video_id]["state"]

    def get_status(self, video_id: int) -> dict:
        return self.status[video_id]

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        """Analyze a single frame and return results found

        Args:
            frame (np.ndarray): Frame to be analyzed

        Returns:
            sv.Detections: Supervision formatted detections in a frame
        """
        try:
            results = self.barbell_detector.track(
                [frame],
                persist=True,  # track bounding box between frames
                conf=YoloV11BarbellDetection.CONF_THRESH,
                verbose=False,  # prevent console output
                classes=[1])  # filter detections to only Barbell - not Bar

            result = results[0]

            # save pre/post processing time, inference time
            for k, v in result.speed.items():
                self.speeds[k].append(result.speed[k])

            # update smoother with the results
            detections = self.__update_sv(result)
        except Exception as e:
            print(e)
            sys.exit()
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
        # annotated_frame = self.__trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.__label_annotator.annotate(
            annotated_frame, detections, labels=label)

        # write v_x, v_y, BarbellPhase
        for idx, text in enumerate(formatted_strs):
            annotated_frame = cv2.putText(img=annotated_frame,
                                          text=text,
                                          org=(30, 150 + 100 * idx),
                                          fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                          color=(0, 0, 255),
                                          fontScale=self.__text_scale*2,
                                          thickness=self.__thickness,
                                          lineType=cv2.LINE_AA)  # use LINE_8 for quicker writing

        # custom trace annotation
        annotated_frame = self.__barbell_tracker.write_trace_annotation(
            annotated_frame)

        return annotated_frame

    async def process_video(self, video_id: int):
        """
        """
        start = time.time()
        logger.info(f"Lift classification starting")
        await asyncio.to_thread(self.__classify_lift_in_thread)
        end = time.time()
        logger.info(
            f"Lift classification in video {video_id} finished. Took {float(timedelta(seconds=end-start).total_seconds()):.2f} seconds")

        start = time.time()
        logger.info(f"Barbell detection starting")
        await asyncio.to_thread(self._detect_barbell_in_thread)
        end = time.time()
        logger.info(
            f"Barbell detection in video {video_id} finished. Took {float(timedelta(seconds=end-start).total_seconds()):.2f} seconds")

    def __classify_lift_in_thread(self):
        """Classify the lift in the video using the YOLO v11 classifier"""
        LIFT_CONF_THRESH = 0.995
        names = self.lift_classifier.names
        try:
            for frame_idx, frame in enumerate(sv.get_video_frames_generator(source_path=self.video_path_in,
                                                                            stride=3)):
                '''
                1. get lift classification from classifier
                2. if meet threshold, update BB tracker with this info
                3. else, continue
                '''
                results = self.lift_classifier(frame, verbose=False)
                for result in results:
                    if result.probs.top1conf >= LIFT_CONF_THRESH:
                        self.__barbell_tracker.set_lift_type(
                            names[result.probs.top1])
                        logger.info(
                            f"Lift classified as: {names[result.probs.top1]}")
                        logger.info(
                            f"Lift classification took {frame_idx+1} frames to reach {result.probs.top1conf:.4f} confidence")
                        return
            else:
                raise Exception(
                    # TODO manual lift classification
                    "Not able to classify lift. Manual lift classification is WIP.")

        except Exception as e:
            print(e)
            self.update_state(self.video_id, f"{STATE_ERROR}: {e}")
            self.update_progress(self.video_id, -1)
            return

    def _detect_barbell_in_thread(self):  # -> Tuple[str, str]:
        """Processes a video frame by frame, annotating the frames with the data from the custom barbell tracker

        Returns:
            Tuple[str, str]: The output video path and the output JSON str
        """

        assert self.get_progress(
            self.video_id) >= 0, f"Error classifying video \nEnding processing of video {self.video_id}"

        barbell_tracker = self.__barbell_tracker   # custom barbell tracker

        try:
            with sv.VideoSink(self.video_path_out, self.__video_info) as sink:

                # get detections by calling the model on each frame
                # get velocity data by passing detections to the barbell tracker
                # format velocty data into formatted_strings, then append to
                # frame and write to output video
                for frame_idx, frame in enumerate(sv.get_video_frames_generator(
                        source_path=self.video_path_in)):
                    detections = self(frame)
                    label, values = barbell_tracker.update(
                        frame_idx, detections, self.__video_info)

                    formatted_strings = [
                        f"{key}: {value}" for key, value in values.items()]
                    annotated_frame = self.__write_to_frame(
                        detections, frame, label, formatted_strings)

                    sink.write_frame(annotated_frame)

                    # update progress
                    self.update_progress(
                        self.video_id, (frame_idx / self.__video_info.total_frames))

        except Exception as e:
            print(e)
            self.update_state(self.video_id, f"{STATE_ERROR}: {e}")
            self.update_progress(self.video_id, -1)
            raise e

        self.update_state(self.video_id, STATE_FINISHED)
        # should be 1 anyway, but just to be sure
        self.update_progress(self.video_id, 1.0)
        self.data[self.video_id] = barbell_tracker.get_json_from_data()
        self.reps[self.video_id] = barbell_tracker.get_reps_history()

        # print average processing time
        processing_time = ""
        for k, v in self.speeds.items():
            avg = sum(v) / len(v)
            processing_time += f"\n\t{k}: {avg:.2f}"
        logger.info(f"Average processing time (ms): {processing_time}")

        return
