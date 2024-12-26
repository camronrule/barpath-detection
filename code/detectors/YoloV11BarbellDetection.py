from collections import defaultdict, deque
from typing import Tuple                     # hold lists of data for each tracker_id
import cv2                                   # write data to video
import pandas as pd                          # create output csv
import supervision as sv                     # annotate video
from enum import Enum                        # BarbellPhase type
from ultralytics import YOLO                 # detection, classification model
from sys import maxsize                      # length of trace annotation
from statistics import fmean                 # to calculate averages in getMean()

import os                                    # set PATH, CONF_THRESH environment vars
import numpy as np                           # img to/from byte array

from BarbellTracker import BarbellTracker    # BarbellTracker class
from BarbellPhase import BarbellPhase        # BarbellPhase type

class YoloV11BarbellDetection:
    PATH        = os.environ.get("YOLO_WEIGHTS_PATH", "best.pt")    # Path to a model
    CONF_THRESH = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.60")) # Confidence threshold


    def __init__(self, in_video_path: str, out_video_path: str) -> None:
        """Initialize a YOLO v11 image detection model with a video

        :param: video_path (str): Path to the input video.
        :param: output_path (str): Path to save the output video.
        """
        self.model = self.__load_model()

        self.in_video_path = in_video_path
        self.out_video_path = out_video_path

        self.__setup_supervision()
        self.__barbell_tracker = self.__setup_barbell_tracker()


    def __load_model(self) -> YOLO:
        """Load YOLO v11 model from path

        :return: model: The loaded YOLO v11 detection model
        """
        model = YOLO(YoloV11BarbellDetection.PATH)
        return model
    
    def __setup_supervision(self) -> None:
        """Setup the Supervision library for video annotation
        """
        self.__video_info = sv.VideoInfo.from_video_path(video_path=self.in_video_path)
        self.__frame_generator = sv.get_video_frames_generator(source_path=self.in_video_path)
        self.__thickness = sv.calculate_optimal_line_thickness(
            resolution_wh=self.__video_info.resolution_wh)
        self.__text_scale = sv.calculate_optimal_text_scale(resolution_wh=self.__video_info.resolution_wh)
        self.__sv_tracker = sv.ByteTrack(frame_rate=self.__video_info.fps)  # add history line to barball
        self.__smoother = sv.DetectionsSmoother()                 # smooth bounding boxes
        self.__box_annotator = sv.BoxCornerAnnotator(thickness=self.__thickness)           # bounding box = corners
        self.__label_annotator = sv.LabelAnnotator(text_scale=self.__text_scale, text_thickness=self.__thickness)
        self.__trace_annotator = sv.TraceAnnotator(trace_length = maxsize) # length = maxsize, draw for the whole video

    def __setup_barbell_tracker(self) -> BarbellTracker:
        """Setup the BarbellTracker class for tracking and phase detection
        """
        return BarbellTracker()

    def __update_sv(self, results: list) -> sv.Detections:
        """Update the Supervision library with detections anf filter results
        
        :param: results: List of detections from the model
        :return: detections: Filtered detections"""
        detections = sv.Detections.from_ultralytics(results)
        detections = self.__sv_tracker.update_with_detections(detections)
        detections = self.__smoother.update_with_detections(detections)

        # filter detections
        detections = detections[detections.confidence > 0.5] # only confident detection
        detections = detections[detections.class_id == 1] # only barbell

        return detections

    
    #async
    def __call__(self, frame: np.ndarray) -> sv.Detections:
        """Analyze a single frame and return any detections found

        :param: frame: Frame to be analyzed
        :return: frame: Frame with barbell bounding box and speed label
        """
        result = self.model([frame], conf=YoloV11BarbellDetection.CONF_THRESH)[0]
        detections = self.__update_sv(result) # update sv_tracker, smoother with the results
        return detections
    

    def __write_to_frame(self, detections: sv.Detections, frame: np.ndarray, label: list, formatted_strs: list) -> None:
        """Plots bounding boxes, labels, and data on frame from results and tracking
        
        :param: detections: Supvervision formatted detections in a frame
        :param: frame: The frame which will be drawn on
        :param: label: The label that will br drawn on the barbell (e.g. speed)
        :param: formatted_strs: The formatted strings that will be drawn on the frame

        :return: annotated_frame: The frame with the drawn annotations"""


        annotated_frame = self.__box_annotator.annotate(frame.copy(), detections)
        annotated_frame = self.__trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.__label_annotator.annotate(annotated_frame, detections, labels=label)

        for idx, text in enumerate(formatted_strs):
            annotated_frame = cv2.putText(img=annotated_frame, text=text, 
                                        org=(30, 150 + 100 * idx), 
                                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                                        color=(0, 0, 255), 
                                        fontScale=self.__text_scale*2,
                                        thickness=self.__thickness,
                                        lineType=cv2.LINE_AA) # use LINE_8 for quicker writing
        return annotated_frame


    def process_video(self) -> None:
        """
        Processes a video frame by frame and saves an annotated output.
        """

        cap = cv2.VideoCapture(self.in_video_path)
        barbell_tracker = self.__barbell_tracker

        with sv.VideoSink(self.out_video_path, self.__video_info) as sink:
            ret, frame = cap.read()

            for frame_idx, frame in enumerate(self.__frame_generator):
                detections = self(frame)  
                label, values = barbell_tracker.update(frame_idx, detections, self.__video_info)

                formatted_strings = [f"{key}: {value}" for key, value in values.items()]
                annotated_frame = self.__write_to_frame(detections, frame, label, formatted_strings)

                sink.write_frame(annotated_frame)
        
        cap.release()

        
    # Example usage
if __name__ == "__main__":

    input_video_path = "../../data/videos/IMG_6527.MOV"
    output_video_path = "../../data/videos/IMG_6527_OUT.MOV"

    # Initialize the object detector
    detector = YoloV11BarbellDetection(input_video_path, output_video_path)

    # Process a video and save the annotated output
    detector.process_video()