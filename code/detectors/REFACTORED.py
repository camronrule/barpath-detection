from collections import defaultdict, deque
from typing import Tuple   # hold lists of data for each tracker_id
import cv2                                   # write data to video
import pandas as pd                          # create output csv
import supervision as sv                     # annotate video
from enum import Enum                        # BarbellPhase type
from ultralytics import YOLO                 # detection, classification model
from sys import maxsize                      # length of trace annotation
from statistics import fmean                 # to calculate averages in getMean()

import os                                    # set PATH, CONF_THRESH environment vars
import platform                              # get best device to run model on
from torch import cuda                       # check if cuda is available
import numpy as np                           # img to/from byte array

class YoloV11BarbellDetection:
    PATH        = os.environ.get("YOLO_WEIGHTS_PATH", "best.pt")    # Path to a model
    CONF_THRESH = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.60")) # Confidence threshold

    """Initialize a YOLO v11 image detection model with a video
    """
    def __init__(self):
        self.model = self.__load_model()
        self.device = self.__get_device()
        #self.model.to(self.device)
        self.classes = self.model.names

    def __get_device(self) -> str:
        """Gets best device for your system

        :return: device: The best device to run YOLO model on
        """
        if platform.system().lower() == "darwin":
            return "mps"
        if cuda.is_available():
            return "cuda"
        return "cpu"
    
    def __load_model(self) -> YOLO:
        """Load YOLO v11 model from path

        :return: model: The loaded YOLO v11 detection model
        """
        model = YOLO(YoloV11BarbellDetection.PATH)
        return model
    
    #async
    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """Analyze a single frame and return any detections found

        :param: frame: Frame to be analyzed

        :return: frame: Frame with barbell bounding box and speed label
        :return: labels: Labels of all detections found
        """
        results = self.model([frame], conf=YoloV11BarbellDetection.CONF_THRESH)
        frame, labels = self.plot_boxes(results, frame)
        return frame, labels
    
    def plot_boxes(self, results: list, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """Plots bounding boxes and labels on frame from results
        
        :param: results: Labels and coordinates of detections in a frame
        :param: frame: The frame which will be drawn on
        
        :return: frame: Frame with bounding boxes and labels drawn on 
        :return: labels: The labels present in the frame"""
        for r in results:
            boxes = r.boxes
            labels = []
            for box in boxes:
                c = box.cls
                l = self.model.names[int(c)]
                labels.append(l)
        frame = results[0].plot()
        return frame, labels
    
    def process_video(self, video_path, output_path):
        """
        Processes a video frame by frame and saves an annotated output.

        Args:
            video_path (str): Path to the input video.
            output_path (str): Path to save the output video.
        """
        cap = cv2.VideoCapture(video_path)

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame using the __call__ method
            annotated_frame, labels = self(frame)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

        # Release resources
        cap.release()
        out.release()
        
    # Example usage
if __name__ == "__main__":
    # Initialize the object detector
    detector = YoloV11BarbellDetection()

    # Process a video and save the annotated output
    input_video_path = "../../data/videos/IMG_6527.MOV"
    output_video_path = "../../data/videos/IMG_6527_OUT.MOV"
    detector.process_video(input_video_path, output_video_path)