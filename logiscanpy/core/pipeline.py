import multiprocessing as mp
from typing import Union

import numpy as np

from logiscanpy.core.detector.detector import YOLOv8, YOLOv8Seg
from logiscanpy.core.tracker.tracker import ByteTrack


class DetectionProcess(mp.Process):

    def __init__(
            self,
            input_queue: mp.Queue,
            output_queue: mp.Queue,
            onnx_model: str,
            confidence_threshold: float,
            iou_threshold: float,
            is_seg: bool = True,
            target_class_id: int = 0,
    ):
        """
        Initializes the DetectionProcess with the given parameters.

        Args:
            input_queue (mp.Queue): The queue to get images from.
            output_queue (mp.Queue): The queue to put detections into.
            onnx_model (str): The path to the ONNX model.
            confidence_threshold (float): The confidence threshold for detections.
            iou_threshold (float): The IoU threshold for detections.
            is_seg (bool, optional): Whether to use YOLOv8Seg. Defaults to True.
            target_class_id (int, optional): The target class id. Defaults to 0.
        """
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.onnx_model = onnx_model
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.is_seg = is_seg
        self.target_class_id = target_class_id

    def initialize_model(self) -> Union[YOLOv8, YOLOv8Seg]:
        """
        Initializes the model based on the is_seg flag.

        Returns:
            Union[YOLOv8, YOLOv8Seg]: The initialized detector instance.
        """
        if self.is_seg:
            return YOLOv8Seg(self.onnx_model)
        else:
            return YOLOv8(self.onnx_model, self.confidence_threshold, self.iou_threshold)

    def run(self):
        """
        Runs the detection process in a loop, processing images from the input queue.
        """
        model = self.initialize_model()
        while True:
            img = self.input_queue.get()
            if img is None:
                break

            if self.is_seg:
                detections, _, _ = model.detect(
                    img,
                    class_id=self.target_class_id,
                    iou_threshold=self.iou_threshold,
                    conf_threshold=self.confidence_threshold
                )
                detections = np.array(detections)
                class_ids = np.array([det[-1] for det in detections])
                scores = np.array([det[-2] for det in detections])
                boxes = np.array([det[:4] for det in detections])

            else:
                detections = model.detect(img, [self.target_class_id])
                boxes = detections[:, :4]
                scores = detections[:, 4]
                class_ids = detections[:, 5]

            self.output_queue.put((boxes, scores, class_ids))


class TrackingProcess(mp.Process):

    def __init__(
            self,
            input_queue: mp.Queue,
            output_queue: mp.Queue,
            tracker: ByteTrack,
    ):
        """
        Initializes the TrackingProcess with the given parameters.

        Args:
            input_queue (mp.Queue): The queue to get detections from.
            output_queue (mp.Queue): The queue to put tracked detections into.
            tracker (Tracker): The tracker instance to update with detections.
        """
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.tracker = tracker

    def run(self):
        """
        Runs the tracking process in a loop, processing detections from the input queue.
        """
        while True:
            item = self.input_queue.get()
            if item is None:
                break

            boxes, scores, class_ids = item
            tracked_detections = self.tracker.update_with_detections(boxes, scores, class_ids)
            self.output_queue.put(tracked_detections)


class Pipeline:

    def __init__(
            self,
            onnx_model_path: str,
            confidence_thres: float,
            iou_thres: float,
            is_seg: bool,
            target_class_id: int
    ):
        self.onnx_model_path = onnx_model_path
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.is_seg = is_seg
        self.target_class_id = target_class_id
        self.input_queue = mp.Queue()
        self.detection_queue = mp.Queue()
        self.tracking_queue = mp.Queue()
        self.detection_process = None
        self.tracking_process = None

    def start_processes(self):
        """
        Starts the detection and tracking processes.
        """
        self.detection_process = DetectionProcess(
            self.input_queue,
            self.detection_queue,
            self.onnx_model_path,
            self.confidence_thres,
            self.iou_thres,
            is_seg=self.is_seg,
            target_class_id=self.target_class_id
        )
        tracker = ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
            minimum_consecutive_frames=1,
        )
        self.tracking_process = TrackingProcess(self.detection_queue, self.tracking_queue, tracker)
        self.detection_process.start()
        self.tracking_process.start()

    def stop_processes(self):
        """
        Stops the detection and tracking processes.
        """
        self.input_queue.put(None)
        self.detection_queue.put(None)
        self.detection_process.join()
        self.tracking_process.join()

    def get_tracked_detections(self) -> np.ndarray:
        """
        Gets the tracked detections from the tracking queue.

        Returns:
            np.ndarray: The tracked detections.
        """
        return self.tracking_queue.get()

    def put_frame(self, frame: np.ndarray):
        """
        Puts a frame into the input queue for detection.

        Args:
            frame (np.ndarray): The frame to be processed.
        """
        self.input_queue.put(frame)
