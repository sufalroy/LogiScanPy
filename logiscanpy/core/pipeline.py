import multiprocessing as mp

import numpy as np

from logiscanpy.core.detection.detection_factory import DetectionFactory, Engine
from logiscanpy.core.tracking.tracker import ByteTrack


class DetectionProcess(mp.Process):

    def __init__(
            self,
            input_queue: mp.Queue,
            output_queue: mp.Queue,
            engine: Engine,
            model_path: str,
            confidence_threshold: float,
            iou_threshold: float,
    ):
        """
        Initializes the DetectionProcess with the given parameters.

        Args:
            input_queue (mp.Queue): The queue to get images from.
            output_queue (mp.Queue): The queue to put detections into.
            engine (Engine): The detection engine to use.
            model_path (str): The path to the model.
            confidence_threshold (float): The confidence threshold for detections.
            iou_threshold (float): The IoU threshold for detections.
        """
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.engine = engine
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.factory = DetectionFactory()

    def run(self):
        """
        Runs the detection process in a loop, processing images from the input queue.
        """
        detector = self.factory.create_detector(
            self.engine,
            self.model_path,
            self.confidence_threshold,
            self.iou_threshold
        )

        while True:
            img = self.input_queue.get()
            if img is None:
                break

            detections = detector.detect(img)

            if len(detections) > 0:
                boxes = detections[:, :4]
                scores = detections[:, 4]
                class_ids = detections[:, 5]
            else:
                boxes = np.array([])
                scores = np.array([])
                class_ids = np.array([])

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
            engine: Engine,
            model_path: str,
            confidence_thres: float,
            iou_thres: float,
    ):
        self.engine = engine
        self.model_path = model_path
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
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
            self.engine,
            self.model_path,
            self.confidence_thres,
            self.iou_thres,
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
