from enum import Enum, auto

from logiscanpy.core.detection import Detector
from logiscanpy.core.detection.yolov8 import YOLOv8


class Model(Enum):
    YOLOv8 = auto()


class DetectionFactory:
    def __init__(self):
        self.detector_map = {
            Model.YOLOv8: YOLOv8,
        }

    def create_detector(
            self,
            model: Model,
            model_path: str,
            confidence_threshold: float,
            iou_threshold: float,
    ) -> Detector:
        if model in self.detector_map:
            detector = self.detector_map[model]
            return detector(model_path, confidence_threshold, iou_threshold)
        else:
            raise ValueError("Unsupported object detection model type.")
