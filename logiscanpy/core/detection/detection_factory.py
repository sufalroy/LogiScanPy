from enum import Enum, auto

from logiscanpy.core.detection import Detector
from logiscanpy.core.detection.yolov8_trt import YOLOv8TRT
from logiscanpy.core.detection.yolov8_ort import YOLOv8ORT, YOLOv8SegORT
from logiscanpy.core.detection.yolov8_ov import YOLOv8OV


class Engine(Enum):
    ORT_OBJECT_DETECTION = auto()
    ORT_OBJECT_DETECTION_SEGMENTATION = auto()
    OV_OBJECT_DETECTION_SEGMENTATION = auto()
    TRT_OBJECT_DETECTION_SEGMENTATION = auto()


class DetectionFactory:
    def __init__(self):
        self.detector_map = {
            Engine.ORT_OBJECT_DETECTION: YOLOv8ORT,
            Engine.ORT_OBJECT_DETECTION_SEGMENTATION: YOLOv8SegORT,
            Engine.OV_OBJECT_DETECTION_SEGMENTATION: YOLOv8OV,
            Engine.TRT_OBJECT_DETECTION_SEGMENTATION: YOLOv8TRT,
        }

    def create_detector(
            self,
            engine: Engine,
            model_path: str,
            confidence_threshold: float,
            iou_threshold: float
    ) -> Detector:
        if engine in self.detector_map:
            detector_class = self.detector_map[engine]
            return detector_class(model_path, confidence_threshold, iou_threshold)
        else:
            raise ValueError("Unsupported engine or detector type.")
