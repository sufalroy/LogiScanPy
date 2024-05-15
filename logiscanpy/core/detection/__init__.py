from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    def __init__(self, model_path: str, confidence_threshold: float, iou_threshold: float):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Abstract method to be implemented by derived classes.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Returns:
            np.ndarray: Array of detections in the format [x1, y1, x2, y2, score, class_id].
        """
        pass
