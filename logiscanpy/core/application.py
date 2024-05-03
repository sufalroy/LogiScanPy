import logging
from typing import Dict, Optional

import cv2

from logiscanpy.core.pipeline import Pipeline
from logiscanpy.core.solutions.object_counter import ObjectCounter
from logiscanpy.utility.calibration import calibrate_region
from logiscanpy.utility.config import load_class_names
from logiscanpy.utility.publisher import Publisher
from logiscanpy.utility.video_capture import RtspVideoCapture, VideoCapture

_LOGGER = logging.getLogger(__name__)
_TARGET_RESOLUTION = (640, 640)
_DEFAULT_NAMES = load_class_names("../config/coco.yaml")


class LogiScanPy:
    """LogiScanPy class for object detection and counting."""

    def __init__(self, config: Dict[str, str]):
        self._config = config
        self._pipeline: Optional[Pipeline] = None
        self._video_capture: Optional[VideoCapture] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._object_counter: Optional[ObjectCounter] = None
        self._publisher: Optional[Publisher] = None
        self._window_name = "LogiScan.v.0.1.0"

    def initialize(self) -> bool:
        """Initialize the LogiScanPy instance.

        Returns:
            bool: True if initialization is successful, False otherwise.
        """
        _LOGGER.info("Initializing LogiScanPy...")

        _LOGGER.info("Initializing Detection and Tracking Pipeline")
        self._pipeline = Pipeline(
            onnx_model_path=self._config.get("weights"),
            confidence_thres=float(self._config.get("confidence")),
            iou_thres=0.45,
            is_seg=bool(self._config.get("is_seg")),
            target_class_id=int(self._config.get("class_id"))
        )
        self._pipeline.start_processes()

        _LOGGER.debug("Opening video source: %s", self._config["video"])
        self._video_capture = (
            RtspVideoCapture(self._config["video"])
            if self._config.get("rtsp", False)
            else VideoCapture(self._config["video"])
        )

        if not self._video_capture.is_opened():
            _LOGGER.error("Failed to open video source: %s", self._config["video"])
            return False

        if self._config.get("save", False):
            _LOGGER.debug("Creating output video file: %s", self._config["output"])
            self._video_writer = cv2.VideoWriter(
                self._config["output"],
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                _TARGET_RESOLUTION,
            )

        frame = self._video_capture.read()
        if frame is None:
            _LOGGER.error("Failed to read first frame from video source")
            return False

        frame = cv2.resize(frame, _TARGET_RESOLUTION)
        _LOGGER.debug("Calibrating region of interest...")
        polygons = calibrate_region(frame)

        class_names_file = self._config.get("class_names_file")
        try:
            _NAMES = load_class_names(class_names_file)
        except Exception as e:
            _LOGGER.error(f"Error reading class names file: {e}")
            _NAMES = _DEFAULT_NAMES

        _LOGGER.debug("Initializing object counter")
        self._object_counter = ObjectCounter()
        self._object_counter.set_args(
            reg_pts=polygons,
            classes_names=_NAMES,
            debug=True
        )

        _LOGGER.debug("Initializing publisher")
        self._publisher = Publisher()

        _LOGGER.info("Initialization completed successfully")
        return True

    def _reset_counts(self, previous_counts: Dict[str, int]) -> None:
        """Reset the object counts."""
        _LOGGER.info("Resetting object counts")
        self._object_counter.reset_counts()
        previous_counts.clear()

    def _publish_counts(self, previous_counts: Dict[str, int]) -> None:
        """Publish object counts to a message broker.

        Args:
            previous_counts (Dict[str, int]): Previous object counts.
        """
        class_wise_count = self._object_counter.get_class_wise_count()
        for class_name, count in class_wise_count.items():
            if class_name not in previous_counts or count > previous_counts[class_name]:
                _LOGGER.debug("Publishing count for %s: %d", class_name, count)
                self._publisher.publish_message(class_name, count)
                previous_counts[class_name] = count

    def run(self) -> None:
        """Run the object detection and counting process."""
        _LOGGER.info("Starting video processing...")
        previous_counts: Dict[str, int] = {}

        while True:
            frame = self._video_capture.read()
            if frame is None:
                _LOGGER.info(
                    "Video frame is empty or video processing has been successfully completed."
                )
                break

            frame = cv2.resize(frame, _TARGET_RESOLUTION)

            self._pipeline.put_frame(frame)
            tracks = self._pipeline.get_tracked_detections()

            frame = self._object_counter.start_counting(frame, tracks)
            self._publish_counts(previous_counts)

            if self._config.get("save", False):
                self._video_writer.write(frame)

            if self._config.get("show", False):
                cv2.namedWindow(self._window_name)
                cv2.imshow(self._window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self._reset_counts(previous_counts)

    def cleanup(self) -> None:
        """Clean up resources."""
        _LOGGER.info("Cleaning up resources...")
        self._video_capture.release()
        if self._config.get("save", False):
            self._video_writer.release()
        self._pipeline.stop_processes()
        cv2.destroyAllWindows()
        self._publisher.close_connection()
        _LOGGER.info("Cleanup completed successfully")

    def run_app(self) -> None:
        """Run the LogiScanPy application."""
        _LOGGER.info("Starting LogiScanPy application")

        if not self.initialize():
            _LOGGER.error("Initialization failed, exiting application")
            return

        try:
            self.run()
        finally:
            self.cleanup()

        _LOGGER.info("LogiScanPy application completed successfully")
