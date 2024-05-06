import logging
import time
from typing import Dict, Optional

import cv2

from logiscanpy.core.actions import Action
from logiscanpy.core.actions.action_factory import ActionFactory
from logiscanpy.core.pipeline import Pipeline
from logiscanpy.core.solutions import Solution
from logiscanpy.core.solutions.solution_factory import SolutionFactory
from logiscanpy.utility.calibration import calibrate_region
from logiscanpy.utility.config import load_class_names
from logiscanpy.utility.video_capture import RtspVideoCapture, VideoCapture

_LOGGER = logging.getLogger(__name__)
_TARGET_RESOLUTION = (640, 640)
_DEFAULT_NAMES = load_class_names("../config/coco.yaml")


class LogiScanPy:
    """LogiScanPy class for performing video analytics."""

    def __init__(self, config: Dict[str, str]):
        self._config = config
        self._pipeline: Optional[Pipeline] = None
        self._video_capture: Optional[VideoCapture] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._solution: Optional[Solution] = None
        self._action: Optional[Action] = None
        self._window_name = "LogiScan.v.0.1.0"

    def initialize(self) -> bool:
        """Initialize the LogiScanPy instance.

        Returns:
            bool: True if initialization is successful, False otherwise.
        """
        _LOGGER.info("Initializing LogiScanPy...")

        _LOGGER.debug("Initializing Detection and Tracking Pipeline")
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

        _LOGGER.debug("Initializing solutions...")
        self._solution = SolutionFactory.create_solution(self._config)
        self._solution.set_params(
            reg_pts=polygons,
            classes_names=_NAMES,
            debug=self._config.get("show", False),
        )

        self._action = ActionFactory.create_action(self._config.get('solution_type'))

        _LOGGER.info("Initialization completed successfully")
        return True

    def run(self) -> None:
        """Run the object detection and counting process."""
        _LOGGER.info("Starting video processing...")

        fps = 0
        frame_count = 0

        while True:
            start_time = time.time()

            frame = self._video_capture.read()
            if frame is None:
                _LOGGER.info(
                    "Video frame is empty or video processing has been successfully completed."
                )
                break

            frame = cv2.resize(frame, _TARGET_RESOLUTION)

            self._pipeline.put_frame(frame)
            tracks = self._pipeline.get_tracked_detections()

            frame = self._solution.process_frame(frame, tracks)
            action_data = self._solution.get_action_data()

            self._action.execute(action_data)

            if self._config.get("save", False):
                self._video_writer.write(frame)

            if self._config.get("show", False):
                cv2.namedWindow(self._window_name)

                end_time = time.time()
                frame_count += 1
                elapsed_time = end_time - start_time
                if elapsed_time > 0:
                    fps = 1.0 / elapsed_time

                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(self._window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    self._solution.reset()

    def cleanup(self) -> None:
        """Clean up resources."""
        _LOGGER.info("Cleaning up resources...")
        self._video_capture.release()
        if self._config.get("save", False):
            self._video_writer.release()
        self._pipeline.stop_processes()
        self._action.cleanup()
        cv2.destroyAllWindows()
        _LOGGER.info("Cleanup completed successfully")

    def run_app(self) -> None:
        """Run the LogiScanPy application."""
        _LOGGER.info("Starting LogiScanPy application")

        if not self.initialize():
            _LOGGER.error("Initialization failed, exiting application")
            self.cleanup()
            return

        try:
            self.run()
        finally:
            self.cleanup()

        _LOGGER.info("LogiScanPy application completed successfully")
