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


class Processor:
    """Processor class for performing video analytics."""

    def __init__(self, config: Dict[str, str]):
        """Initialize the Processor instance.

        Args:
            config (Dict[str, str]): Configuration dictionary.
        """
        self._config = config
        self._pipeline: Optional[Pipeline] = None
        self._video_capture: Optional[VideoCapture] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._solution: Optional[Solution] = None
        self._action: Optional[Action] = None
        self._window_name: str = f"LogiScan {self._config.get('id')}"
        self._stop_event = False

    def initialize(self) -> bool:
        """Initialize the Processor instance.

        Returns:
            bool: True if initialization is successful, False otherwise.
        """
        _LOGGER.info("[INIT] Initializing Processor...")
        _LOGGER.debug("[VIDEO] Opening video source: %s", self._config["video"])
        self._video_capture = (
            RtspVideoCapture(self._config["video"])
            if self._config.get("rtsp", False)
            else VideoCapture(self._config["video"])
        )

        if not self._video_capture.is_opened():
            _LOGGER.error("[VIDEO] Failed to open video source: %s", self._config["video"])
            return False

        if self._config.get("save", False):
            _LOGGER.debug("[OUTPUT] Creating output video file: %s", self._config["output"])
            self._video_writer = cv2.VideoWriter(
                self._config["output"],
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                _TARGET_RESOLUTION,
            )

        frame = self._video_capture.read()
        if frame is None:
            _LOGGER.error("[FRAME] Failed to read first frame from video source")
            return False

        frame = cv2.resize(frame, _TARGET_RESOLUTION)

        _LOGGER.debug("[CALIBRATION] Calibrating region of interest...")
        calibration_window_name = f"Calibration {self._config.get('id')}"
        polygons = calibrate_region(frame, calibration_window_name)

        class_names_file = self._config.get("class_names_file")
        try:
            _NAMES = load_class_names(class_names_file)
        except Exception as e:
            _LOGGER.error(f"[CONFIG] Error reading class names file: {e}")
            _NAMES = _DEFAULT_NAMES

        _LOGGER.debug("[SOLUTION] Initializing solutions...")
        self._solution = SolutionFactory.create_solution(self._config)
        self._solution.set_params(
            reg_pts=polygons,
            classes_names=_NAMES,
            debug=self._config.get("show", False),
        )

        self._action = ActionFactory.create_action(self._config.get('solution_type'))

        _LOGGER.debug("[PIPELINE] Initializing Detection and Tracking Pipeline")
        self._pipeline = Pipeline(
            engine=self._config.get("engine"),
            model_path=self._config.get("weights"),
            confidence_thres=float(self._config.get("confidence")),
            iou_thres=0.7,
        )
        self._pipeline.start_processes()

        _LOGGER.info("[INIT] Initialization completed successfully")
        return True

    def run(self) -> None:
        """Start video processing."""
        _LOGGER.info("[RUN] Starting video processing...")

        self._process_frames()

        _LOGGER.info("[STOP] Video processing stopped")

    def _process_frames(self) -> None:
        """Process video frames."""
        fps = 0
        frame_count = 0

        while not self._stop_event:
            start_time = time.time()

            frame = self._video_capture.read()
            if frame is None:
                _LOGGER.info("[FRAME] Video frame is empty or video processing has been successfully completed.")
                self._stop_event = True
                break

            frame = cv2.resize(frame, _TARGET_RESOLUTION)

            self._pipeline.put_frame(frame)
            tracks = self._pipeline.get_tracked_detections()

            frame = self._solution.process_frame(frame, tracks)
            self._action.execute(self._solution)

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
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow(self._window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._stop_event = True
                if key == ord("r"):
                    self._solution.reset()

    def cleanup(self) -> None:
        """Clean up resources."""
        _LOGGER.info("[CLEANUP] Cleaning up resources...")
        self._video_capture.release()
        if self._config.get("save", False):
            self._video_writer.release()
        self._pipeline.stop_processes()
        self._action.cleanup()
        cv2.destroyWindow(self._window_name)
        _LOGGER.info("[CLEANUP] Cleanup completed successfully")
