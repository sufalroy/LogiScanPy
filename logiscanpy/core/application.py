import logging
from typing import Dict

import cv2
from ultralytics import YOLO

from logiscanpy.core.object_counter import ObjectCounter
from logiscanpy.utility.calibration import calibrate_region
from logiscanpy.utility.publisher import Publisher
from logiscanpy.utility.video_capture import RtspVideoCapture, VideoCapture

logger = logging.getLogger(__name__)

TARGET_RESOLUTION = (640, 640)


class LogiScanPy:
    """LogiScanPy class for object detection and counting."""

    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.model = None
        self.video_capture = None
        self.video_writer = None
        self.object_counter = None
        self.publisher = None

    def initialize(self) -> bool:
        """Initialize the LogiScanPy instance.

        Returns:
            bool: True if initialization is successful, False otherwise.
        """
        logger.info("Initializing LogiScanPy...")

        logger.debug("Loading YOLOv8 model: %s", self.config["weights"])
        self.model = YOLO(self.config["weights"])

        logger.debug("Opening video source: %s", self.config["video"])
        self.video_capture = (
            RtspVideoCapture(self.config["video"])
            if self.config["rtsp"]
            else VideoCapture(self.config["video"])
        )

        if not self.video_capture.is_opened():
            logger.error("Failed to open video source: %s", self.config["video"])
            return False

        if self.config["save"]:
            logger.debug("Creating output video file: %s", self.config["output"])
            self.video_writer = cv2.VideoWriter(
                self.config["output"],
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                TARGET_RESOLUTION,
            )

        frame = self.video_capture.read()
        if frame is None:
            logger.error("Failed to read first frame from video source")
            return False

        frame = cv2.resize(frame, TARGET_RESOLUTION)

        logger.debug("Calibrating region of interest...")
        polygon_vertices = calibrate_region(frame)

        logger.debug("Initializing object counter")
        self.object_counter = ObjectCounter()
        self.object_counter.set_args(
            view_img=self.config["show"],
            reg_pts=polygon_vertices,
            classes_names=self.model.names,
        )

        logger.debug("Initializing publisher")
        self.publisher = Publisher()

        logger.info("Initialization completed successfully")
        return True

    def run(self) -> None:
        """Run the object detection and counting process."""
        logger.info("Starting video processing...")
        previous_counts: Dict[str, int] = {}

        while True:
            frame = self.video_capture.read()
            if frame is None:
                logger.info("Video frame is empty or video processing has been successfully completed.")
                break

            frame = cv2.resize(frame, TARGET_RESOLUTION)
            tracks = self.model.track(
                frame,
                persist=True,
                show=False,
                classes=[self.config["class_id"]],
                verbose=False,
                conf=self.config["confidence"],
                tracker="bytetrack.yaml",
            )

            frame = self.object_counter.start_counting(frame, tracks)
            self.publish_counts(previous_counts)

            if self.config["save"]:
                self.video_writer.write(frame)

    def publish_counts(self, previous_counts: Dict[str, int]) -> None:
        """Publish object counts to a message broker.

        Args:
            previous_counts (Dict[str, int]): Previous object counts.
        """
        class_wise_count = self.object_counter.get_class_wise_count()
        for class_name, count in class_wise_count.items():
            if class_name not in previous_counts or count > previous_counts[class_name]:
                logger.debug("Publishing count for %s: %d", class_name, count)
                self.publisher.publish_message(class_name, count)
                previous_counts[class_name] = count

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources...")

        self.video_capture.release()

        if self.config["save"]:
            self.video_writer.release()

        cv2.destroyAllWindows()
        self.publisher.close_connection()

        logger.info("Cleanup completed successfully")

    def run_app(self) -> None:
        """Run the LogiScanPy application."""
        logger.info("Starting LogiScanPy application")

        if not self.initialize():
            logger.error("Initialization failed, exiting application")
            return

        try:
            self.run()
        finally:
            self.cleanup()

        logger.info("LogiScanPy application completed successfully")
