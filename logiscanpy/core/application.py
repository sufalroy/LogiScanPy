import logging
from typing import Dict

import cv2
from ultralytics import YOLO

from logiscanpy.core import object_counter
from logiscanpy.utility.calibration import calibrate_region
from logiscanpy.utility.publisher import Publisher
from logiscanpy.utility.video_capture import RtspVideoCapture, VideoCapture

logger = logging.getLogger(__name__)

TARGET_RESOLUTION = (640, 480)


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

        if self.config["rtsp"]:
            self.video_capture = RtspVideoCapture(self.config["video"])
        else:
            self.video_capture = VideoCapture(self.config["video"])

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

        logger.debug("Reading first frame from video source")
        frame = self.video_capture.read()

        if frame is None:
            logger.error("Failed to read first frame from video source")
            return False

        frame = cv2.resize(frame, TARGET_RESOLUTION)

        logger.debug("Calibrating region of interest...")
        polygon_vertices = calibrate_region(frame)

        logger.debug("Initializing object counter")
        self.object_counter = object_counter.ObjectCounter()
        self.object_counter.set_args(
            view_img=self.config["show"],
            reg_pts=polygon_vertices,
            count_reg_color=(0, 0, 255),
            classes_names=self.model.names,
            line_thickness=2,
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
            logger.debug("Reading frame from video source")
            frame = self.video_capture.read()
            if frame is None:
                logger.info("Video frame is empty or video processing has been successfully completed.")
                break

            logger.debug("Resizing frame to target resolution: %s", TARGET_RESOLUTION)
            frame = cv2.resize(frame, TARGET_RESOLUTION)

            logger.debug("Running object detection and tracking")
            tracks = self.model.track(
                frame,
                persist=True,
                show=False,
                classes=[self.config["class_id"]],
                verbose=False,
                conf=self.config["confidence"],
                tracker="bytetrack.yaml",
            )

            logger.debug("Counting objects in the frame")
            frame = self.object_counter.start_counting(frame, tracks)

            logger.debug("Publishing object counts")
            self.publish_counts(previous_counts)

            if self.config["save"]:
                logger.debug("Writing frame to output video file")
                self.video_writer.write(frame)

    def publish_counts(self, previous_counts: Dict[str, int]) -> None:
        """Publish object counts to a message broker.

        Args:
            previous_counts (Dict[str, int]): Previous object counts.
        """
        for class_name, class_wise_counts in self.object_counter.get_class_wise_count().items():
            count = class_wise_counts["in"]
            if class_name not in previous_counts or count > previous_counts[class_name]:
                logger.debug("Publishing count for %s: %d", class_name, count)
                self.publisher.publish_message(class_name, count)
                previous_counts[class_name] = count

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources...")

        logger.debug("Releasing video capture resource")
        self.video_capture.release()

        if self.config["save"]:
            logger.debug("Releasing video writer resource")
            self.video_writer.release()

        logger.debug("Destroying OpenCV windows")
        cv2.destroyAllWindows()

        logger.debug("Closing publisher connection")
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
