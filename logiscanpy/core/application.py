import logging

import cv2
from ultralytics import YOLO

from logiscanpy.core import object_counter
from logiscanpy.utility.video_capture import RtspVideoCapture, VideoCapture
from logiscanpy.utility.publisher import Publisher

logger = logging.getLogger(__name__)

TARGET_RESOLUTION = (640, 480)


class LogiScanPy:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.video_capture = None
        self.video_writer = None
        self.object_counter = None
        self.publisher = None

    def initialize(self):
        logger.info("Loading YOLOv8 model...")
        self.model = YOLO(self.config["weights"])

        logger.info("Opening video source: %s", self.config["video"])

        if self.config["rtsp"]:
            self.video_capture = RtspVideoCapture(self.config["video"])
        else:
            self.video_capture = VideoCapture(self.config["video"])

        if not self.video_capture.is_opened():
            logger.error("Failed to open video source")
            return False

        if self.config["save"]:
            logger.info("Creating output video file: %s", self.config["output"])
            self.video_writer = cv2.VideoWriter(
                self.config["output"],
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                TARGET_RESOLUTION,
            )

        line_points = [(1, 200), (1000, 200)]

        self.object_counter = object_counter.ObjectCounter()
        self.object_counter.set_args(
            view_img=self.config["show"],
            reg_pts=line_points,
            count_reg_color=(0, 0, 255),
            classes_names=self.model.names,
            draw_tracks=False,
            line_thickness=2,
        )

        self.publisher = Publisher()

        return True

    def run(self):
        logger.info("Starting video processing...")

        previous_out_counts = {}

        while True:
            frame = self.video_capture.read()
            if frame is None:
                logger.info("Video frame is empty or video processing has been successfully completed.")
                break

            frame = cv2.resize(frame, TARGET_RESOLUTION)
            tracks = self.model.track(frame, persist=True, show=False, classes=[self.config["class_id"]], verbose=False,
                                      conf=self.config["confidence"])
            frame = self.object_counter.start_counting(frame, tracks)

            self.publish_counts(previous_out_counts)

            if self.config["save"]:
                self.video_writer.write(frame)

    def publish_counts(self, previous_out_counts):
        for class_name, counts in self.object_counter.get_class_wise_count().items():
            out_count = counts["out"]
            if class_name not in previous_out_counts or out_count > previous_out_counts[class_name]:
                self.publisher.publish_message(class_name, out_count)
                previous_out_counts[class_name] = out_count

    def cleanup(self):
        self.video_capture.release()
        if self.config["save"]:
            self.video_writer.release()
        cv2.destroyAllWindows()
        self.publisher.close_connection()

    def run_app(self):
        if not self.initialize():
            return

        try:
            self.run()
        finally:
            self.cleanup()
