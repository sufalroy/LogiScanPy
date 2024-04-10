import argparse
import logging
import cv2
import os

from ultralytics import YOLO
from core import object_counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_RESOLUTION = (1280, 720)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Object Counting Script")
    parser.add_argument("--weights", type=str, default="../weights/sack_yolov8_50e.pt",
                        help="Path to YOLOv8 weights file")
    parser.add_argument("--video", type=str, default="../videos/sack.mp4",
                        help="Path to input video file")
    parser.add_argument("--output", type=str, default="../outputs/output.avi",
                        help="Path to output video file")
    parser.add_argument("--class-id", type=int, default=0,
                        help="Class ID to count (default: 0)")
    return parser.parse_args()


def process_video(args):
    """Process the video for object counting."""
    logger.info("Loading YOLOv8 model...")
    model = YOLO(args.weights)
    logger.info("Opening video file: %s", args.video)
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), "Error reading video file"

    width, height, fps = (int(cap.get(x)) for x in
                          (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                           cv2.CAP_PROP_FPS))

    line_points = [(350, 200), (900, 200)]

    classes_to_count = [args.class_id]

    logger.info("Creating output video file: %s", args.output)
    video_writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        TARGET_RESOLUTION)

    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True,
                     reg_pts=line_points,
                     count_reg_color=(0, 0, 255),
                     classes_names=model.names,
                     draw_tracks=True,
                     line_thickness=2)

    logger.info("Starting video processing...")
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            logger.info("Video frame is empty or video processing has been successfully completed.")
            break

        image = cv2.resize(image, TARGET_RESOLUTION)
        tracks = model.track(image, persist=True, show=False, classes=classes_to_count, verbose=False)
        image = counter.start_counting(image, tracks)

        video_writer.write(image)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    logger.info("Video processing completed.")


def main():
    """Entry point for the object counting script."""
    args = parse_arguments()
    process_video(args)


if __name__ == "__main__":
    main()
