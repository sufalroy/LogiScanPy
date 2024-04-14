import argparse
import logging
import os

from logiscanpy.core.application import LogiScanPy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Namespace: An object containing parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Object Counting Script")
    parser.add_argument(
        "--weights",
        type=str,
        default="../weights/sack_yolov8_50e_v2.pt",
        help="Path to YOLOv8 weights file",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="../videos/sack.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--rtsp",
        type=bool,
        default=False,
        help="Video is RTSP video stream",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../outputs/output.avi",
        help="Path to output video file",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Class ID to count (default: 0)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence level (default: 0.5)",
    )
    parser.add_argument(
        "--show",
        type=bool,
        default=True,
        help="Show object counting output",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save object counting output",
    )
    return parser.parse_args()


def main():
    """Entry point for the object counting script."""
    args = parse_arguments()
    app = LogiScanPy(args)
    app.run_app()


if __name__ == "__main__":
    main()
