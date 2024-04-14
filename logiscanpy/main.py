import configparser
import logging
import colorlog
import os

from logiscanpy.core.application import LogiScanPy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


def read_config() -> dict:
    """Reads the configuration from the config.ini file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    config = configparser.ConfigParser()
    config.read("../config/config.ini")

    app_config = {}

    weights = config.get("app", "weights", fallback=None)
    if not isinstance(weights, str):
        raise TypeError("The 'weights' parameter must be a string.")
    app_config["weights"] = weights

    video = config.get("app", "video", fallback=None)
    if not isinstance(video, str):
        raise TypeError("The 'video' parameter must be a string.")
    app_config["video"] = video

    rtsp = config.getboolean("app", "rtsp", fallback=False)
    if not isinstance(rtsp, bool):
        raise TypeError("The 'rtsp' parameter must be a boolean.")
    app_config["rtsp"] = rtsp

    output = config.get("app", "output", fallback=None)
    if not isinstance(output, str):
        raise TypeError("The 'output' parameter must be a string.")
    app_config["output"] = output

    class_id = config.getint("app", "class_id", fallback=0)
    if not isinstance(class_id, int):
        raise TypeError("The 'class_id' parameter must be an integer.")
    app_config["class_id"] = class_id

    confidence = config.getfloat("app", "confidence", fallback=0.5)
    if not isinstance(confidence, float):
        raise TypeError("The 'confidence' parameter must be a float.")
    app_config["confidence"] = confidence

    show = config.getboolean("app", "show", fallback=True)
    if not isinstance(show, bool):
        raise TypeError("The 'show' parameter must be a boolean.")
    app_config["show"] = show

    save = config.getboolean("app", "save", fallback=False)
    if not isinstance(save, bool):
        raise TypeError("The 'save' parameter must be a boolean.")
    app_config["save"] = save

    return app_config


def main():
    """Entry point for the object counting script."""
    config = read_config()
    app = LogiScanPy(config)
    app.run_app()


if __name__ == "__main__":
    main()
