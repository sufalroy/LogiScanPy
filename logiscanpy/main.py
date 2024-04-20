import configparser
import logging
import os
from typing import Dict

import colorlog

from logiscanpy.core.application import LogiScanPy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


def read_config() -> Dict[str, str]:
    """Reads the configuration from the config.ini file.

    Returns:
        Dict[str, str]: A dictionary containing the configuration parameters.

    Raises:
        TypeError: If any configuration parameter has an invalid type.
    """
    config = configparser.ConfigParser()
    config.read("../config/config.ini")

    app_config = {}

    def get_typed_value(section, option, value_type):
        try:
            return value_type(config.get(section, option, fallback=None))
        except (ValueError, TypeError):
            raise TypeError(f"The '{option}' parameter must be a {value_type.__name__}.")

    app_config["weights"] = get_typed_value("app", "weights", str)
    app_config["video"] = get_typed_value("app", "video", str)
    app_config["rtsp"] = get_typed_value("app", "rtsp", bool)
    app_config["output"] = get_typed_value("app", "output", str)
    app_config["class_id"] = get_typed_value("app", "class_id", int)
    app_config["confidence"] = get_typed_value("app", "confidence", float)
    app_config["show"] = get_typed_value("app", "show", bool)
    app_config["save"] = get_typed_value("app", "save", bool)

    return app_config


def main() -> None:
    """Entry point for the object counting script."""
    config = read_config()
    app = LogiScanPy(config)
    app.run_app()


if __name__ == "__main__":
    main()
