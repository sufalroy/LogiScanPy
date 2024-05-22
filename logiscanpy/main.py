import logging
import os

import colorlog

from logiscanpy.core.application import Application
from logiscanpy.utility.config import read_config

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

_LOGGER = logging.getLogger(__name__)

_CONFIG_PATH = "../config/config.ini"


def main():
    configs = read_config(_CONFIG_PATH)
    app = Application(configs)

    if not app.initialize():
        _LOGGER.error("Failed to initialize the application")
        return

    try:
        app.run()
    except KeyboardInterrupt:
        _LOGGER.info("Application stopped by user")

    app.cleanup()


if __name__ == "__main__":
    main()
