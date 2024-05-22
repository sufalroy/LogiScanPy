import configparser
import os
from typing import List, Dict

import yaml

from logiscanpy.core.detection.detection_factory import Engine


def load_class_names(config_path: str) -> List[str]:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    class_names = config.get('names', [])
    return class_names


def read_config(config_path: str) -> List[Dict[str, str]]:
    """Reads the configuration from the config.ini file.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the configuration parameters for each camera stream.

    Raises:
        ValueError: If any configuration parameter is invalid or does not exist.
        TypeError: If any configuration parameter has an invalid type.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    app_configs = []

    for section in config.sections():
        if section.startswith("camera_"):
            app_config = {}

            id = config.get(section, "id", fallback=None)
            app_config["id"] = id

            weights = config.get(section, "weights", fallback=None)
            if weights is None or not os.path.isfile(weights):
                raise ValueError(f"The 'weights' parameter must be a valid file path. Got: {weights}")
            app_config["weights"] = weights

            rtsp = config.getboolean(section, "rtsp", fallback=False)
            app_config["rtsp"] = rtsp

            video = config.get(section, "video", fallback=None)
            if not rtsp and (video is None or not os.path.isfile(video)):
                raise ValueError(f"The 'video' parameter must be a valid file path. Got: {video}")
            app_config["video"] = video

            output = config.get(section, "output", fallback=None)
            if output is not None:
                output_dir = os.path.dirname(output)
                if not os.path.isdir(output_dir):
                    raise ValueError(f"The 'output' directory does not exist: {output_dir}")
            app_config["output"] = output

            class_names_file = config.get(section, "class_names_file", fallback=None)
            if class_names_file is not None and not os.path.isfile(class_names_file):
                raise ValueError(f"The 'class_names_file' parameter must be a valid file path. Got: {class_names_file}")
            app_config["class_names_file"] = class_names_file

            confidence = config.getfloat(section, "confidence", fallback=0.5)
            app_config["confidence"] = confidence

            show = config.getboolean(section, "show", fallback=True)
            app_config["show"] = show

            save = config.getboolean(section, "save", fallback=False)
            app_config["save"] = save

            solution_type = config.get(section, "solution_type", fallback=None)
            app_config["solution_type"] = solution_type

            engine_str = config.get(section, "engine", fallback=None)
            if engine_str is not None:
                try:
                    engine_enum = Engine[engine_str]
                    app_config["engine"] = engine_enum
                except KeyError:
                    raise ValueError(
                        f"Invalid engine type '{engine_str}'. Must be one of {list(Engine.__members__.keys())}")

            app_configs.append(app_config)

    return app_configs
