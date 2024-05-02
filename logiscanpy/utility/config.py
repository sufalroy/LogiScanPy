import yaml
from typing import List


def load_class_names(config_path: str) -> List[str]:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    class_names = config.get('names', [])
    return class_names
