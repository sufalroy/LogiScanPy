from typing import Dict

from logiscanpy.core.actions import Action


class TimeAlert(Action):
    """Alerts based on time spent by objects.

    This action checks if the time spent by objects (identified by track IDs) is more than a given value.
    If more than one object has spent more time than the specified threshold, an alert is printed.
    """

    def __init__(self, time_threshold: float = 30.0):
        """Initializes the TimeAlert with a specified time threshold.

        Args:
            time_threshold (float): The time threshold for triggering an alert.
        """
        self._time_threshold = time_threshold

    def execute(self, data: Dict[int, Dict[str, float]]) -> None:
        """Checks if the time spent by objects exceeds the threshold and prints an alert if necessary.

        This method iterates over the provided data, which is a nested dictionary where the outer keys are track IDs (integers),
        the inner keys are class names (strings), and the values are time spent (floats). It checks if the time spent by any object
        is more than the specified threshold. If more than one object meets this condition, an alert is printed.

        Args:
            data (Dict[int, Dict[str, float]]): The time spent by objects, organized by track ID and class name.
        """
        alert_count = 0
        for track_id, class_times in data.items():
            for class_name, time_spent in class_times.items():
                if time_spent > self._time_threshold:
                    alert_count += 1
                    break

        if alert_count > 1:
            print(f"Alert: More than one object has spent more than {self._time_threshold} seconds.")

    def cleanup(self) -> None:
        """Cleans up resources used by the action."""
        pass
