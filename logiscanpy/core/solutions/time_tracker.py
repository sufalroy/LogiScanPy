import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from logiscanpy.core.solutions import Solution

_LOGGER = logging.getLogger(__name__)


class TimeTracker(Solution):
    """A class to track the time spent by objects within multiple regions."""

    def __init__(self):
        """Initializes the TimeTracker instance with default values."""
        self._reg_pts: List[List[Tuple[int, int]]] = [
            [(300, 300), (400, 300), (400, 400), (300, 400)]
        ]
        self._regions: List[Polygon] = [Polygon(self._reg_pts[0])]
        self._classes_names: List[str] = []
        self._times: Dict[int, Dict[str, float]] = defaultdict(dict)
        self._entry_times: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._region_color: Tuple[int, int, int] = (255, 0, 0)
        self._region_thickness: int = 2
        self._debug: bool = True

    def set_params(
            self, classes_names: List[str], reg_pts: List[List[Tuple[int, int]]], debug: bool
    ):
        """Configures the TimeTracker instance's region points and class names.

        Args:
            classes_names (List[str]): List of class names.
            reg_pts (List[List[Tuple[int, int]]]): List of lists of region points.
            debug (bool): Whether to enable debug mode.
        """
        self._reg_pts = reg_pts
        self._regions = [Polygon(pts) for pts in reg_pts if len(pts) >= 3]
        if not self._regions:
            _LOGGER.warning(
                "Invalid Region points provided, region_points must be >= 3 for polygons."
            )
        self._classes_names = classes_names
        self._debug = debug

    def process_frame(self, im0: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """
        Main function to track the time spent by objects within the specified regions.

        Args:
            im0 (np.ndarray): Current frame from the video stream.
            tracks (np.ndarray): NumPy array of tracked detections with shape (num_detections, 6).
                                 Each row represents [x1, y1, x2, y2, class_id, track_id].

        Returns:
            np.ndarray: Frame with annotations and time information (if debug is True).
        """
        if self._debug:
            for region in self._regions:
                region_pts = np.array(region.exterior.coords, np.int32).reshape((-1, 1, 2))
                cv2.polylines(im0, [region_pts], True, self._region_color, self._region_thickness)

        if len(tracks) == 0:
            return im0

        current_time = time.time()

        for detection in tracks:
            x1, y1, x2, y2, class_id, track_id = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id, track_id = int(class_id), int(track_id)

            box_center = Point(((x1 + x2) / 2, (y1 + y2) / 2))
            class_name = self._classes_names[class_id]

            for region_idx, region in enumerate(self._regions):
                if region.contains(box_center):
                    if track_id not in self._entry_times[region_idx]:
                        self._entry_times[region_idx][track_id] = current_time
                    else:
                        entry_time = self._entry_times[region_idx][track_id]
                        time_spent = current_time - entry_time
                        self._times[track_id][class_name] = (self._times[track_id].get(class_name, 0.0) + time_spent)
                        self._entry_times[region_idx][track_id] = current_time

            if self._debug:
                label = f"{class_name}#{track_id}: {int(self._times[track_id].get(class_name, 0.0))}s"
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return im0

    def get_times(self) -> Dict[int, Dict[str, float]]:
        """Returns a dictionary containing the time spent by each object within the regions.

        Returns: Dict[int, Dict[str, float]]: A nested dictionary where the outer keys are track IDs (integers),
        the inner keys are class names (strings), and the values are time spent (floats).
        """
        return self._times.copy()

    def reset_times(self):
        """Resets the time spent and entry times."""
        self._times.clear()
        self._entry_times.clear()
