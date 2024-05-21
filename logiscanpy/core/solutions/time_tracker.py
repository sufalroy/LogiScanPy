import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics.utils.plotting import Annotator, colors

from logiscanpy.core.solutions import Solution

_LOGGER = logging.getLogger(__name__)


class TimeTracker(Solution):
    """A class to track the time spent by objects within multiple regions."""

    def __init__(self):
        """Initializes the TimeTracker instance with default values."""
        self.reg_pts: List[List[Tuple[int, int]]] = [
            [(300, 300), (400, 300), (400, 400), (300, 400)]
        ]
        self.regions: List[Polygon] = [Polygon(self.reg_pts[0])]
        self.classes_names: List[str] = []
        self.times: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.entry_times: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.region_color: Tuple[int, int, int] = (255, 0, 0)
        self.region_thickness: int = 2
        self.debug: bool = True
        self.annotator: Optional[Annotator] = None

    def set_params(
            self, classes_names: List[str], reg_pts: List[List[Tuple[int, int]]], debug: bool
    ):
        """Configures the TimeTracker instance's region points and class names.

        Args:
            classes_names (List[str]): List of class names.
            reg_pts (List[List[Tuple[int, int]]]): List of lists of region points.
            debug (bool): Whether to enable debug mode.
        """
        self.reg_pts = reg_pts
        self.regions = [Polygon(pts) for pts in reg_pts if len(pts) >= 3]
        if not self.regions:
            _LOGGER.warning(
                "Invalid Region points provided, region_points must be >= 3 for polygons."
            )
        self.classes_names = classes_names
        self.debug = debug

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
        self.annotator = Annotator(im0, self.region_thickness, self.classes_names)
        current_time = time.time()

        if self.debug:
            for region in self.regions:
                region_pts = np.array(region.exterior.coords, np.int32).reshape((-1, 1, 2))
                self.annotator.draw_region(reg_pts=region_pts, color=self.region_color, thickness=self.region_thickness)

        if not tracks.size:
            return im0

        for detection in tracks:
            x1, y1, x2, y2, class_id, track_id = detection.astype(int)

            box_center = Point(((x1 + x2) / 2, (y1 + y2) / 2))
            class_name = self.classes_names[class_id]

            for region_idx, region in enumerate(self.regions):
                if region.contains(box_center):
                    if track_id not in self.entry_times[region_idx]:
                        self.entry_times[region_idx][track_id] = current_time
                    else:
                        entry_time = self.entry_times[region_idx][track_id]
                        time_spent = current_time - entry_time
                        self.times[track_id][class_name] = self.times[track_id].get(class_name, 0.0) + time_spent
                        self.entry_times[region_idx][track_id] = current_time

            if self.debug:
                label = f"{class_name}#{track_id}: {int(self.times[track_id].get(class_name, 0.0))}s"
                self.annotator.box_label(
                    (x1, y1, x2, y2),
                    label=label,
                    color=colors(int(track_id), True),
                )

        return self.annotator.im

    def get_action_data(self) -> Dict[int, Dict[str, float]]:
        """Returns a dictionary containing the time spent by each object within the regions.

        Returns: Dict[int, Dict[str, float]]: A nested dictionary where the outer keys are track IDs (integers),
        the inner keys are class names (strings), and the values are time spent (floats).
        """
        return self.times.copy()

    def reset(self):
        """Resets the time spent and entry times."""
        self.times.clear()
        self.entry_times.clear()
