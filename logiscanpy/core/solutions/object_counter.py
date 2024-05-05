import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from logiscanpy.core.solutions import Solution

_LOGGER = logging.getLogger(__name__)


class ObjectCounter(Solution):
    """A class to manage the counting of objects entering multiple regions."""

    def __init__(self):
        """Initializes the Counter with default values."""
        self._reg_pts: List[List[Tuple[int, int]]] = [[(300, 300), (400, 300), (400, 400), (300, 400)]]
        self._counting_regions: List[Polygon] = [Polygon(self._reg_pts[0])]
        self._names: List[str] = []
        self._in_counts: int = 0
        self._count_ids: List[int] = []
        self._class_wise_count: Dict[str, int] = {}
        self._track_history: defaultdict = defaultdict(list)
        self._region_color: Tuple[int, int, int] = (255, 0, 0)
        self._region_thickness: int = 2
        self._debug = True

    def set_params(self, classes_names: List[str], reg_pts: List[List[Tuple[int, int]]], debug: bool):
        """Configures the Counter's counting region points and class names.

        Args:
            classes_names (List[str]): List of class names.
            reg_pts (List[List[Tuple[int, int]]]): List of lists of region points.
            debug (bool): Whether to enable debug mode.
        """
        self._reg_pts = reg_pts
        self._counting_regions = [Polygon(pts) for pts in reg_pts if len(pts) >= 3]
        if not self._counting_regions:
            _LOGGER.warning(
                "Invalid Region points provided, region_points must be >= 3 for polygons."
            )
        self._names = classes_names
        self._debug = debug

    def process_frame(self, im0: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """
        Main function to start the object counting process.

        Args:
            im0 (np.ndarray): Current frame from the video stream.
            tracks (np.ndarray): NumPy array of tracked detections with shape (num_detections, 6).
                                 Each row represents [x1, y1, x2, y2, class_id, track_id].

        Returns:
            np.ndarray: Frame with annotations and counting results (if debug is True).
        """
        if self._debug:
            for region in self._counting_regions:
                region_pts = np.array(region.exterior.coords, np.int32).reshape((-1, 1, 2))
                cv2.polylines(im0, [region_pts], True, self._region_color, self._region_thickness)

        if len(tracks) == 0:
            return im0

        for detection in tracks:
            x1, y1, x2, y2, class_id, track_id = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id, track_id = int(class_id), int(track_id)

            box_center = Point(((x1 + x2) / 2, (y1 + y2) / 2))

            if self._names[class_id] not in self._class_wise_count:
                self._class_wise_count[self._names[class_id]] = 0

            self._track_history[track_id].append(box_center)
            if len(self._track_history[track_id]) > 30:
                self._track_history[track_id].pop(0)

            prev_position = (
                self._track_history[track_id][-2]
                if len(self._track_history[track_id]) > 1
                else None
            )

            if prev_position is not None and track_id not in self._count_ids:
                for region in self._counting_regions:
                    if region.contains(box_center):
                        self._count_ids.append(track_id)
                        if not region.contains(prev_position):
                            self._in_counts += 1
                            self._class_wise_count[self._names[class_id]] += 1
                            break

            if self._debug:
                label = f"{self._names[class_id]}#{track_id}"
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if self._debug:
            label = [
                f"{str.capitalize(key)}: IN {value}"
                for key, value in self._class_wise_count.items()
                if value != 0
            ]
            label = "".join(label)
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            cv2.putText(
                im0,
                label,
                (im0.shape[1] - label_width - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        return im0

    def get_action_data(self) -> Dict[str, int]:
        """Returns a dictionary containing the class-wise counts for objects entering the region.

        Returns:
            Dict[str, int]: A dictionary where keys are class names (strings) and values are in counts (integers).
        """
        return self._class_wise_count.copy()

    def reset(self):
        """Resets the in counts, count IDs, and class-wise counts."""
        self._in_counts = 0
        self._count_ids.clear()
        self._class_wise_count.clear()
