import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics.utils.plotting import Annotator, colors

from logiscanpy.core.solutions import Solution

_LOGGER = logging.getLogger(__name__)


class ObjectCounter(Solution):
    """A class to manage the counting of objects entering multiple regions."""

    def __init__(self):
        """Initializes the Counter with default values."""
        self.reg_pts: List[List[Tuple[int, int]]] = [[(300, 300), (400, 300), (400, 400), (300, 400)]]
        self.counting_regions: List[Polygon] = [Polygon(self.reg_pts[0])]
        self.names: List[str] = []
        self.in_counts: int = 0
        self.count_ids: List[int] = []
        self.class_wise_count: Dict[str, int] = {}
        self.track_history: defaultdict = defaultdict(list)
        self.region_color: Tuple[int, int, int] = (255, 0, 0)
        self.region_thickness: int = 2
        self.debug: bool = True
        self.annotator: Optional[Annotator] = None

    def set_params(self, classes_names: List[str], reg_pts: List[List[Tuple[int, int]]], debug: bool):
        """Configures the Counter's counting region points and class names.

        Args:
            classes_names (List[str]): List of class names.
            reg_pts (List[List[Tuple[int, int]]]): List of lists of region points.
            debug (bool): Whether to enable debug mode.
        """
        self.reg_pts = reg_pts
        self.counting_regions = [Polygon(pts) for pts in reg_pts if len(pts) >= 3]
        if not self.counting_regions:
            _LOGGER.warning(
                "Invalid Region points provided, region_points must be >= 3 for polygons."
            )
        self.names = classes_names
        self.debug = debug

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
        self.annotator = Annotator(im0, self.region_thickness, self.names)

        if self.debug:
            for region in self.counting_regions:
                region_pts = np.array(region.exterior.coords, np.int32).reshape((-1, 1, 2))
                self.annotator.draw_region(reg_pts=region_pts, color=self.region_color,
                                           thickness=self.region_thickness)

        if len(tracks) == 0:
            return im0

        for detection in tracks:
            x1, y1, x2, y2, class_id, track_id = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id, track_id = int(class_id), int(track_id)

            box_center = Point(((x1 + x2) / 2, (y1 + y2) / 2))

            if self.names[class_id] not in self.class_wise_count:
                self.class_wise_count[self.names[class_id]] = 0

            self.track_history[track_id].append(box_center)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            prev_position = (
                self.track_history[track_id][-2]
                if len(self.track_history[track_id]) > 1
                else None
            )

            if prev_position is not None and track_id not in self.count_ids:
                for region in self.counting_regions:
                    if region.contains(box_center):
                        self.count_ids.append(track_id)
                        if not region.contains(prev_position):
                            self.in_counts += 1
                            self.class_wise_count[self.names[class_id]] += 1
                            break

            if self.debug:
                label = f"{self.names[class_id]}#{track_id}"
                self.annotator.box_label(
                    (x1, y1, x2, y2),
                    label=label,
                    color=colors(int(track_id), True),
                )

        if self.debug:
            labels_dict = {
                str.capitalize(key): f"IN {value}"
                for key, value in self.class_wise_count.items()
                if value != 0
            }
            self.annotator.display_analytics(im0, labels_dict, (255, 255, 255), (0, 0, 0), 5)

        return im0

    def get_action_data(self) -> Dict[str, int]:
        """Returns a dictionary containing the class-wise counts for objects entering the region.

        Returns:
            Dict[str, int]: A dictionary where keys are class names (strings) and values are in counts (integers).
        """
        return self.class_wise_count.copy()

    def reset(self):
        """Resets the in counts, count IDs, and class-wise counts."""
        self.in_counts = 0
        self.count_ids.clear()
        self.class_wise_count.clear()
