import logging
from collections import defaultdict
from typing import List, Tuple, Dict

from shapely.geometry import Point, Polygon
from ultralytics.utils.plotting import Annotator, colors

logger = logging.getLogger(__name__)


class ObjectCounter:
    """A class to manage the counting of objects entering a region."""

    def __init__(self):
        """Initializes the Counter with default values."""
        self._reg_pts: List[Tuple[int, int]] = [(300, 300), (400, 300), (400, 400), (300, 400)]
        self._counting_region = None
        self._names = None
        self._annotator = None
        self._in_counts = 0
        self._count_ids: List[int] = []
        self._class_wise_count: Dict[str, int] = {}
        self._track_history: defaultdict = defaultdict(list)
        self._region_color = (255, 0, 0)
        self._region_thickness = 2

    def set_args(
            self,
            classes_names: List[str],
            reg_pts: List[Tuple[int, int]],
    ):
        """Configures the Counter's counting region points."""
        if len(reg_pts) >= 3:
            self._reg_pts = reg_pts
            self._counting_region = Polygon(self._reg_pts)
        else:
            logger.warning("Invalid Region points provided, region_points must be >= 3 for polygons.")
            self._counting_region = Polygon(self._reg_pts)

        self._names = classes_names

    def _extract_and_process_tracks(self, im0, tracks):
        """Extracts and processes tracks for object counting."""
        self._annotator = Annotator(im0, 2, self._names)
        self._annotator.draw_region(
            reg_pts=self._reg_pts,
            color=self._region_color,
            thickness=self._region_thickness,
        )

        if tracks[0].boxes.id is not None and tracks[0].masks is not None:
            masks = tracks[0].masks.xy
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            for box, mask, track_id, cls in zip(boxes, masks, track_ids, clss):
                box_center = Point(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
                self._annotator.seg_bbox(
                    mask,
                    track_label=f"{self._names[cls]}#{track_id}",
                    mask_color=colors(int(track_id), True),
                )

                if self._names[cls] not in self._class_wise_count:
                    self._class_wise_count[self._names[cls]] = 0

                self._track_history[track_id].append(box_center)
                if len(self._track_history[track_id]) > 30:
                    self._track_history[track_id].pop(0)

                prev_position = self._track_history[track_id][-2] if len(self._track_history[track_id]) > 1 else None

                if prev_position is not None and track_id not in self._count_ids:
                    if self._counting_region.contains(box_center):
                        self._count_ids.append(track_id)
                        if not self._counting_region.contains(prev_position):
                            self._in_counts += 1
                            self._class_wise_count[self._names[cls]] += 1

            label = "LogiScan Analytics \t"
            label_parts = [
                f"{str.capitalize(key)}: IN {value} \t"
                for key, value in self._class_wise_count.items()
                if value != 0
            ]
            label += "".join(label_parts)
            label = label.rstrip()
            label = label.split("\t")

            if label:
                self._annotator.display_counts(
                    counts=label,
                    count_txt_color=(255, 255, 255),
                    count_bg_color=(0, 0, 0),
                )

        return self._annotator.result()

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.

        Returns:
            ndarray: Frame with annotations and counting results.
        """
        return self._extract_and_process_tracks(im0, tracks)

    def reset_counts(self):
        """Resets the in counts, count IDs, and class-wise counts."""
        self._in_counts = 0
        self._count_ids.clear()
        self._class_wise_count.clear()

    def get_class_wise_count(self) -> Dict[str, int]:
        """Returns a dictionary containing the class-wise counts for objects entering the region.

        Returns:
            Dict[str, int]: A dictionary where keys are class names (strings) and values are in counts (integers).
        """
        return self._class_wise_count.copy()
