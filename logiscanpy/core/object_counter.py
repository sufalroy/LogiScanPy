import logging
from collections import defaultdict
from typing import List, Tuple

import cv2
from shapely.geometry import Point, Polygon
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

logger = logging.getLogger(__name__)


class ObjectCounter:
    """A class to manage the counting of objects entering a region in a real-time video stream."""

    def __init__(self):
        """Initializes the Counter with default values."""
        self._window_name = "LogiScan.v.0.1.0"
        self._reg_pts: List[Tuple[int, int]] = [(300, 300), (400, 300), (400, 400), (300, 400)]
        self._counting_region = None
        self._im0 = None
        self._tf = 2
        self._view_img = False
        self._names = None
        self._annotator = None
        self._in_counts = 0
        self._count_ids: List[int] = []
        self._class_wise_count: dict = {}
        self._count_txt_color = (255, 255, 255)
        self._count_bg_color = (0, 0, 0)
        self._track_history: defaultdict = defaultdict(list)
        self._env_check = check_imshow(warn=True)
        self._region_color = (255, 0, 0)
        self._region_thickness = 2

    def set_args(
            self,
            classes_names: List[str],
            reg_pts: List[Tuple[int, int]],
            view_img: bool = False,
    ):
        """Configures the Counter's image, bounding box line thickness, and counting region points."""
        self._view_img = view_img
        if len(reg_pts) >= 3:
            self._reg_pts = reg_pts
            self._counting_region = Polygon(self._reg_pts)
        else:
            logger.warning("Invalid Region points provided, region_points must be >= 3 for polygons.")
            self._counting_region = Polygon(self._reg_pts)

        self._names = classes_names

    def _extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""
        self._annotator = Annotator(self._im0, self._tf, self._names)
        self._annotator.draw_region(
            reg_pts=self._reg_pts,
            color=self._region_color,
            thickness=self._region_thickness
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
                    mask_color=colors(int(track_id), True)
                )

                self._class_wise_count.setdefault(self._names[cls], 0)

                self._track_history[track_id].append(box_center)
                self._track_history[track_id] = self._track_history[track_id][-30:]

                prev_position = self._track_history[track_id][-2] if len(self._track_history[track_id]) > 1 else None

                if prev_position is not None and track_id not in self._count_ids:
                    if self._counting_region.contains(box_center) and not self._counting_region.contains(prev_position):
                        self._count_ids.append(track_id)
                        self._in_counts += 1
                        self._class_wise_count[self._names[cls]] += 1

            label = "LogiScan Analytics \t"
            label_parts = [f"{str.capitalize(key)}: IN {value} \t" for key, value in self._class_wise_count.items() if
                           value != 0]
            label += "".join(label_parts)
            label = label.rstrip()
            label = label.split("\t")

            if label:
                self._annotator.display_counts(
                    counts=label,
                    count_txt_color=self._count_txt_color,
                    count_bg_color=self._count_bg_color
                )

    def _display_frames(self):
        """Display frames."""
        if self._env_check:
            cv2.namedWindow(self._window_name)
            cv2.imshow(self._window_name, self._im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self._im0 = im0
        self._extract_and_process_tracks(tracks)
        if self._view_img:
            self._display_frames()
        return self._im0

    def reset_counts(self):
        """Resets the in counts, count IDs, and class-wise counts."""
        self._in_counts = 0
        self._count_ids.clear()
        self._class_wise_count.clear()

    def get_class_wise_count(self) -> dict:
        """Returns a dictionary containing the class-wise counts for objects entering the region.

        Returns:
            dict: A dictionary where keys are class names (strings) and values are in counts (integers).
        """
        return self._class_wise_count.copy()
