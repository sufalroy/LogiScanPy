import cv2
from collections import defaultdict
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors
import logging

check_requirements("shapely>=2.0.0")
from shapely.geometry import LineString, Point, Polygon

logger = logging.getLogger(__name__)


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""
        self._window_name = "LogiScan.v.0.1.0"
        self._reg_pts = [(20, 400), (1260, 400)]
        self._line_dist_thresh = 15
        self._counting_region = None
        self._im0 = None
        self._tf = None
        self._view_img = False
        self._view_in_counts = True
        self._view_out_counts = True
        self._names = None
        self._annotator = None
        self._in_counts = 0
        self._out_counts = 0
        self._count_ids = []
        self._class_wise_count = {}
        self._count_txt_thickness = 0
        self._count_txt_color = (255, 255, 255)
        self._count_bg_color = (255, 255, 255)
        self._cls_txtdisplay_gap = 50
        self._fontsize = 0.6
        self._track_history = defaultdict(list)
        self._track_thickness = 2
        self._draw_tracks = False
        self._track_color = None
        self._env_check = check_imshow(warn=True)
        self._region_color = (255, 0, 255)
        self._region_thickness = 5

    def set_args(self, classes_names, reg_pts, count_reg_color=(255, 0, 255), count_txt_color=(0, 0, 0),
                 count_bg_color=(255, 255, 255), line_thickness=2, track_thickness=2, view_img=False,
                 view_in_counts=True, view_out_counts=True, draw_tracks=False, track_color=None, region_thickness=5,
                 line_dist_thresh=15, cls_txtdisplay_gap=50):
        """Configures the Counter's image, bounding box line thickness, and counting region points."""
        self._tf = line_thickness
        self._view_img = view_img
        self._view_in_counts = view_in_counts
        self._view_out_counts = view_out_counts
        self._track_thickness = track_thickness
        self._draw_tracks = draw_tracks
        if len(reg_pts) == 2:
            logger.info("Line Counter Initiated.")
            self._reg_pts = reg_pts
            self._counting_region = LineString(self._reg_pts)
        elif len(reg_pts) >= 3:
            logger.info("Polygon Counter Initiated.")
            self._reg_pts = reg_pts
            self._counting_region = Polygon(self._reg_pts)
        else:
            logger.warning("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
            logger.info("Using Line Counter Now")
            self._counting_region = LineString(self._reg_pts)
        self._names = classes_names
        self._track_color = track_color
        self._count_txt_color = count_txt_color
        self._count_bg_color = count_bg_color
        self._region_color = count_reg_color
        self._region_thickness = region_thickness
        self._line_dist_thresh = line_dist_thresh
        self._cls_txtdisplay_gap = cls_txtdisplay_gap

    def _extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""
        self._annotator = Annotator(self._im0, self._tf, self._names)
        self._annotator.draw_region(reg_pts=self._reg_pts, color=self._region_color, thickness=self._region_thickness)
        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()
            for box, track_id, cls in zip(boxes, track_ids, clss):
                self._annotator.box_label(box, label=f"{self._names[cls]}#{track_id}",
                                          color=colors(int(track_id), True))
                if self._names[cls] not in self._class_wise_count:
                    if len(self._names[cls]) > 5:
                        self._names[cls] = self._names[cls][:5]
                    self._class_wise_count[self._names[cls]] = {"in": 0, "out": 0}
                track_line = self._track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)
                if self._draw_tracks:
                    self._annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self._track_color if self._track_color else colors(int(track_id), True),
                        track_thickness=self._track_thickness,
                    )
                prev_position = self._track_history[track_id][-2] if len(self._track_history[track_id]) > 1 else None
                if len(self._reg_pts) >= 3:
                    is_inside = self._counting_region.contains(Point(track_line[-1]))
                    if prev_position is not None and is_inside and track_id not in self._count_ids:
                        self._count_ids.append(track_id)
                        if (box[0] - prev_position[0]) * (self._counting_region.centroid.x - prev_position[0]) > 0:
                            self._in_counts += 1
                            self._class_wise_count[self._names[cls]]["in"] += 1
                        else:
                            self._out_counts += 1
                            self._class_wise_count[self._names[cls]]["out"] += 1
                elif len(self._reg_pts) == 2:
                    if prev_position is not None and track_id not in self._count_ids:
                        distance = Point(track_line[-1]).distance(self._counting_region)
                        if distance < self._line_dist_thresh and track_id not in self._count_ids:
                            self._count_ids.append(track_id)
                            if (box[0] - prev_position[0]) * (self._counting_region.centroid.x - prev_position[0]) > 0:
                                self._in_counts += 1
                                self._class_wise_count[self._names[cls]]["in"] += 1
                            else:
                                self._out_counts += 1
                                self._class_wise_count[self._names[cls]]["out"] += 1

        label = "LogiScan Analytics \t"
        for key, value in self._class_wise_count.items():
            if value["in"] != 0 or value["out"] != 0:
                if not self._view_in_counts and not self._view_out_counts:
                    label = None
                elif not self._view_in_counts:
                    label += f"{str.capitalize(key)}: IN {value['in']} \t"
                elif not self._view_out_counts:
                    label += f"{str.capitalize(key)}: OUT {value['out']} \t"
                else:
                    label += f"{str.capitalize(key)}: IN {value['in']} OUT {value['out']} \t"

        label = label.rstrip()
        label = label.split("\t")
        if label is not None:
            self._annotator.display_counts(
                counts=label,
                count_txt_color=self._count_txt_color,
                count_bg_color=self._count_bg_color,
            )

    def _display_frames(self):
        """Display frames."""
        if self._env_check:
            cv2.namedWindow(self._window_name)
            cv2.imshow(self._window_name, self._im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):
        """Main function to start the object counting process."""
        self._im0 = im0
        self._extract_and_process_tracks(tracks)
        if self._view_img:
            self._display_frames()
        return self._im0

    def reset_counts(self):
        """Resets the in and out counts, as well as the count IDs and class-wise counts."""
        self._in_counts = 0
        self._out_counts = 0
        self._count_ids = []
        self._class_wise_count = {}

    def get_class_wise_count(self) -> dict:
        """Returns a dictionary containing the class-wise counts for objects entering and exiting the region.

        Returns: dict: A dictionary where keys are class names (strings) and values are dictionaries with "in" and
        "out" counts (integers).
        """
        return self._class_wise_count.copy()
