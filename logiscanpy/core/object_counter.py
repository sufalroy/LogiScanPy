import cv2
from collections import defaultdict
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")
from shapely.geometry import LineString, Point, Polygon


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""
        self.is_drawing = False
        self.selected_point = None
        self.reg_pts = [(20, 400), (1260, 400)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = True
        self.names = None
        self.annotator = None
        self.window_name = "HUL LogiScan.v.0.0.1"
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.count_txt_thickness = 0
        self.count_txt_color = (255, 255, 255)
        self.count_bg_color = (255, 255, 255)
        self.cls_txtdisplay_gap = 50
        self.fontsize = 0.6
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = None
        self.env_check = check_imshow(warn=True)

    def set_args(self, classes_names, reg_pts, count_reg_color=(255, 0, 255), count_txt_color=(0, 0, 0),
                 count_bg_color=(255, 255, 255), line_thickness=2, track_thickness=2, view_img=False,
                 view_in_counts=True, view_out_counts=True, draw_tracks=False, track_color=None, region_thickness=5,
                 line_dist_thresh=15, cls_txtdisplay_gap=50):
        """Configures the Counter's image, bounding box line thickness, and counting region points."""
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        if len(reg_pts) == 2:
            print("Line Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = LineString(self.reg_pts)
        elif len(reg_pts) >= 3:
            print("Polygon Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)
        self.names = classes_names
        self.track_color = track_color
        self.count_txt_color = count_txt_color
        self.count_bg_color = count_bg_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh
        self.cls_txtdisplay_gap = cls_txtdisplay_gap

    def mouse_event_for_region(self, event, x, y, flags, params):
        """This function is designed to move region with mouse events in a real-time video stream."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if isinstance(point, (tuple, list)) and len(point) >= 2 and (
                        abs(x - point[0]) < 10 and abs(y - point[1]) < 10):
                    self.selected_point = i
                    self.is_drawing = True
                    break
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""
        self.annotator = Annotator(self.im0, self.tf, self.names)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)
        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()
            for box, track_id, cls in zip(boxes, track_ids, clss):
                self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))
                if self.names[cls] not in self.class_wise_count:
                    if len(self.names[cls]) > 5:
                        self.names[cls] = self.names[cls][:5]
                    self.class_wise_count[self.names[cls]] = {"in": 0, "out": 0}
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color if self.track_color else colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )
                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
                if len(self.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))
                    if prev_position is not None and is_inside and track_id not in self.count_ids:
                        self.count_ids.append(track_id)
                        if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]["in"] += 1
                        else:
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]["out"] += 1
                elif len(self.reg_pts) == 2:
                    if prev_position is not None and track_id not in self.count_ids:
                        distance = Point(track_line[-1]).distance(self.counting_region)
                        if distance < self.line_dist_thresh and track_id not in self.count_ids:
                            self.count_ids.append(track_id)
                            if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.names[cls]]["in"] += 2
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.names[cls]]["out"] += 1
        label = "HUL LogiScan Analytics \t"
        for key, value in self.class_wise_count.items():
            if value["in"] != 0 or value["out"] != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    label = None
                elif not self.view_in_counts:
                    label += f"{str.capitalize(key)}: IN {value['in']} \t"
                elif not self.view_out_counts:
                    label += f"{str.capitalize(key)}: OUT {value['out']} \t"
                else:
                    label += f"{str.capitalize(key)}: IN {value['in']} OUT {value['out']} \t"
        label = label.rstrip()
        label = label.split("\t")
        if label is not None:
            self.annotator.display_counts(
                counts=label,
                count_txt_color=self.count_txt_color,
                count_bg_color=self.count_bg_color,
            )

    def display_frames(self):
        """Display frame."""
        if self.env_check:
            cv2.namedWindow(self.window_name)
            if len(self.reg_pts) == 4:
                cv2.setMouseCallback(self.window_name, self.mouse_event_for_region, {"region_points": self.reg_pts})
            cv2.imshow(self.window_name, self.im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):
        """Main function to start the object counting process."""
        self.im0 = im0
        self.extract_and_process_tracks(tracks)
        if self.view_img:
            self.display_frames()
        return self.im0

    def reset_counts(self):
        """Resets the in and out counts, as well as the count IDs and class-wise counts."""
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
