import cv2
import numpy as np
from typing import List, Tuple


def draw_polygon(event: int, x: int, y: int, flags: int,
                 params: Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]) -> None:
    """Callback function for mouse events to draw a polygon.

    Args:
        event: OpenCV mouse event code.
        x: X-coordinate of the mouse event.
        y: Y-coordinate of the mouse event.
        flags: OpenCV mouse event flags.
        params: Additional parameters (polygons list, img, frame).
    """
    polygons, img, frame = params
    current_polygon = polygons[-1] if polygons else []
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.polylines(img, [np.array(current_polygon, np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
    elif event == cv2.EVENT_RBUTTONDOWN:
        current_polygon.clear()
        img[:] = frame.copy()


def calibrate_region(frame: np.ndarray, window_name: str) -> List[List[Tuple[int, int]]]:
    """Calibrate the polygon regions for speed estimation.

    Args:
        frame: Initial video frame.
        window_name: Name of the calibration window.

    Returns:
        List[List[Tuple[int, int]]]: List of lists of vertices defining the polygon regions.
    """
    polygons = [[]]
    img = frame.copy()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_polygon, (polygons, img, frame))
    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            polygons[-1].clear()
            img[:] = frame.copy()
        elif key == ord('n'):
            if polygons[-1]:
                polygons.append([])
    cv2.destroyWindow(window_name)
    if not polygons[-1]:
        polygons.pop()
    return polygons
