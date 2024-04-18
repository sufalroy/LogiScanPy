import cv2
import numpy as np


def draw_polygon(event, x, y, flags, params):
    """
    Callback function for mouse events to draw a polygon.

    Args:
        event: OpenCV mouse event code.
        x: X-coordinate of the mouse event.
        y: Y-coordinate of the mouse event.
        flags: OpenCV mouse event flags.
        params: Additional parameters (vertices list, img, frame).
    """
    vertices, img, frame = params

    if event == cv2.EVENT_LBUTTONDOWN:
        vertices.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.polylines(img, [np.array(vertices, np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 2)

    elif event == cv2.EVENT_RBUTTONDOWN:
        vertices.clear()
        img[:] = frame.copy()


def calibrate_region(frame):
    """
    Calibrate the polygon region for speed estimation.

    Args:
        frame: Initial video frame.

    Returns:
        List[Tuple[int, int]]: List of vertices defining the polygon region.
    """
    vertices = []
    img = frame.copy()
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", draw_polygon, (vertices, img, frame))

    while True:
        cv2.imshow("Calibration", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            vertices.clear()
            img[:] = frame.copy()

    cv2.destroyAllWindows()
    return vertices
