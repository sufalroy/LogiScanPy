from typing import Tuple, List

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    """Compute Intersection over Union (IoU) of two sets of bounding boxes.

    Both sets of boxes are expected to be in `(x_min, y_min, x_max, y_max)` format.

    Args:
        boxes_true (np.ndarray): 2D `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4)` where `N` is number of true objects.
        boxes_detection (np.ndarray): 2D `np.ndarray` representing detection boxes.
            `shape = (M, 4)` where `M` is number of detected objects.

    Returns:
        np.ndarray: Pairwise IoU of boxes from `boxes_true` and `boxes_detection`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)
    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_detection[:, 2:])
    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    return area_inter / (area_true[:, None] + area_detection - area_inter)


def iou_distance(atracks: List, btracks: List) -> np.ndarray:
    """Computes the IoU distance between two lists of tracks.

    Args:
        atracks (List): List of tracks or bounding boxes.
        btracks (List): List of tracks or bounding boxes.

    Returns:
        np.ndarray: Cost matrix of IoU distances.
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if _ious.size != 0:
        _ious = box_iou_batch(np.asarray(atlbrs), np.asarray(btlbrs))
    cost_matrix = 1 - _ious

    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: List) -> np.ndarray:
    """Fuses detection scores with the IoU cost matrix.

    Args:
        cost_matrix (np.ndarray): Cost matrix of IoU distances.
        detections (List): List of detections.

    Returns:
        np.ndarray: Fused cost matrix.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def indices_to_matches(
        cost_matrix: np.ndarray, indices: np.ndarray, thresh: float
) -> Tuple[np.ndarray, tuple, tuple]:
    """
    Convert indices of matched pairs to matched pairs, unmatched indices of both sets,
    and masked cost values below a threshold.

    Args:
        cost_matrix (np.ndarray): The cost matrix.
        indices (np.ndarray): Array of matched indices.
        thresh (float): Threshold value for matching.

    Returns:
        Tuple[np.ndarray, tuple, tuple]: A tuple containing:
            - matched pairs of indices,
            - unmatched indices from the first set,
            - unmatched indices from the second set.
    """
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))
    return matches, unmatched_a, unmatched_b


def linear_assignment(
        cost_matrix: np.ndarray, thresh: float
) -> [np.ndarray, Tuple[int], Tuple[int, int]]:
    """
    Perform linear assignment on a cost matrix and return matched pairs of indices,
    and indices of unmatched elements from both sets.

    Args:
        cost_matrix (np.ndarray): The cost matrix.
        thresh (float): Threshold value for matching.

    Returns:
        [np.ndarray, Tuple[int], Tuple[int, int]]: A list containing:
            - matched pairs of indices,
            - indices of unmatched elements from the first set,
            - indices of unmatched elements from the second set.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    cost_matrix[cost_matrix > thresh] = thresh + 1e-4
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    indices = np.column_stack((row_ind, col_ind))

    return indices_to_matches(cost_matrix, indices, thresh)
