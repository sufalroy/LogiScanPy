from collections import OrderedDict
from enum import Enum
from typing import List, Tuple

import numpy as np

from logiscanpy.core.tracking import matching
from logiscanpy.core.tracking.kalman_filter import KalmanFilter


class TrackState(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class BaseTrack:
    _count = 0

    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.NEW
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_counter():
        BaseTrack._count = 0
        BaseTrack.track_id = 0
        BaseTrack.start_frame = 0
        BaseTrack.frame_id = 0
        BaseTrack.time_since_update = 0

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    _external_count = 0

    def __init__(self, tlwh: List[float], score: float, class_ids: List[int], minimum_consecutive_frames: int):
        super().__init__()
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = score
        self.class_ids = class_ids
        self.tracklet_len = 0
        self.internal_track_id = 0
        self.external_track_id = -1
        self.minimum_consecutive_frames = minimum_consecutive_frames

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = []
            multi_covariance = []
            for i, st in enumerate(stracks):
                multi_mean.append(st.mean.copy())
                multi_covariance.append(st.covariance)
                if st.state != TrackState.TRACKED:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = (STrack.shared_kalman
                                            .multi_predict(np.asarray(multi_mean), np.asarray(multi_covariance)))

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        self.kalman_filter = kalman_filter
        self.internal_track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = frame_id == 1
        if self.minimum_consecutive_frames == 1:
            self.external_track_id = self.next_external_id()
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id: int, new_id: bool = False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.frame_id = frame_id
        if new_id:
            self.internal_track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.TRACKED
        if self.tracklet_len == self.minimum_consecutive_frames:
            self.is_activated = True
            if self.external_track_id == -1:
                self.external_track_id = self.next_external_id()

        self.score = new_track.score

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self) -> np.ndarray:
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def next_external_id():
        STrack._external_count += 1
        return STrack._external_count

    @staticmethod
    def reset_external_counter():
        STrack._external_count = 0

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self) -> str:
        return f"OT_{self.internal_track_id}_({self.start_frame}-{self.frame_id})"


class ByteTrack:
    def __init__(
            self,
            track_activation_threshold: float = 0.25,
            lost_track_buffer: int = 30,
            minimum_matching_threshold: float = 0.8,
            frame_rate: int = 30,
            minimum_consecutive_frames: int = 1,
    ):
        self.track_activation_threshold = track_activation_threshold
        self.minimum_matching_threshold = minimum_matching_threshold
        self.frame_id = 0
        self.det_thresh = self.track_activation_threshold + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.kalman_filter = KalmanFilter()
        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []

    def reset(self):
        self.frame_id = 0
        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []
        BaseTrack.reset_counter()
        STrack.reset_external_counter()

    def update_with_detections(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        tensors = self._convert_detections_to_tensors(boxes, scores, class_ids)
        tracks = self.update_with_tensors(tensors=tensors)
        if len(tracks) > 0:
            detection_bounding_boxes = tensors[:, :4]
            track_bounding_boxes = np.asarray([track.tlbr for track in tracks])

            ious = matching.box_iou_batch(detection_bounding_boxes, track_bounding_boxes)
            iou_costs = 1 - ious

            matches, _, _ = matching.linear_assignment(iou_costs, 0.5)

            tracked_detections = np.zeros((len(matches), 6), dtype=np.float32)
            tracked_detections[:, :4] = detection_bounding_boxes[matches[:, 0]]
            tracked_detections[:, 4] = class_ids[matches[:, 0]]
            tracked_detections[:, 5] = np.array([int(tracks[i_track].external_track_id) for _, i_track in matches])

            return tracked_detections
        else:
            return np.empty((0, 6), dtype=np.float32)

    def update_with_tensors(self, tensors: np.ndarray) -> List[STrack]:
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []
        class_ids = tensors[:, 5]
        scores = tensors[:, 4]
        bboxes = tensors[:, :4]
        remain_inds = scores > self.track_activation_threshold
        inds_low = scores > 0.1
        inds_high = scores < self.track_activation_threshold
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        class_ids_keep = class_ids[remain_inds]
        class_ids_second = class_ids[inds_second]

        detections = self._create_detections(dets, scores_keep, class_ids_keep, self.minimum_consecutive_frames)
        detections_second = self._create_detections(dets_second, scores_second, class_ids_second,
                                                    self.minimum_consecutive_frames)

        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []

        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = self._joint_tracks(tracked_stracks, self.lost_tracks)
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.minimum_matching_threshold)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.TRACKED]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.LOST:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.TRACKED]
        self.tracked_tracks = self._joint_tracks(self.tracked_tracks, activated_stracks)
        self.tracked_tracks = self._joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = self._sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = self._sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks.extend(removed_stracks)
        self.tracked_tracks, self.lost_tracks = self._remove_duplicate_tracks(self.tracked_tracks, self.lost_tracks)
        output_stracks = [track for track in self.tracked_tracks if track.is_activated]
        return output_stracks

    @staticmethod
    def _convert_detections_to_tensors(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        return np.hstack((boxes, scores[:, np.newaxis], class_ids[:, np.newaxis]))

    @staticmethod
    def _create_detections(dets: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                           minimum_consecutive_frames: int) -> List[STrack]:
        detections = []
        if len(dets) > 0:
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, c, minimum_consecutive_frames)
                for tlbr, s, c in zip(dets, scores, class_ids)
            ]
        return detections

    @staticmethod
    def _joint_tracks(track_list_a: List[STrack], track_list_b: List[STrack]) -> List[STrack]:
        seen_track_ids = set()
        result = []
        for track in track_list_a + track_list_b:
            if track.internal_track_id not in seen_track_ids:
                seen_track_ids.add(track.internal_track_id)
                result.append(track)
        return result

    @staticmethod
    def _sub_tracks(track_list_a: List, track_list_b: List) -> List[int]:
        tracks = {track.internal_track_id: track for track in track_list_a}
        track_ids_b = {track.internal_track_id for track in track_list_b}
        for track_id in track_ids_b:
            tracks.pop(track_id, None)
        return list(tracks.values())

    @staticmethod
    def _remove_duplicate_tracks(tracks_a: List, tracks_b: List) -> Tuple[List, List]:
        pairwise_distance = matching.iou_distance(tracks_a, tracks_b)
        matching_pairs = np.where(pairwise_distance < 0.15)
        duplicates_a, duplicates_b = set(), set()
        for track_index_a, track_index_b in zip(*matching_pairs):
            time_a = tracks_a[track_index_a].frame_id - tracks_a[track_index_a].start_frame
            time_b = tracks_b[track_index_b].frame_id - tracks_b[track_index_b].start_frame
            if time_a > time_b:
                duplicates_b.add(track_index_b)
            else:
                duplicates_a.add(track_index_a)
        result_a = [track for index, track in enumerate(tracks_a) if index not in duplicates_a]
        result_b = [track for index, track in enumerate(tracks_b) if index not in duplicates_b]
        return result_a, result_b
