from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort


class YOLOv8:
    """YOLOv8 detection model."""

    def __init__(self, onnx_model: str, confidence_thres: float = 0.5, iou_thres: float = 0.45):
        """Initialize YOLOv8 object.

        Args:
            onnx_model (str): Path to the ONNX model file.
            confidence_thres (float): Confidence threshold for detection.
            iou_thres (float): Intersection over Union threshold for non-maximum suppression.
        """
        self.session = self._build_ort_session(onnx_model)
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.input_height, self.input_width = self.session.get_inputs()[0].shape[2:4]

    @staticmethod
    def _build_ort_session(onnx_model: str) -> ort.InferenceSession:
        """Build the ONNX Runtime session.

        Args:
            onnx_model (str): Path to the ONNX model.

        Returns:
            ort.InferenceSession: The ONNX Runtime session.
        """
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" \
            else ["CPUExecutionProvider"]
        return ort.InferenceSession(onnx_model, providers=providers)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess the input image for ONNX model.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Preprocessed image data.
        """
        self.img_height, self.img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def postprocess(self, outputs, target_class_ids=None) -> np.ndarray:
        """Postprocess the output of the ONNX model.

        Args:
            outputs (np.ndarray): Output from the ONNX model.
            target_class_ids (List[int], optional): List of class IDs to detect. If None, all classes are detected.

        Returns:
            np.ndarray: Array of detections in the format [x1, y1, x2, y2, score, class_id].
        """
        outputs = np.transpose(np.squeeze(outputs))
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        mask = np.amax(outputs[:, 4:], axis=1) >= self.confidence_thres

        if target_class_ids is not None:
            class_ids = np.argmax(outputs[:, 4:], axis=1)
            mask &= np.isin(class_ids, target_class_ids)

        outputs = outputs[mask]

        boxes = np.concatenate((
            (outputs[:, :2] - outputs[:, 2:4] / 2) * np.array([x_factor, y_factor]),
            (outputs[:, :2] + outputs[:, 2:4] / 2) * np.array([x_factor, y_factor])
        ), axis=1).astype(int)
        class_ids = np.argmax(outputs[:, 4:], axis=1).astype(np.int32)
        scores = np.amax(outputs[:, 4:], axis=1)

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.confidence_thres, self.iou_thres)
        boxes = boxes[indices]
        class_ids = class_ids[indices]
        scores = scores[indices]
        detections = np.hstack((boxes, scores[:, np.newaxis], class_ids[:, np.newaxis]))
        return detections

    def detect(self, img, target_class_ids=None) -> np.ndarray:
        """Detect objects in the input image.

        Args:
            img (np.ndarray): Input image.
            target_class_ids (List[int], optional): List of class IDs to detect. If None, all classes are detected.

        Returns:
            np.ndarray: Array of detections in the format [x1, y1, x2, y2, score, class_id].
        """
        img_data = self.preprocess(img)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
        return self.postprocess(outputs, target_class_ids)


class YOLOv8Seg:
    """YOLOv8 segmentation model."""

    def __init__(self, onnx_model: str):
        """Initialize the YOLOv8 segmentation model.

        Args:
            onnx_model (str): Path to weights file.
        """
        self.session = self._build_ort_session(onnx_model)
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

    @staticmethod
    def _build_ort_session(onnx_model: str) -> ort.InferenceSession:
        """Build the ONNX Runtime session.

        Args:
            onnx_model (str): Path to the ONNX model.

        Returns:
            ort.InferenceSession: The ONNX Runtime session.
        """
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" \
            else ["CPUExecutionProvider"]
        return ort.InferenceSession(onnx_model, providers=providers)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        """Preprocess the input image for inference.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
                img_process (np.ndarray): image preprocessed for inference.
                ratio (Tuple[float, float]): width, height ratios in letterbox.
                pad (Tuple[float, float]): width padding, height padding in letterbox.
        """
        shape = img.shape[:2]
        new_shape = (self.model_height, self.model_width)
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, (ratio, ratio), (pad_w, pad_h)

    def postprocess(
            self,
            preds: np.ndarray,
            im0: np.ndarray,
            ratio: Tuple[float, float],
            pad: Tuple[float, float],
            conf_threshold: float,
            iou_threshold: float,
            nm: int = 32,
            target_class_id: int = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Postprocess the model predictions.

        Args:
            preds (np.ndarray): predictions come from ort.session.run().
            im0 (np.ndarray): [h, w, c] original input image.
            ratio (Tuple[float, float]): width, height ratios in letterbox.
            pad (Tuple[float, float]): width padding, height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.
            target_class_id (int, optional): The class ID to detect. If None, all classes are detected.

        Returns:
            Tuple[List[np.ndarray], List[List[np.ndarray]], np.ndarray]:
                boxes (List[np.ndarray]): list of bounding boxes.
                segments (List[List[np.ndarray]]): list of lists of segments.
                masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]
        x = np.einsum("bcn->bnc", x)
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        if len(x) > 0:
            if target_class_id is not None:
                x = x[x[:, 5] == target_class_id]

            if len(x) > 0:
                x[..., [0, 1]] -= x[..., [2, 3]] / 2
                x[..., [2, 3]] += x[..., [0, 1]]
                x[..., :4] -= [pad[0], pad[1], pad[0], pad[1]]
                x[..., :4] /= min(ratio)
                x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
                x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])
                masks = self._process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
                segments = self._masks2segments(masks)
                return x[..., :6].tolist(), segments, masks
            else:
                empty_boxes: List[np.ndarray] = []
                empty_segments: List[np.ndarray] = []
                empty_masks: np.ndarray = np.array([])
                return empty_boxes, empty_segments, empty_masks
        else:
            empty_boxes: List[np.ndarray] = []
            empty_segments: List[np.ndarray] = []
            empty_masks: np.ndarray = np.array([])
            return empty_boxes, empty_segments, empty_masks

    def detect(
            self,
            im0: np.ndarray,
            conf_threshold: float = 0.4,
            iou_threshold: float = 0.45,
            nm: int = 32,
            class_id: int = 0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Detect objects in the given image.

        Args:
            im0 (np.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.
            nm (int): the number of masks.
            class_id (int, optional): The class ID to detect. If None, all classes are detected.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
                boxes (List[np.ndarray]): list of bounding boxes.
                segments (List[np.ndarray]): list of segments.
                masks (np.ndarray): [N, H, W], output masks.
        """
        im, ratio, pad = self.preprocess(im0)
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})
        boxes, segments, masks = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad=pad,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
            target_class_id=class_id,
        )
        return boxes, segments, masks

    @staticmethod
    def _masks2segments(masks: np.ndarray) -> List[np.ndarray]:
        """Convert masks to segments (contours).

        Args:
            masks (np.ndarray): Masks.

        Returns:
            List[np.ndarray]: List of segments.
        """
        segments = []
        for x in masks.astype("uint8"):
            contours = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            if contours:
                segment = np.array(contours[np.array([len(x) for x in contours]).argmax()]).reshape(-1, 2)
            else:
                segment = np.zeros((0, 2))
            segments.append(segment.astype("float32"))
        return segments

    @staticmethod
    def _crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Crop masks based on bounding boxes.

        Args:
            masks (np.ndarray): Masks.
            boxes (np.ndarray): Bounding boxes.

        Returns:
            np.ndarray: Cropped masks.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def _process_mask(
            self,
            protos: np.ndarray,
            masks_in: np.ndarray,
            bboxes: np.ndarray,
            im0_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Process the output masks.

        Args:
            protos (np.ndarray): Mask prototypes.
            masks_in (np.ndarray): Input masks.
            bboxes (np.ndarray): Bounding boxes.
            im0_shape (Tuple[int, int, int]): Original image shape.

        Returns:
            np.ndarray: Processed masks.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)
        masks = np.ascontiguousarray(masks)
        masks = self._scale_mask(masks, im0_shape)
        masks = np.einsum("HWN -> NHW", masks)
        masks = self._crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def _scale_mask(
            masks: np.ndarray,
            im0_shape: Tuple[int, int, int],
            ratio_pad: Tuple[float, float] = None
    ) -> np.ndarray:
        """Scale masks to the original image size.

        Args:
            masks (np.ndarray): Masks to be scaled.
            im0_shape (Tuple[int, int, int]): Original image shape.
            ratio_pad (Tuple[float, float], optional): Padding ratio.

        Returns:
            np.ndarray: Scaled masks.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2
        else:
            pad = ratio_pad[1]

        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR)
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks
