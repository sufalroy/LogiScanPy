from typing import Tuple, Dict, List

import cv2
import numpy as np
import torch
from numpy import ndarray
from openvino import Core, CompiledModel
from ultralytics.utils import ops

from logiscanpy.core.detection import Detector


class YOLOv8OV(Detector):
    """YOLOv8 object detection and instance segmentation model using OpenVINO.

    Args:
        model_path (str): Path to the OpenVINO YOLOv8 detection model.
        confidence_threshold (float): Minimum confidence threshold for detection.
        iou_threshold (float): Intersection over Union (IoU) threshold for non-maximum suppression.
        agnostic_nms (bool): Whether to apply class-agnostic non-maximum suppression.
        max_detections (int): Maximum number of detections after non-maximum suppression.
        retina_mask (bool): Whether to use retina mask postprocessing instead of native decoding.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.7, agnostic_nms: bool = False, max_detections: int = 300,
                 retina_mask: bool = False):
        """Initialize YOLOv8OV object with specified parameters."""
        super().__init__(model_path, confidence_threshold, iou_threshold)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.agnostic_nms = agnostic_nms
        self.max_detections = max_detections
        self.retina_mask = retina_mask
        self.det_compiled_model = self._build_ov_model(model_path)

    def preprocess(self, img0: np.ndarray) -> np.ndarray:
        """Preprocess image according to YOLOv8 input requirements. Takes image in np.array format, resizes it to
        specific size using letterbox resize and changes data layout from HWC to CHW.

        Args:
            img0 (np.ndarray): Image for preprocessing.

        Returns:
            img (np.ndarray): Image after preprocessing.
        """
        img = self._letterbox(img0)[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img

    def postprocess(
            self,
            pred_boxes: np.ndarray,
            input_hw: Tuple[int, int],
            orig_img: np.ndarray,
            pred_masks: np.ndarray = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        YOLOv8 model postprocessing function. Applied non-maximum suppression algorithm to detections and rescale boxes
        to original image size

        Args: pred_boxes (np.ndarray): model output prediction boxes. input_hw (np.ndarray): preprocessed image
        orig_img (np.ndarray): image before preprocessing pred_masks (np.ndarray, optional): Model output prediction
        masks. If not provided, only boxes will be postprocessed.

        Returns:
             pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format
             [x1, y1, x2, y2, score, label] and segment - segmentation polygons for each element in batch
        """
        nms_kwargs = {"agnostic": self.agnostic_nms, "max_det": self.max_detections}
        preds = ops.non_max_suppression(
            torch.from_numpy(pred_boxes),
            self.confidence_threshold,
            self.iou_threshold,
            nc=1,
            **nms_kwargs
        )

        results = []
        proto = torch.from_numpy(pred_masks) if pred_masks is not None else None
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            if not len(pred):
                results.append({"det": [], "segment": []})
                continue

            if proto is None:
                pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
                results.append({"det": pred})
                continue

            if self.retina_mask:
                pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])
                segments = [ops.scale_coords(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)
                pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
                segments = [ops.scale_coords(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]

            results.append({"det": pred[:, :6].numpy(), "segment": segments})

        return results

    def detect(self, image: np.ndarray) -> ndarray:
        """OpenVINO YOLOv8 model inference function. Preprocess image, runs model inference and postprocess results
        using NMS.

            Args:
                image (np.ndarray): input image.

              Returns:
                  np.ndarray: Array of detections in the format [x1, y1, x2, y2, score, label].
        """
        num_outputs = len(self.det_compiled_model.outputs)
        preprocessed_image = self.preprocess(image)
        input_tensor = self._image_to_tensor(preprocessed_image)
        result = self.det_compiled_model(input_tensor)
        boxes = result[self.det_compiled_model.output(0)]
        masks = None
        if num_outputs > 1:
            masks = result[self.det_compiled_model.output(1)]
        input_hw = input_tensor.shape[2:]
        detections = self.postprocess(
            pred_boxes=boxes,
            input_hw=input_hw,
            orig_img=image,
            pred_masks=masks,
        )
        return detections[0]["det"]

    @staticmethod
    def _build_ov_model(det_model_path: str) -> CompiledModel:
        core = Core()
        det_ov_model = core.read_model(det_model_path)
        return core.compile_model(det_ov_model, 'CPU')

    @staticmethod
    def _letterbox(
            img: np.ndarray,
            new_shape: Tuple[int, int] = (640, 640),
            color: Tuple[int, int, int] = (114, 114, 114),
            auto: bool = False,
            scale_fill: bool = False,
            scaleup: bool = False,
            stride: int = 32,
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """Resize image and padding for detection. Takes image as input, resizes image to fit into new shape with
        saving original aspect ratio and pads it to meet stride-multiple constraints

        Args:
            img (np.ndarray): Image for preprocessing.
            new_shape (Tuple[int, int]): Image size after preprocessing in format [height, width].
            color (Tuple[int, int, int]): Color for filling padded area.
            auto (bool): Use dynamic input size, only padding for stride constraints applied.
            scale_fill (bool): Scale image to fill new_shape.
            scaleup (bool): Allow scale image if it is lower than desired input size, can affect model accuracy.
            stride (int): Input padding stride.

        Returns:
            Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
             Preprocessed image, height and width scaling ratios, height and width padding sizes.
        """
        shape = img.shape[:2]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)

    @staticmethod
    def _image_to_tensor(image: np.ndarray) -> np.ndarray:
        """Preprocess image according to YOLOv8 input requirements. Takes image in np.array format, resizes it to
        specific size using letterbox resize and changes data layout from HWC to CHW.

        Args:
            image (np.ndarray): Image for preprocessing.

        Returns:
            input_tensor (np.ndarray): Input tensor in NCHW format with float32 values in [0, 1] range.
        """
        input_tensor = image.astype(np.float32)
        input_tensor /= 255.0
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor
