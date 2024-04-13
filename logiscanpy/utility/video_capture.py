import logging
import time
from typing import Optional, Union

import av
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoCapture:
    """Class for capturing video streams from RTSP URLs or local video files.

    Handles network errors, reconnects for RTSP streams, and mitigates frame loss."""

    def __init__(self, source: Union[str, int], rtsp: bool = False, reconnect_attempts: int = 5,
                 reconnect_interval: float = 5.0, rtsp_options: Optional[dict] = None):
        """Initializes the VideoCapture instance.

        Args:
            source: The RTSP URL of the video stream or the path to the local video file.
            rtsp: Boolean flag indicating whether the source is an RTSP URL or a local file.
            reconnect_attempts (optional): Number of attempts to reconnect on network errors (default: 5).
            reconnect_interval (optional): Time to wait between reconnect attempts (default: 5 seconds).
            rtsp_options (optional): Additional options for RTSP streams (default: None).
        """
        self._source = source
        self._cap = None
        self._rtsp = rtsp
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_interval = reconnect_interval
        self._rtsp_options = rtsp_options or {}

        if self._rtsp:
            self._rtsp_cap = RTSPVideoCapture(source, reconnect_attempts, reconnect_interval, self._rtsp_options)
        else:
            try:
                self._cap = cv2.VideoCapture(source)
                if not self._cap.isOpened():
                    logger.error("Invalid video source")
                    self._cap = None
            except Exception as e:
                logger.error(f"Error opening video source: {e}")
                self._cap = None

    def read(self) -> Optional[np.ndarray]:
        """Reads a frame from the video stream, handling reconnections for RTSP.

        Returns:
            A numpy array representing the frame, or None if an error occurs
            or the stream is not open.
        """
        if self._rtsp:
            return self._rtsp_cap.read()
        elif self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                return None
            return frame
        else:
            return None

    def release(self) -> None:
        """Releases the video stream."""
        if self._rtsp:
            self._rtsp_cap.release()
        elif self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        """Checks if the video capture is opened successfully.

        Returns:
            True if the capture is opened, False otherwise.
        """
        if self._rtsp:
            return self._rtsp_cap.is_opened()
        else:
            return self._cap is not None and self._cap.isOpened()


class RTSPVideoCapture:
    """Class for capturing video streams from RTSP URLs, with reconnection logic."""

    def __init__(self, rtsp_url: str, reconnect_attempts: int = 5, reconnect_interval: float = 5.0,
                 rtsp_options: Optional[dict] = None):
        """Initializes the RTSPVideoCapture instance.

        Args:
            rtsp_url: The RTSP URL of the video stream.
            reconnect_attempts (optional): Number of attempts to reconnect on network errors (default: 5).
            reconnect_interval (optional): Time to wait between reconnect attempts (default: 5 seconds).
            rtsp_options (optional): Additional options for RTSP streams (default: None).
        """
        self._rtsp_url = rtsp_url
        self._container = None
        self._video_stream = None
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_interval = reconnect_interval
        self._last_frame = None
        self._rtsp_options = rtsp_options or {}

    def open_stream(self) -> None:
        """Opens the RTSP video stream."""
        self.close_stream()
        for attempt in range(self._reconnect_attempts):
            try:
                options = {'rtsp_transport': 'tcp', **self._rtsp_options}
                self._container = av.open(self._rtsp_url, mode='r', options=options)
                self._video_stream = next((stream for stream in self._container.streams if stream.type == 'video'),
                                          None)
                return
            except av.error.HTTPError as ex:
                logger.error(f"Error opening RTSP stream (attempt {attempt + 1}/{self._reconnect_attempts}): {ex}")
                time.sleep(self._reconnect_interval)
            except Exception as ex:
                logger.error(f"Error opening RTSP stream (attempt {attempt + 1}/{self._reconnect_attempts}): {ex}")
                time.sleep(self._reconnect_interval)

        logger.error(f"Failed to open RTSP stream after {self._reconnect_attempts} attempts.")

    def close_stream(self) -> None:
        """Closes the RTSP video stream."""
        if self._container:
            try:
                self._container.close()
            except Exception as e:
                logger.error(f"Error closing RTSP stream: {e}")
            finally:
                self._container = None
                self._video_stream = None

    def read(self) -> Optional[np.ndarray]:
        """Reads a frame from the RTSP video stream, handling reconnections and frame loss.

        Returns:
            A numpy array representing the frame, or None if an error occurs
            or the stream is not open.
        """
        if self._container is None:
            self.open_stream()
            if self._container is None:
                return None

        try:
            for packet in self._container.demux(video=0):
                for frame in packet.decode():
                    if frame:
                        self._last_frame = frame.to_ndarray(format='bgr24')
                        return self._last_frame

        except av.error.HTTPError as ex:
            logger.error(f"Error reading RTSP stream: {ex}")
            self.close_stream()
            time.sleep(self._reconnect_interval)
            self.open_stream()
        except Exception as ex:
            logger.error(f"Error reading RTSP stream: {ex}")
            self.close_stream()
            time.sleep(self._reconnect_interval)
            self.open_stream()

        return self._last_frame

    def release(self) -> None:
        """Releases the RTSP video stream."""
        self.close_stream()

    def is_opened(self) -> bool:
        """Checks if the RTSP video capture is opened successfully.

        Returns:
            True if the capture is opened, False otherwise.
        """
        return self._container is not None
