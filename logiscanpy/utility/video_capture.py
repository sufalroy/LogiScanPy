import logging
import threading
import time
from typing import Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


class VideoCapture:
    """Class for capturing video streams from local video files."""

    def __init__(self, source):
        """Initializes the VideoCapture instance.

        Args:
            source: The path to the local video file.
        """
        self._source = source
        self._cap = None
        self.open_capture()

    def open_capture(self):
        """Opens the video capture."""
        self.close_capture()
        self._cap = cv2.VideoCapture(self._source)

        if not self._cap.isOpened():
            logger.error("Failed to open video capture")
            self._cap = None

    def close_capture(self):
        """Closes the video capture."""
        if self._cap:
            self._cap.release()
            self._cap = None

    def read(self):
        """Reads a frame from the video stream, handling reconnections.

        Returns:
            A numpy array representing the frame, or None if an error occurs
            or the stream is not open.
        """
        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            self.close_capture()

        return frame

    def release(self):
        """Releases the video stream."""
        self.close_capture()

    def is_opened(self):
        """Checks if the video capture is opened successfully.

        Returns:
            True if the capture is opened, False otherwise.
        """
        return self._cap is not None and self._cap.isOpened()


class RtspVideoCapture(threading.Thread):
    """A thread-safe class for capturing video from an RTSP stream."""

    def __init__(self,
                 source: str,
                 reconnect_attempts: int = 5,
                 reconnect_interval: float = 5.0,
                 name: str = 'RtspVideoCapture'):
        """
        Initialize the RtspVideoCapture instance.

        Args:
            source (str): The RTSP stream URL.
            reconnect_attempts (int): The number of attempts to reconnect if the stream is lost.
            reconnect_interval (float): The interval (in seconds) between reconnection attempts.
            name (str): The name of the thread.
        """
        super().__init__(name=name)
        self.source = source
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_interval = reconnect_interval
        self.capture = None
        self.condition_variable = threading.Condition()
        self.running = False
        self.frame = None
        self.latest_sequence_number = 0
        self.callback = None
        self._open_capture()
        self.start()

    def __enter__(self):
        """Enable using the `with` statement for resource management."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting the `with` block."""
        self.release()

    def _open_capture(self) -> None:
        """Open the video capture for the RTSP stream."""
        self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            logger.error("Failed to open video capture")
            self.capture = None

    def _close_capture(self) -> None:
        """Close the video capture."""
        if self.capture:
            self.capture.release()
            self.capture = None

    def start(self) -> None:
        """Start the video capture thread."""
        self.running = True
        super().start()

    def release(self, timeout: Optional[float] = None) -> None:
        """Stop and release the video capture resources."""
        self.running = False
        self.join(timeout=timeout)
        self._close_capture()

    def is_opened(self) -> bool:
        """Check if the video capture is opened."""
        return self.capture is not None and self.capture.isOpened()

    def run(self) -> None:
        """The main loop for video capture and reconnection."""
        sequence_number = 0
        while self.running:
            ret, img = self.capture.read()
            if not ret:
                self._close_capture()
                for attempt in range(self.reconnect_attempts):
                    logger.info(f"Reconnecting... attempt {attempt + 1}/{self.reconnect_attempts}")
                    self._open_capture()
                    if self.capture.isOpened():
                        break
                    time.sleep(self.reconnect_interval)
                if not self.capture.isOpened():
                    logger.error("Failed to reconnect after multiple attempts")
                    break
            else:
                sequence_number += 1
                with self.condition_variable:
                    self.frame = img if ret else None
                    self.latest_sequence_number = sequence_number
                    self.condition_variable.notify_all()
                    if self.callback:
                        self.callback(img)

    def read(self,
             wait: bool = True,
             sequence_number: Optional[int] = None,
             timeout: Optional[float] = None) -> Optional[cv2.Mat]:
        """
        Read the next video frame from the capture.

        Args:
            wait (bool): Whether to wait for a new frame if one is not available.
            sequence_number (int, optional): The sequence number of the desired frame.
            timeout (float, optional): The maximum time to wait for a new frame.

        Returns:
            Optional[cv2.Mat]: the frame data.
        """
        with self.condition_variable:
            if wait:
                if sequence_number is None:
                    sequence_number = self.latest_sequence_number + 1
                if sequence_number < 1:
                    sequence_number = 1
                success = self.condition_variable.wait_for(
                    lambda: self.latest_sequence_number >= sequence_number, timeout=timeout)
                if not success:
                    return self.frame
            return self.frame
