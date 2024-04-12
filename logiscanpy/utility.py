import threading
import time
from typing import Optional, Union

import av
import cv2
import numpy as np


class VideoCapture:
    """Class for capturing video streams from RTSP URLs or local video files."""

    def __init__(self, source: Union[str, int]):
        """Initializes the VideoCapture instance.

        Args:
            source: The RTSP URL of the video stream or the path to the local video file.
        """
        self.source = source
        self.cap = None
        self.is_rtsp = False

        try:
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                raise ValueError("Invalid video source")
        except ValueError:
            self.is_rtsp = True
            self.rtsp_cap = RTSPVideoCapture(source)

    def read(self) -> Optional[np.ndarray]:
        """Reads a frame from the video stream.

        Returns:
            A numpy array representing the frame, or None if an error occurs
            or the stream is not open.
        """
        if self.is_rtsp:
            return self.rtsp_cap.read()
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

    def release(self) -> None:
        """Releases the video stream."""
        if self.is_rtsp:
            self.rtsp_cap.release()
        else:
            self.cap.release()

    def __del__(self):
        """Destructor to release the video stream."""
        self.release()


class RTSPVideoCapture:
    """Class for capturing video streams from RTSP URLs."""

    def __init__(self, rtsp_url: str):
        """Initializes the RTSPVideoCapture instance.

        Args:
            rtsp_url: The RTSP URL of the video stream.
        """
        self.rtsp_url = rtsp_url
        self.container = None
        self.video_stream = None

    def open_stream(self) -> None:
        """Opens the RTSP video stream."""
        self.close_stream()
        try:
            self.container = av.open(self.rtsp_url, mode='r', options={'rtsp_transport': 'tcp'})
            self.video_stream = next((stream for stream in self.container.streams if stream.type == 'video'), None)
        except av.error.HTTPError as ex:
            self._log_and_close_stream(ex)
        except Exception as ex:
            self._log_and_close_stream(ex)

    def close_stream(self) -> None:
        """Closes the RTSP video stream."""
        if self.container:
            self.container.close()
            self.container = None
            self.video_stream = None

    def read(self) -> Optional[np.ndarray]:
        """Reads a frame from the RTSP video stream.

        Returns:
            A numpy array representing the frame, or None if an error occurs
            or the stream is not open.
        """
        try:
            if self.container is None:
                self.open_stream()
                if self.container is None:
                    return None

            for packet in self.container.demux(video=0):
                for frame in packet.decode():
                    if frame:
                        return frame.to_ndarray(format='bgr24')

        except av.error.HTTPError as ex:
            self._log_and_close_stream(ex)
            time.sleep(5)
        except Exception as ex:
            self._log_and_close_stream(ex)
            time.sleep(5)

        return None

    def release(self) -> None:
        """Releases the RTSP video stream."""
        self.close_stream()

    @staticmethod
    def _log_and_close_stream(exception: Exception) -> None:
        """Logs an exception and closes the stream.

        Args:
            exception: The exception to be logged.
        """
        print(f"Error: {str(exception)}")


class ThreadedFrameCapture(threading.Thread):
    def __init__(self, capture, name='TFC'):
        self.capture = capture
        assert self.capture.isOpened()
        self.cond = threading.Condition()
        self.running = False
        self.frame = None
        self.latestnum = 0
        self.callback = None

        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            (rv, img) = self.capture.read()
            assert rv
            counter += 1

            with self.cond:
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum + 1
                if seqnumber < 1:
                    seqnumber = 1

                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return self.latestnum, self.frame

            return self.latestnum, self.frame
