import av
import time

import numpy as np
from typing import Optional


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
        """Opens the RTSP video stream.

        This method closes any existing stream and attempts to open a new
        stream using the provided RTSP URL. If an error occurs, the stream
        is closed.
        """
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
