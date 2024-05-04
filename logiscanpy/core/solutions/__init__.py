from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Solution(ABC):

    @abstractmethod
    def process_frame(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """Processes a video frame with detected tracks.

        Args:
            frame (np.ndarray): The input video frame.
            tracks (np.ndarray): The detected tracks in the frame.

        Returns:
            np.ndarray: The processed frame.
        """

    @abstractmethod
    def set_params(self, **kwargs: Any) -> None:
        """Sets the parameters for the solution.

        Args:
            **kwargs: Keyword arguments containing the parameters.
        """
