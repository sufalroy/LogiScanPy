from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Solution(ABC):
    """Abstract base class for solutions in the LogiScanPy application.

    This class defines the interface for all solution types in the application.
    Each solution type must implement the methods defined in this class.
    """

    @abstractmethod
    def get_action_data(self) -> Any:
        """Retrieves data required for the action associated with this solution.

        Returns:
            Any: The data required for the action. The type of data returned depends on the specific solution.
        """

    @abstractmethod
    def reset(self) -> None:
        """Resets the solution to its initial state.

        This method is called to clear any internal state of the solution, allowing it to be reused or reinitialized.
        """

    @abstractmethod
    def process_frame(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """Processes a video frame with detected tracks.

        This method takes a video frame and a set of detected tracks, processes them according to the solution's logic,
        and returns the processed frame.

        Args:
            frame (np.ndarray): The input video frame.
            tracks (np.ndarray): The detected tracks in the frame.

        Returns:
            np.ndarray: The processed frame.
        """

    @abstractmethod
    def set_params(self, **kwargs: Any) -> None:
        """Sets the parameters for the solution.

        This method allows the solution to be configured with various parameters.
        The parameters are passed as keyword arguments.

        Args:
            **kwargs: Keyword arguments containing the parameters.
        """
