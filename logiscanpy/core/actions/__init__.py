from abc import ABC, abstractmethod
from typing import Any


class Action(ABC):
    """Abstract base class for actions in the LogiScanPy application."""

    @abstractmethod
    def execute(self, data: Any) -> None:
        """Executes the action with the given data.

        Args:
            data (Any): The data required for the action.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleans up resources used by the action."""
        pass
