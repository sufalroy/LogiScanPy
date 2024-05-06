from abc import ABC, abstractmethod
from logiscanpy.core.solutions import Solution


class Action(ABC):
    """Abstract base class for actions in the LogiScanPy application."""

    @abstractmethod
    def execute(self, solution: Solution) -> None:
        """Executes the action with the given solution.

        Args:
           solution (Solution): The solution object.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleans up resources used by the action."""
        pass
