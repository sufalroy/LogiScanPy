import logging

from logiscanpy.core.actions import Action
from logiscanpy.core.solutions import Solution
from logiscanpy.utility.publisher import Publisher

_LOGGER = logging.getLogger(__name__)


class CountPublisher(Action):
    """Publishes object counts to a RabbitMQ exchange.

    This action publishes the counts of detected objects to a RabbitMQ exchange.
    It only publishes counts that have changed since the last execution.
    """

    def __init__(self):
        """Initializes the CountPublisher with a Publisher instance and an empty dictionary for previous counts."""
        self._publisher = Publisher()
        self._previous_counts = {}

    def execute(self, solution: Solution) -> None:
        """Publishes counts for detected objects.

        This method iterates over the provided data, which should be a dictionary mapping
        object class names to their counts. It only publishes counts that have changed
        since the last execution.

        Args:
            solution (Solution): The solution object.
        """
        for class_name, count in solution.get_action_data().items():
            if class_name not in self._previous_counts or count != self._previous_counts[class_name]:
                _LOGGER.debug("Publishing count for %s: %d", class_name, count)
                self._publisher.publish_message(class_name, count)
                self._previous_counts[class_name] = count

    def cleanup(self) -> None:
        """Closes the RabbitMQ connection used by the Publisher."""
        self._publisher.close_connection()
