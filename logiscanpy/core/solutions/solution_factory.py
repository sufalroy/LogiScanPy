from typing import Dict

from logiscanpy.core.solutions import Solution
from logiscanpy.core.solutions.object_counter import ObjectCounter
from logiscanpy.core.solutions.time_tracker import TimeTracker


class SolutionFactory:

    @staticmethod
    def create_solution(config: Dict[str, str]) -> Solution:
        """Creates a solution instance based on the configuration.

        Args:
            config (Dict[str, str]): Configuration dictionary.

         Returns:
            Solution: The created solution instance.

        Raises:
            ValueError: If an invalid solution type is specified in the configuration.
        """
        solution_type = config.get('solution_type', 'object_counter')
        solution_class = {
            'object_counter': ObjectCounter,
            'time_tracker': TimeTracker,
        }.get(solution_type, None)

        if solution_class is None:
            raise ValueError(f'Invalid solution type: {solution_type}')

        return solution_class()
