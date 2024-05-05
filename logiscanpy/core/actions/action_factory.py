from logiscanpy.core.actions import Action
from logiscanpy.core.actions.count_publisher import CountPublisher
from logiscanpy.core.actions.time_alert import TimeAlert


class ActionFactory:
    @staticmethod
    def create_action(solution_type: str) -> Action:
        action_classes = {
            'object_counter': CountPublisher,
            'time_tracker': TimeAlert,
        }
        action_class = action_classes.get(solution_type)
        if action_class is None:
            raise ValueError(f'Invalid solution type: {solution_type}')
        return action_class()
