"""
FILE: define user-defined verifier 

"""


from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .task import TaskInstance

"""
WHAT: represents a single verifier for a task

ARGUMENTS:
    - 
"""
class Verifier(ABC):


    """
    Evaluate the agent's attempt adn return a reward.
    """
    @abstractmethod
    def evaluate(self, instance: "TaskInstance", trajectory: list[Any], final_state: Any) -> float:
        """
        Evaluate the agent's attempt and return a reward.
        """
        pass
