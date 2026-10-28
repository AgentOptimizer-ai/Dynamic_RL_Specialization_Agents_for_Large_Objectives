
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .task import Task, TaskInstance


"""
WHAT: 
    represents an user-defined environment the agent operates in.
    each environment has multiple tasks. 

ATTRIBUTES:
    - name: unique identifier
    - tasks: the defined tasks within the environment the agent tries to learn, a type of problem within an environment.
"""
class Environment(ABC):

    def __init__(self, name: str, tasks: list["Task"]):
        if not tasks:
            raise ValueError("Environment must contain at least one task")

        self.name = name
        self.tasks = tasks
    
    """
    Return all tasks defined in this environment
    """
    def get_tasks(self) -> list["Task"]:
        return self.tasks


    """
    Prepares environment for a new task instance. Called when episode starts.
    """
    @abstractmethod
    def reset(self, instance: "TaskInstance") -> Any:
        pass


    """
    Execute an agent action:
    Returns:
        - observation: result of executing action, the info the environment retusn to agent about its current situation.
        - done: whether the episode has ended.
    """
    @abstractmethod
    def step(self, action: Any) -> tuple[Any, bool]:
        pass
