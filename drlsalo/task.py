"""
FILE: define tasks + task instances

"""

from typing import Any, Callable, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .verifier import Verifier

"""
WHAT: represents a task that agent has to learn within an environment.

ATTRIBUTES:
    - name: unique task identifier
    - description: explains what this task is about
    - generator: a function that dynamically generates new task-instance objects for training.
    - verifier: evaluates the agent's attempt at solving a task instance and returns a reward. 
"""
class Task:

    def __init__(self, name: str, description: str, generator: Callable[[], "TaskInstance"], verifier: "Verifier"):
        self.name = name
        self.description = description
        self.generator = generator
        self.verifier = verifier

    """
    Generates a new task instace of this task type
    """
    def generate_instance(self) -> "TaskInstance":
        return self.generator()


"""
WHAT:
    represents a single concrete problem generated from a task.

ATTRIBUTES:
    - id: unique identifier for task instance
    - task_name: name of the task-obj this task-instance belongs to
    - description: string describes specific problem the agent must solve
    - initial_state: starting state of the environment for this instance
"""
class TaskInstance:

    def __init__(self, task_name: str, description: str, initial_state: Any, instance_id: str | None = None):
        self.id = instance_id or str(uuid4())
        self.task_name = task_name
        self.description = description
        self.initial_state = initial_state
