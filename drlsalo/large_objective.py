"""
FILE: represents the large objective that our algorithm is trying to optimize, we are given this as input by user.


"""

from dataclasses import dataclass, field
from .environment import Environment

"""
WHAT: 
    represents a large objective which contains multiple environments.

ATTRIBUTES:
    - description: string that describes overall goal of the large objective.
    - environments: list of environment-objs for this large objective.
    - training_budget: maximum number of training rollouts for now.
"""
class LargeObjective:

    def __init__(self, description: str, environments: list[Environment] | None = None, training_budget: int = 1000):
        self.description = description
        self.environments = environments if environments is not None else []
        self.training_budget = training_budget

        if self.training_budget <= 0:
            raise ValueError("Training budget must be positive")
        if not self.environments:
            raise ValueError("At least one environment is required")
        

    """
    Return all tasks across every environment in large objective.    
    """
    def get_tasks(self):
        tasks_list = []

        for environment in self.environments:
            for task in environment.get_tasks():
                tasks_list.append(task)
        return tasks_list


