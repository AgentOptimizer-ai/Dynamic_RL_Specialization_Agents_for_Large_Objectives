
"""
FILE: to test if we can create a large objective with environments,
      tasks, task instances, and verifiers.

RUN (root):

    python -m tests.test_input_creation

"""

import pytest

from drlsalo.large_objective import LargeObjective
from drlsalo.environment import Environment
from drlsalo.task import Task, TaskInstance
from drlsalo.verifier import Verifier


# SIMPLE VERIFIER FOR TESTING
class TestVerifier(Verifier):

    def evaluate(self, instance, trajectory, final_state) -> float:
        return 1.0 if final_state["solved"] else 0.0


# SIMPLE ENVIRONMENT FOR TESTING
class TestEnvironment(Environment):

    def reset(self, instance):
        self.state = instance.initial_state.copy()
        return self.state

    def step(self, action):
        if action == "solve":
            self.state["solved"] = True
            return "Problem solved", True

        return "Unknown action", False


# GENERATE NEW TASK INSTANCES
def generate_instance():
    return TaskInstance(
        task_name="Solve Problem",
        description="Solve the generated problem",
        initial_state={"solved": False}
    )


@pytest.fixture(scope="module")
def objective():
    print("\n----------CREATING LARGE OBJECTIVE----------")

    # 1. CREATE TASKS
    task_a = Task(
        name="Task A",
        description="Solve problems of type A",
        generator=generate_instance,
        verifier=TestVerifier()
    )

    task_b = Task(
        name="Task B",
        description="Solve problems of type B",
        generator=generate_instance,
        verifier=TestVerifier()
    )

    # 2. CREATE ENVIRONMENTS
    env_a = TestEnvironment(
        name="Environment A",
        tasks=[task_a]
    )

    env_b = TestEnvironment(
        name="Environment B",
        tasks=[task_b]
    )

    # 3. CREATE LARGE OBJECTIVE
    return LargeObjective(
        description="Learn to solve problems across environments",
        environments=[env_a, env_b],
        training_budget=1000
    )


def test_large_objective_creation(objective):
    print("\n----------TESTING LARGE OBJECTIVE----------")

    assert isinstance(objective, LargeObjective)
    assert objective.training_budget == 1000
    assert len(objective.environments) == 2


def test_environment_creation(objective):
    print("\n----------TESTING ENVIRONMENTS----------")

    env_a, env_b = objective.environments

    assert isinstance(env_a, Environment)
    assert isinstance(env_b, Environment)

    assert env_a.name == "Environment A"
    assert env_b.name == "Environment B"

    assert len(env_a.get_tasks()) == 1
    assert len(env_b.get_tasks()) == 1


def test_task_creation(objective):
    print("\n----------TESTING TASKS----------")

    tasks = objective.get_tasks()

    assert len(tasks) == 2
    assert tasks[0].name == "Task A"
    assert tasks[1].name == "Task B"

    assert isinstance(tasks[0].verifier, Verifier)
    assert callable(tasks[0].generator)


def test_task_instance_creation(objective):
    print("\n----------TESTING TASK INSTANCE GENERATION----------")

    task = objective.get_tasks()[0]

    instance_1 = task.generate_instance()
    instance_2 = task.generate_instance()

    assert isinstance(instance_1, TaskInstance)
    assert instance_1.task_name == "Solve Problem"
    assert instance_1.initial_state["solved"] is False

    # VERIFY UNIQUE INSTANCE IDS
    assert instance_1.id != instance_2.id


def test_environment_interaction(objective):
    print("\n----------TESTING ENVIRONMENT INTERACTION----------")

    env = objective.environments[0]
    task = env.get_tasks()[0]
    instance = task.generate_instance()

    # RESET ENVIRONMENT
    observation = env.reset(instance)
    assert observation["solved"] is False

    # EXECUTE ACTION
    observation, done = env.step("solve")

    assert observation == "Problem solved"
    assert done is True
    assert env.state["solved"] is True


def test_verifier(objective):
    print("\n----------TESTING VERIFIER----------")

    env = objective.environments[0]
    task = env.get_tasks()[0]
    instance = task.generate_instance()

    # RESET AND SOLVE INSTANCE
    env.reset(instance)
    env.step("solve")

    # EVALUATE FINAL STATE
    reward = task.verifier.evaluate(
        instance=instance,
        trajectory=[],
        final_state=env.state
    )

    assert reward == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--disable-warnings"]))
