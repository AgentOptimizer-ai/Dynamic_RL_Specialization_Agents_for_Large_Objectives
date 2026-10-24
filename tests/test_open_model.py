"""
FILE: to test if we can run the open-weight-local-model used for this research.

RUN (root):

    python -m tests.test_open_model

"""

import pytest
from drlsalo.model import ModelManager


@pytest.fixture(scope="module")
def model():
    # 1. LOAD SHARED LOCAL OPEN-WEIGHT LLM
    print("\n----------Loading Qwen model...----------")
    model = ModelManager()

    # CREATE GENERALISTS LORA ADAPTER
    model.create_adapter("generalist")

    # 4. CLONE GENERALIST INTO SPECIALIST
    model.clone_adapter("generalist", "specialist_1")

    return model


def test_generalist_inference(model):
    # 3. TEST INFERENCE
    print("\n----------TESTING GENERALIST----------:")
    response = model.generate(adapter="generalist", prompt="What is 2 + 2? Answer in one sentence.")
    print(response)

    assert isinstance(response, str)
    assert len(response.strip()) > 0


def test_specialist_inference(model):
    # 5. TEST SPECIALIST INFERNECE
    print("\n----------TESTING SPECIALIST----------:")
    response = model.generate(adapter="specialist_1", prompt="What is the capital of France? Answer briefly.")
    print(response)

    assert isinstance(response, str)
    assert len(response.strip()) > 0


def test_adapters_exist(model):
    # 6. VERIFY BOTH ADAPTERS EXIST
    print("\n----------VERIFY ADAPTERS----------:", model.adapters)

    assert "generalist" in model.adapters
    assert "specialist_1" in model.adapters


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--disable-warnings"]))