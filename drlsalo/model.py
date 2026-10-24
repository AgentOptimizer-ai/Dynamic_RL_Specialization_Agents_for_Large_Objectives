
"""
FILE: loads and manages the shared frozen open-weight LLM with independently trainable LoRA adapters that don't affect each other.

"""

from __future__ import annotations

from copy import deepcopy

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelManager:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # load tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # load base-llm
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        # the actual shared LLM loaded into memory, freeze base model parameters because we have LoRA adapters that update during RL training.
        self.model.requires_grad_(False)

        # config settings for all LoRA adpaters.
        self.lora_config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], bias="none")

        # store all registered agent LoRA-adapters
        self.adapters: set[str] = set()


    """
    WHAT: create a new LoRA adapter
    """
    def create_adapter(self, name: str) -> None:
        if name in self.adapters:
            raise ValueError(f"Adapter already exists: {name}")

        # add the new adapter to the model LLM
        self.model.add_adapter(deepcopy(self.lora_config), adapter_name=name)
        # add new adapter to list of adapters
        self.adapters.add(name)
        # make sure to activate adapter
        self.activate(name)

    """
    WHAT: activate a given LoRA adapter that belongs to an agent.
    """
    def activate(self, name: str) -> None:
        if name not in self.adapters:
            raise KeyError(f"Unknown adapter: {name}")
        self.model.set_adapter(name)

    """
    WHAT: get the LoRA adapter parameters
    """
    def adapter_parameters(self, name: str):
        if name not in self.adapters:
            raise KeyError(name)
        # return its parameters
        return {
            key: param
            for key, param in self.model.named_parameters()
            if f".{name}." in key and "lora_" in key
        }

    """
    WHAT: clone LoRA adapter.
    ARGS:
        parent: which agent we need to copy from, parent adapter-name. 
        child: what to name the new agent's adapter, child adapter-name.
    """
    def clone_adapter(self, parent: str, child: str) -> None:
        # we create specialist-agent by copying its parent's LoRA adapter
        if parent not in self.adapters:
            raise KeyError(parent)

        if child in self.adapters:
            raise ValueError(f"Adapter already exists: {child}")

        # get the parent's LoRA parameters
        source = {
            key.replace(f".{parent}.", f".{child}."):
                param.detach().clone()
            for key, param in self.adapter_parameters(parent).items()
        }

        # create the specialist adapter
        self.create_adapter(child)                  # make sure adapter is saved as well
        target = self.adapter_parameters(child)     # get the child-adapter paramters

        if source.keys() != target.keys():
            raise RuntimeError("Parent and child LoRA parameter layouts differ")

        # copy parent-adapter weights into child-adapter
        with torch.no_grad():
            for key, param in target.items():
                param.copy_(source[key])

    """
    WHAT: generate a response using the given agent-adapter
    ARGs:
        adapter: the lora-adapter we want to use
        prompt: prompt to generate infernece
    """
    @torch.inference_mode()
    def generate(self, adapter: str, prompt: str, max_new_tokens: int = 48) -> str:
        # use selected lora-adapter for the agent we want to run inference
        self.activate(adapter)
        self.model.eval()

        messages = [{"role": "user", "content": prompt}]

        # convert chat messages into tokens using tokenizer
        inputs = self.tokenizer.apply_chat_template( messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True ).to(self.device)

        # generate reponse using active lora-adapter
        output = self.model.generate( **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id )

        # decode output tokens using tokenizer.
        response = self.tokenizer.decode( output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True )

        return response