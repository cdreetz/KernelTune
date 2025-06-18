"""
Dataset loader for Triton kernel generation training.
Simple prompts only - verification handles everything else.
"""

import json
import random
import numpy as np
from typing import Tuple, Dict, Any, List
from abc import ABC, abstractmethod


class DataLoader(ABC):
    def __init__(self, random: bool = False) -> None:
        self.random = random
        self.current_index = 0

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> 'DataLoader':
        return self

    @abstractmethod
    def __next__(self) -> Any:
        pass


SYSTEM_PROMPT = """You are an expert at writing high-performance GPU kernels using Triton. 
Format your response as:
<think>
Your reasoning process
</think>
<code>
import triton
import triton.language as tl

@triton.jit  
def your_kernel(...):
    # Implementation
    pass
</code>
"""


class TritonKernelLoader(DataLoader):
    def __init__(self, prompts: List[str], random: bool = False) -> None:
        super().__init__(random)
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self) -> 'TritonKernelLoader':
        return self

    def __next__(self) -> str:
        if self.current_index >= len(self.prompts):
            raise StopIteration

        if self.random:
            idx = random.randint(0, len(self.prompts) - 1)
        else:
            idx = self.current_index
            self.current_index += 1

        return self.prompts[idx]

    def reset(self):
        self.current_index = 0


def load_kernel_dataset(dataset_path: str) -> List[str]:
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    prompts = []
    for item in data:
        if isinstance(item, dict):
            if 'prompt' in item:
                prompts.append(item['prompt'])
            elif 'instruction' in item:
                prompts.append(item['instruction'])
            elif 'text' in item:
                prompts.append(item['text'])
        elif isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, list) and len(item) >= 1:
            prompts.append(item[0])

    return prompts


def get_dataloaders(dataset_path: str, test_split: float = 0.1) -> Tuple[TritonKernelLoader, TritonKernelLoader]:
    prompts = load_kernel_dataset(dataset_path)
    
    total_samples = len(prompts)
    test_size = int(total_samples * test_split)
    test_indices = random.sample(range(total_samples), test_size)
    test_indices_set = set(test_indices)
    
    train_prompts = [prompts[i] for i in range(total_samples) if i not in test_indices_set]
    test_prompts = [prompts[i] for i in test_indices]
    
    train_loader = TritonKernelLoader(train_prompts, random=True)
    test_loader = TritonKernelLoader(test_prompts, random=False)
    
    return train_loader, test_loader


def create_simple_test_dataset(output_path: str, num_samples: int = 20):
    prompts = [
        "Write a Triton kernel for element-wise addition of two tensors",
        "Write a Triton kernel for ReLU activation",
        "Write a Triton kernel for matrix multiplication",
        "Write a Triton kernel for sum reduction",
        "Write a Triton kernel for element-wise multiplication",
        "Write a Triton kernel for GELU activation",
        "Write a Triton kernel for max reduction with blocktiling",
        "Write a Triton kernel for sigmoid activation",
        "Write a Triton kernel for element-wise subtraction",
        "Write an optimized Triton kernel for matrix multiplication with shared memory"
    ]
    
    dataset = []
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        dataset.append({"prompt": prompt})
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created test dataset with {num_samples} examples")


if __name__ == "__main__":
    create_simple_test_dataset('test_dataset.json', 10)
    train_loader, test_loader = get_dataloaders('test_dataset.json')
    
    print(f"Train: {len(train_loader)}, Test: {len(test_loader)}")
    prompt = next(train_loader)
    print(f"Example: {prompt}")
    
    import os
    os.remove('test_dataset.json')
    print("✓ Ready")