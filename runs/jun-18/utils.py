"""
Utility functions for GRPO training with Triton kernels.
"""

import os
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Optional
from transformers import PreTrainedModel


def seed_everything(seed: int) -> None:
    """Set random seed for reproducibility across multiple libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_generation_log(log_data: Dict[str, Any], log_file: str) -> None:
    """Write generation log data to a text file for Triton kernels."""
    with open(log_file, 'w') as f:
        # Write prompt section
        f.write("###### ORIGINAL PROMPT #####\n\n")
        f.write(log_data['prompt']['text'] + "\n\n")

        # Write each generation
        for i, gen in enumerate(log_data['generations'], 1):
            f.write(f"#### GENERATION {i} RESPONSE ####\n\n")
            f.write(gen['response'] + "\n\n")
            f.write(f"#### GENERATION {i} SCORES ####\n")
            
            # Write individual scores
            f.write(f"Base Correctness: {gen['scores']['base_correctness']:.4f}\n")
            f.write(f"Instruction Adherence: {gen['scores']['instruction_adherence']:.4f}\n") 
            f.write(f"Total Reward: {gen['scores']['total_reward']:.4f}\n\n")

        # Write summary statistics
        if 'summary_stats' in log_data:
            f.write("#### SUMMARY STATISTICS ####\n")
            f.write(f"Mean rewards per group: {log_data['summary_stats']['mean_rewards_per_group']}\n")
            f.write(f"Std rewards per group: {log_data['summary_stats']['std_rewards_per_group']}\n")
            f.write(f"Advantages: {log_data['summary_stats']['advantages']}\n")


def create_sample_dataset(output_path: str, num_samples: int = 50) -> None:
    """Create a sample dataset of Triton kernel prompts."""
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
        "Write an optimized Triton kernel for matrix multiplication with shared memory",
        "Write a Triton kernel for softmax operation",
        "Write a Triton kernel for layer normalization",
        "Write a Triton kernel for batch normalization",
        "Write a Triton kernel for 1D convolution",
        "Write a Triton kernel for average pooling",
        "Write a Triton kernel for max pooling",
        "Write a Triton kernel for element-wise division",
        "Write a Triton kernel for dropout operation",
        "Write a Triton kernel for transpose operation",
        "Write a Triton kernel for broadcast addition"
    ]
    
    dataset = []
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        # Add some variation
        if i > len(prompts):
            variations = [
                " with optimized memory access",
                " using block tiling",
                " with coalesced memory patterns",
                " for large tensors",
                " with vectorized operations"
            ]
            prompt += random.choice(variations)
        dataset.append({"prompt": prompt})
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created sample dataset with {num_samples} prompts at {output_path}")


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    """Get per-token log probabilities from model."""
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    # Only keep the logits we need
    if logits.size(1) > logits_to_keep + 1:
        logits = logits[:, -(logits_to_keep + 1):, :]
    
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit

    input_ids = input_ids[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)


def get_model_memory_usage(model: PreTrainedModel) -> Dict[str, str]:
    """Get memory usage statistics for a model."""
    if not torch.cuda.is_available():
        return {"status": "CUDA not available"}
    
    # Get model size
    param_count = sum(p.numel() for p in model.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    # Get GPU memory stats
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    
    return {
        "parameter_count": f"{param_count:,}",
        "parameter_size": f"{param_size_mb:.2f} MB",
        "gpu_allocated": f"{allocated:.2f} MB",
        "gpu_reserved": f"{reserved:.2f} MB"
    }


def prepare_model_for_training(model: PreTrainedModel) -> PreTrainedModel:
    """Prepare model for training by enabling gradient checkpointing if available."""
    model.train()
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")
    
    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    return model