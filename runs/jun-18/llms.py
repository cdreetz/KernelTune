"""
Module for loading LLMs and their tokenizers from huggingface.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
from typing import Dict, Tuple


def get_llm_tokenizer(model_name: str, device: str) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load and configure a language model and its tokenizer.

    Args:
        model_name: Name or path of the pretrained model to load
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        tuple containing:
            - The loaded language model
            - The configured tokenizer for that model
    """
    # Load model with appropriate settings
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        "device_map": None,
    }
    
    # Try to use flash attention if available
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            **model_kwargs
        ).to(device)
    except:
        # Fallback to standard attention
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        ).to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Disable caching for training
    model.config.use_cache = False

    return model, tokenizer


def prepare_model_for_training(model: PreTrainedModel) -> PreTrainedModel:
    """
    Prepare model for training by enabling gradient checkpointing and other optimizations.
    
    Args:
        model: The model to prepare for training
        
    Returns:
        The prepared model
    """
    model.train()
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")
    
    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    return model


def get_model_memory_usage(model: PreTrainedModel) -> Dict[str, str]:
    """
    Get memory usage statistics for a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with memory usage information
    """
    if not torch.cuda.is_available():
        return {"status": "CUDA not available"}
    
    # Get model size
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    # Get GPU memory stats
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    
    return {
        "total_parameters": f"{param_count:,}",
        "trainable_parameters": f"{trainable_count:,}",
        "parameter_size": f"{param_size_mb:.2f} MB",
        "gpu_allocated": f"{allocated:.2f} MB",
        "gpu_reserved": f"{reserved:.2f} MB"
    }