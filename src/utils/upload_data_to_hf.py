#!/usr/bin/env python3
"""
Script to upload the SFT dataset to Hugging Face Hub
"""

import os
from datasets import load_from_disk, Dataset
from huggingface_hub import HfApi, login
import argparse

def flatten_metadata(dataset):
    """
    Flatten metadata fields into separate columns
    
    Args:
        dataset: HuggingFace dataset with nested metadata
        
    Returns:
        Dataset with flattened metadata fields
    """
    print("Flattening metadata fields...")
    
    flattened_data = []
    for example in dataset:
        # Start with the main fields
        flattened_example = {
            "prompt": example["prompt"],
            "completion": example["completion"]
        }
        
        # Extract metadata fields as separate columns
        metadata = example.get("metadata", {})
        flattened_example["id"] = metadata.get("id", "")
        flattened_example["operation"] = metadata.get("operation", "")
        flattened_example["type"] = metadata.get("type", "")
        flattened_example["has_triton_docs"] = metadata.get("has_triton_docs", False)
        
        flattened_data.append(flattened_example)
    
    # Create new dataset with flattened structure
    flattened_dataset = Dataset.from_list(flattened_data)
    print(f"✓ Flattened metadata for {len(flattened_dataset)} examples")
    
    return flattened_dataset

def upload_dataset_to_hf(
    dataset_path="sft-triton-dataset-6k",
    repo_name="sft-triton-dataset-6k",
    username=None,
    private=False
):
    """
    Upload the SFT dataset to Hugging Face Hub
    
    Args:
        dataset_path: Path to the local dataset folder
        repo_name: Name for the repository on HF Hub
        username: Your HF username (if None, will be inferred from token)
        private: Whether to make the dataset private
    """
    
    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    try:
        login()  # This will use HF_TOKEN env var or prompt for token
        print("✓ Successfully logged in to Hugging Face")
    except Exception as e:
        print(f"✗ Failed to login: {e}")
        print("Please run: huggingface-cli login")
        return False
    
    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_from_disk(dataset_path)
        print(f"✓ Loaded dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False
    
    # Flatten metadata fields
    dataset = flatten_metadata(dataset)
    
    # Get user info if username not provided
    api = HfApi()
    if username is None:
        try:
            user_info = api.whoami()
            username = user_info["name"]
            print(f"✓ Using username: {username}")
        except Exception as e:
            print(f"✗ Failed to get user info: {e}")
            return False
    
    # Create repository name
    repo_id = f"{username}/{repo_name}"
    
    # Add dataset card content
    dataset_card = f"""---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- triton
- gpu-kernels
- code-generation
- synthetic-data
size_categories:
- 1K<n<10K
---

# Triton Kernel SFT Dataset

This dataset contains {len(dataset)} examples for supervised fine-tuning (SFT) of models to generate Triton GPU kernels.

## Dataset Description

The dataset consists of two types of examples:
1. **Synthetic queries** (60%): Generated queries asking for Triton kernels for various operations
2. **Convert queries** (40%): PyTorch code conversion requests to Triton kernels

## Dataset Structure

Each example contains:
- `prompt`: The instruction/query asking for a Triton kernel
- `completion`: The corresponding Triton kernel implementation
- `id`: Unique identifier for the example
- `operation`: The operation type (e.g., "matmul", "softmax", etc.)
- `type`: The query type ("synthetic" or "convert")
- `has_triton_docs`: Boolean indicating if Triton documentation was used during generation

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

## Generation

This dataset was generated using automated prompting techniques with language models to create diverse training examples for Triton kernel generation.
"""
    
    # Upload the dataset
    print(f"Uploading dataset to {repo_id}...")
    try:
        dataset.push_to_hub(
            repo_id,
            private=private,
            commit_message="Initial upload of Triton SFT dataset with flattened metadata"
        )
        print(f"✓ Successfully uploaded dataset to: https://huggingface.co/datasets/{repo_id}")
        
        # Upload dataset card
        api.upload_file(
            path_or_fileobj=dataset_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card"
        )
        print("✓ Dataset card uploaded")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to upload dataset: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload SFT dataset to Hugging Face Hub")
    parser.add_argument("--dataset-path", default="data/sft_hf_dataset_combined", help="Path to dataset folder")
    parser.add_argument("--repo-name", default="triton-sft-dataset", help="Repository name on HF Hub")
    parser.add_argument("--username", help="HF username (optional, will be inferred)")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    
    args = parser.parse_args()
    
    success = upload_dataset_to_hf(
        dataset_path=args.dataset_path,
        repo_name=args.repo_name,
        username=args.username,
        private=args.private
    )
    
    if success:
        print("\n🎉 Dataset upload completed successfully!")
    else:
        print("\n❌ Dataset upload failed. Please check the errors above.")

if __name__ == "__main__":
    main() 
