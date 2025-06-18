#!/usr/bin/env python3
"""
Simple training script to get started with GRPO training.
"""

import argparse
import subprocess
import sys
import os


def create_sample_data():
    """Create sample dataset if it doesn't exist."""
    if not os.path.exists("sample_triton_dataset.json"):
        print("Creating sample dataset...")
        import utils
        utils.create_sample_dataset("sample_triton_dataset.json", 50)
        print("Sample dataset created!")
    else:
        print("Sample dataset already exists.")


def run_training(args):
    """Run the GRPO training."""
    cmd = [
        sys.executable, "main.py",
        "--model_name", args.model_name,
        "--dataset_path", args.dataset_path,
        "--output_dir", args.output_dir,
        "--num_train_iters", str(args.num_train_iters),
        "--learning_rate", str(args.learning_rate),
        "--num_chains", str(args.num_chains),
        "--temperature", str(args.temperature),
        "--save_steps", str(args.save_steps),
        "--eval_iterations", str(args.eval_iterations),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--max_grad_norm", str(args.max_grad_norm),
        "--kl_weight_beta", str(args.kl_weight_beta)
    ]
    
    if args.update_ref_model:
        cmd.append("--update_ref_model")
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Train Triton kernel model with GRPO")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="cdreetz/kwen2.5-1.5b", 
                       help="Your fine-tuned model")
    parser.add_argument("--dataset_path", type=str, default="sample_triton_dataset.json",
                       help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="triton_grpo_results",
                       help="Output directory for checkpoints")
    
    # Training settings
    parser.add_argument("--num_train_iters", type=int, default=500,
                       help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--num_chains", type=int, default=4,
                       help="Number of generations per prompt")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    
    # Checkpoint and evaluation
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_iterations", type=int, default=50,
                       help="Evaluate every N steps")
    
    # Optimization
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Gradient clipping norm")
    parser.add_argument("--kl_weight_beta", type=float, default=0.01,
                       help="KL divergence weight")
    
    # Flags
    parser.add_argument("--update_ref_model", action="store_true",
                       help="Update reference model during training")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--create_data", action="store_true",
                       help="Create sample dataset")
    
    args = parser.parse_args()
    
    # Create sample data if requested or if no dataset exists
    if args.create_data or not os.path.exists(args.dataset_path):
        create_sample_data()
    
    # Run training
    run_training(args)


if __name__ == "__main__":
    main()