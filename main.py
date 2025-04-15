import argparse
import torch
import os
import json
from paa_attack import PurificationAwareAttack
from clipure_model import CLIPureModel
from evaluation import run_evaluation
from utils import set_seed
from config import config
import torch.multiprocessing as mp

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Purification-Aware Attack (PAA) against CLIPure-Cos")
    
    # Model parameters
    parser.add_argument("--clip_model_name", type=str, default=config["clip_model_name"],
                        help="CLIP model name (e.g., ViT-B-32, ViT-L-14)")
    parser.add_argument("--pretrained", type=str, default=config["pretrained"],
                        help="Pretrained model source")
    parser.add_argument("--template_file", type=str, help = "a json file of templates")
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default=config["dataset"],
                        choices=["cifar10", "cifar100", "imagenet"],
                        help="Dataset to use")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset (overrides config)")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Number of examples to use (overrides config)")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"],
                        help="Batch size")
    
    # CLIPure parameters
    parser.add_argument("--purification_steps", type=int, default=config["purification_steps"],
                        help="Number of purification steps")
    parser.add_argument("--purification_step_size", type=float, default=config["purification_step_size"],
                        help="Step size for purification")
    parser.add_argument("--purification_gamma", type=float, default=config["purification_gamma"],
                        help="Momentum factor for purification")
    parser.add_argument("--purification_norm", type=str, default=config["purification_norm"],
                        choices=["L2", "Linf"], help="Norm for purification updates")
    
    # Attack parameters
    parser.add_argument("--epsilon", type=float, default=config["epsilon"],
                        help="Maximum perturbation (e.g., 8/255, 16/255)")
    parser.add_argument("--alpha", type=float, default=config["alpha"],
                        help="Step size for attack")
    parser.add_argument("--attack_steps", type=int, default=config["attack_steps"],
                        help="Number of attack steps")
    parser.add_argument("--attack_norm", type=str, default=config["attack_norm"],
                        choices=["L2", "Linf"], help="Norm for attack")
    parser.add_argument("--random_start", type=bool, default=config["random_start"],
                        help="Whether to use random initialization for attack")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default=config["output_dir"],
                        help="Directory to save results")
    
    # Experiment parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    
    return parser.parse_args()

def update_config(args):
    """
    Update configuration with command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Updated configuration
    """
    updated_config = config.copy()
    
    # Update configuration with command-line arguments
    for key, value in vars(args).items():
        if value is not None:
            if key == "data_dir":
                updated_config["data_dir"][args.dataset] = value
            elif key == "n_examples":
                updated_config["n_examples"][args.dataset] = value
            else:
                updated_config[key] = value
    
    return updated_config

def main():
    """
    Main function for running PAA attack against CLIPure-Cos.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Update configuration
    run_config = update_config(args)
    
    # Print configuration
    print("Configuration:")
    print("-" * 50)
    for key, value in run_config.items():
        print(f"{key}: {value}")
    print("-" * 50)
    
    # Run evaluation
    results = run_evaluation(run_config)
    
    # Print results
    print("\nFinal Results:")
    print("-" * 50)
    print(f"Dataset: {results['dataset']}")
    print(f"CLIP Model: {results['clip_model_name']}")
    print(f"Attack Epsilon: {results['epsilon']}")
    print(f"Clean Accuracy: {results['clean_accuracy']:.2f}%")
    print(f"Robust Accuracy: {results['robust_accuracy']:.2f}%")
    print(f"Attack Success Rate: {results['attack_success_rate']:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    # Set multiprocessing start method to spawn to avoid issues on some systems
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    main()
