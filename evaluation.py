import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import json
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import time

from clipure_model import CLIPureModel
from paa_attack import PurificationAwareAttack
from utils import AverageMeter, load_image_dataset, get_class_names, save_images_with_predictions

def evaluate_accuracy(model, dataloader, classes, device):
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        classes: List of class names
        device: Device to use
        
    Returns:
        Accuracy score
    """
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Evaluating clean accuracy"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass with gradient computation enabled
        with torch.enable_grad():  # Important: enable gradients during evaluation
            predictions, _ = model.classify(images, classes, purify=False)
        
        # Update statistics
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    
    return accuracy

def evaluate_robustness(model, attack, dataloader, classes, device, save_dir=None):
    """
    Evaluate model robustness against an attack.
    
    Args:
        model: Model to evaluate
        attack: Attack function or object
        dataloader: DataLoader for the dataset
        classes: List of class names
        device: Device to use
        save_dir: Directory to save adversarial examples
        
    Returns:
        Robust accuracy score
    """
    model.eval()
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating robustness")):
        images = images.to(device)
        labels = labels.to(device)
        
        # Generate adversarial examples
        adv_images = attack(images, labels)
        
        # Forward pass with CLIPure
        #with torch.no_grad():
        predictions, _ = model.classify(adv_images, classes)
        
        # Update statistics
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Save some adversarial examples
        if save_dir is not None and i < 5:
            save_images_with_predictions(
                original_images=images.cpu(),
                adversarial_images=adv_images.cpu(),
                true_labels=labels.cpu(),
                predicted_labels=predictions.cpu(),
                class_names=classes,
                save_path=os.path.join(save_dir, f"adv_examples_batch_{i}.png")
            )
    
    # Calculate robust accuracy
    robust_accuracy = 100 * correct / total
    
    return robust_accuracy

def run_evaluation(config):
    """
    Run evaluation of PAA attack against CLIPure-Cos.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Results dictionary
    """
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset_name = config['dataset']
    print(f"Loading {dataset_name} dataset...")
    
    # Get class names
    class_names = get_class_names(dataset_name)
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    # Load dataset
    x_test, y_test = load_image_dataset(
        dataset_name=dataset_name,
        data_dir=config['data_dir'][dataset_name],
        n_examples=config['n_examples'][dataset_name]
    )
    test_dataset = TensorDataset(x_test, y_test)
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize CLIPure-Cos model
    print("Initializing CLIPure-Cos model...")
    clipure_model = CLIPureModel(
        clip_model_name=config['clip_model_name'],
        pretrained=config['pretrained'],
        dataset=config['dataset'],
        template_file=config['template_file'],
        device=device,
        purification_steps=config['purification_steps'],
        purification_step_size=config['purification_step_size'],
        purification_gamma=config['purification_gamma'],
        purification_norm=config['purification_norm'],
        logit_scale=config['logit_scale']
    ).to(device)
    
    # Initialize Purification-Aware Attack
    print("Initializing Purification-Aware Attack...")
    paa = PurificationAwareAttack(
        model=clipure_model,
        target_labels=None,  # Untargeted attack
        num_classes=len(class_names),
        epsilon=config['epsilon'],
        alpha=config['alpha'],
        steps=config['attack_steps'],
        purification_steps=config['purification_steps'],
        purification_step_size=config['purification_step_size'],
        purification_gamma=config['purification_gamma'],
        random_start=config['random_start'],
        norm=config['attack_norm'],
        device=device,
    )
    
    # Create directory for saving results
    save_dir = os.path.join(config['output_dir'], f"{dataset_name}_{config['clip_model_name']}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump({k: str(v) for k, v in config.items()}, f, indent=2)
    
    # Evaluate clean accuracy
    #print("Evaluating clean accuracy...")
    #clean_accuracy = evaluate_accuracy(clipure_model, test_loader, class_names, device)
    #print(f"Clean accuracy: {clean_accuracy:.2f}%")
    
    # Evaluate robust accuracy
    print("Evaluating robust accuracy...")
    robust_accuracy = evaluate_robustness(clipure_model, paa, test_loader, class_names, device, save_dir)
    print(f"Robust accuracy: {robust_accuracy:.2f}%")
    
    # Compute attack success rate
    attack_success_rate = 100 - robust_accuracy
    print(f"Attack success rate: {attack_success_rate:.2f}%")
    
    # Save results
    results = {
        "dataset": dataset_name,
        "clip_model_name": config['clip_model_name'],
        "epsilon": config['epsilon'],
        "attack_steps": config['attack_steps'],
        "purification_steps": config['purification_steps'],
        "clean_accuracy": clean_accuracy,
        "robust_accuracy": robust_accuracy,
        "attack_success_rate": attack_success_rate
    }
    
    with open(os.path.join(save_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {save_dir}")
    
    return results

if __name__ == "__main__":
    from config import config
    run_evaluation(config)
