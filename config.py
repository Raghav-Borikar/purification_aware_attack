import torch
import os

class Config:
    """
    Configuration for Purification-Aware Attack (PAA) against CLIPure-Cos
    """
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    
    # CLIP model configuration
    clip_model_name = "ViT-L-14"  # Options: ViT-B-32, ViT-B-16, ViT-L-14, etc.
    pretrained = "openai"
    logit_scale = True
    
    # Dataset configuration
    dataset = "cifar10"  # Options: cifar10, cifar100, imagenet
    data_dir = {
        "cifar10": "./data/cifar10",
        "cifar100": "./data/cifar100",
        "imagenet": "./data/imagenet"
    }
    n_examples = {
        "cifar10": 1000,
        "cifar100": 1000,
        "imagenet": 1000
    }
    batch_size = 16
    
    # CLIPure-Cos purification configuration
    purification_steps = 3
    purification_step_size = 10.0
    purification_gamma = 0.9
    purification_norm = "L2"  # Options: L2, Linf
    
    # Attack configuration
    epsilon = 8/255  # Maximum perturbation
    alpha = 2/255    # Step size
    attack_steps = 10
    attack_norm = "L2"  # Options: L2, Linf
    random_start = True
    
    # Output configuration
    output_dir = "./results"

    # Add to the Config class in config.py
    template_file = "templates.json"

    # Templates for zero-shot classification
    # These can be replaced with your custom templates
    templates = [
        "a photo of a {}.",
        "a rendering of a {}.",
        "a cropped photo of the {}.",
        "the photo of a {}.",
        "a photo of a clean {}.",
        "a photo of a dirty {}.",
        "a dark photo of the {}.",
        "a photo of my {}.",
        "a photo of the cool {}.",
        "a close-up photo of a {}.",
        "a bright photo of the {}.",
        "a cropped photo of a {}.",
        "a photo of the {}.",
        "a good photo of the {}.",
        "a photo of one {}.",
        "a close-up photo of the {}.",
        "a rendition of the {}.",
        "a photo of the clean {}.",
        "a rendition of a {}.",
        "a photo of a nice {}.",
        "a good photo of a {}.",
        "a photo of the nice {}.",
        "a photo of the small {}.",
        "a photo of the weird {}.",
        "a photo of the large {}.",
        "a photo of a cool {}.",
        "a photo of a small {}."
    ]

# Create output directory if it doesn't exist
if not os.path.exists(Config.output_dir):
    os.makedirs(Config.output_dir)

# Export config for easy access
config = vars(Config)
