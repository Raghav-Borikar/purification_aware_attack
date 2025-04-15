import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from PIL import Image
import random
from torch.utils.data import TensorDataset
from robustbench.data import load_clean_dataset
import json

# CLIP-specific normalization parameters
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def get_clip_transform(image_size=224):
    """
    Return CLIP-compatible image preprocessing transforms.
    
    Args:
        image_size: Size to resize images to
        
    Returns:
        Transforms
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

def get_class_names(dataset_name):
    """
    Get class names for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        List of class names
    """
    if dataset_name.lower() == 'cifar10':
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    elif dataset_name.lower() == 'cifar100':
        # Get CIFAR-100 class names
        dataset = datasets.CIFAR100(root='./data', train=False, download=True)
        return dataset.classes
    elif dataset_name.lower() == 'imagenet':
        # Load from file or use a predefined mapping
        try:
            import json
            with open('imagenet-index.json') as f:
                class_idx = json.load(f)
                return [class_idx[str(i)][1] for i in range(1000)]
        except:
            # Return a shortened version for simplicity
            return [f"class_{i}" for i in range(1000)]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_image_dataset(dataset_name, data_dir, n_examples=None, preprocess=None):
    """
    Load dataset with CLIP preprocessing using torchvision directly.
    """
    if preprocess is None:
        preprocess = get_clip_transform()
    
    if dataset_name.lower() == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=preprocess)
    elif dataset_name.lower() == 'cifar100':
        dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=preprocess)
    elif dataset_name in ['imagenet', 'imagenet1k']:
        # Make sure ImageNet is properly structured in `data_dir/val/class_x/xxx.jpeg`
        dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=preprocess)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Limit number of examples if specified
    if n_examples is not None and n_examples < len(dataset):
        indices = range(n_examples)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Convert to tensor format expected by the rest of your code
    x_test = torch.stack([img for img, _ in dataset])
    y_test = torch.tensor([label for _, label in dataset])
    
    return x_test, y_test

def denormalize(tensor, mean=CLIP_MEAN, std=CLIP_STD):
    """
    Reverse CLIP normalization for visualization.
    
    Args:
        tensor: Normalized tensor
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized tensor
    """
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)
    
    # Create inverse normalize transform
    inverse_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    
    return inverse_normalize(tensor)

def save_images_with_predictions(original_images, adversarial_images, true_labels, 
                                predicted_labels, class_names, save_path):
    """
    Save original and adversarial images with predictions.
    
    Args:
        original_images: Original images
        adversarial_images: Adversarial examples
        true_labels: True labels
        predicted_labels: Predicted labels
        class_names: List of class names
        save_path: Path to save the image
    """
    # Denormalize images
    original_images = denormalize(original_images)
    adversarial_images = denormalize(adversarial_images)
    
    # Number of images
    n = min(len(original_images), 8)
    
    # Create figure
    fig, axes = plt.subplots(2, n, figsize=(n*2, 4))
    
    for i in range(n):
        # Original image
        orig_img = original_images[i].permute(1, 2, 0).cpu().numpy()
        orig_img = np.clip(orig_img, 0, 1)
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"True: {class_names[true_labels[i].detach().cpu().item()]}")
        axes[0, i].axis('off')
        
        # Adversarial image
        adv_img = adversarial_images[i].permute(1, 2, 0).detach().numpy()
        adv_img = np.clip(adv_img, 0, 1)
        axes[1, i].imshow(adv_img)
        axes[1, i].set_title(f"Pred: {class_names[predicted_labels[i].detach().cpu().item()]}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_templates(json_path, dataset_name):
    """
    Load dataset-specific templates from a JSON file.
    
    Args:
        json_path: Path to the templates JSON file
        dataset_name: Name of the dataset (e.g., 'cifar10', 'imagenet')
        
    Returns:
        List of templates for the specified dataset
    """
    try:
        with open(json_path, 'r') as f:
            templates_dict = json.load(f)
            
        # Map dataset names to keys in the JSON file
        dataset_mapping = {
            'cifar10': 'cifar10',
            'cifar100': 'cifar100',
            'imagenet': 'imagenet1k',
            # Add mappings for other datasets
        }
        
        # Get the appropriate key for the dataset
        json_key = dataset_mapping.get(dataset_name.lower(), dataset_name.lower())
        
        if json_key in templates_dict:
            return templates_dict[json_key]
        else:
            print(f"Warning: Templates for {dataset_name} not found in {json_path}. Using default templates.")
            return ["a photo of a {}."]
            
    except Exception as e:
        print(f"Error loading templates: {e}")
        return ["a photo of a {}."]

