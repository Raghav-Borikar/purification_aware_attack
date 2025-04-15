import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import get_class_names

class PurificationAwareAttack:
    """
    Purification-Aware Attack (PAA) that targets CLIPure-Cos defense mechanism
    by simulating the purification process during attack optimization.
    """
    def __init__(self, model, target_labels=None, num_classes=1000, 
                 epsilon=8/255, alpha=2/255, steps=10, 
                 purification_steps=10, purification_step_size=30.0,
                 purification_gamma=0.9, random_start=True, 
                 norm="L2", device="cuda"):
        """
        Initialize the Purification-Aware Attack.
        
        Args:
            model: The model to attack (must have a purify_zi method)
            target_labels: Target labels for targeted attack (None for untargeted)
            num_classes: Number of classes in the dataset
            epsilon: Maximum perturbation
            alpha: Step size for attack optimization
            steps: Number of attack steps
            purification_steps: Number of purification steps to simulate
            purification_step_size: Step size for purification simulation
            purification_gamma: Momentum factor for purification simulation
            random_start: Whether to start with random perturbation
            norm: Attack norm ("L2" or "Linf")
            device: Device to use ("cuda" or "cpu")
        """
        self.model = model
        self.target_labels = target_labels
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.purification_steps = purification_steps
        self.purification_step_size = purification_step_size
        self.purification_gamma = purification_gamma
        self.random_start = random_start
        self.norm = norm
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        
    def simulate_purification(self, img_emb, template_emb):
        """
        Simulate the CLIPure-Cos purification process during attack.
        This directly replicates the purification behavior to create robust adversarial examples.
        
        Args:
            img_emb: Image embedding tensor
            template_emb: Template embedding tensor
            
        Returns:
            Purified embedding
        """
        batch_size = img_emb.shape[0]
        momentum = torch.zeros_like(img_emb).to(self.device)
        
        # Initialize purified embedding with a copy that requires gradients
        purified_emb = img_emb
        purified_emb.requires_grad_(True)
        
        # Simulate purification steps
        for _ in range(self.purification_steps):
            # Normalize to unit vectors (magnitude and direction)
            r = torch.norm(purified_emb, dim=1, keepdim=True)
            u = purified_emb / r
            
            # Compute cosine similarity loss with blank template
            logits_uncond = F.cosine_similarity(u, template_emb, dim=1)
            loss = -logits_uncond.mean()  # Negative because we want to maximize similarity
            
            # Compute gradients
            grad = torch.autograd.grad(loss, purified_emb, retain_graph=True)[0]
            
            # Update with momentum (replicating CLIPure-Cos exactly)
            grad_u = r * grad
            
            if self.norm == "Linf":
                momentum = self.purification_gamma * momentum - (1 - self.purification_gamma) * grad_u / (torch.norm(grad_u, p=1, dim=1, keepdim=True) + 1e-10)
                u = u + self.purification_step_size * momentum.sign()
            elif self.norm == "L2":
                momentum = self.purification_gamma * momentum - (1 - self.purification_gamma) * grad_u / (torch.norm(grad_u, p=2, dim=1, keepdim=True) + 1e-10)
                u = u + self.purification_step_size * momentum
            
            # Normalize again
            u = u / (torch.norm(u, dim=1, keepdim=True) + 1e-10)
            
            # Update purified embedding
            purified_emb = r * u
            purified_emb = purified_emb.requires_grad_(True)
            
        return purified_emb
    
    def _compute_attack_loss(self, logits, labels):
        """
        Compute the loss for attack optimization.
        
        Args:
            logits: Model output logits
            labels: True labels or target labels
            
        Returns:
            Loss value
        """
        num_classes = logits.shape[1]
        # Check if labels are in valid range
        if torch.any(labels < 0) or torch.any(labels >= num_classes):
            print(f"WARNING: Labels out of range [0,{num_classes-1}]")
            print(f"Label range: {labels.min().item()} to {labels.max().item()}")
            # Clamp labels to valid range to prevent crash
            labels = torch.clamp(labels, 0, num_classes-1)
        
        if self.target_labels is not None:
            # Targeted attack: minimize loss for target class
            return -self.loss_fn(logits, self.target_labels)
        else:
            # Untargeted attack: maximize loss for true class
            return self.loss_fn(logits, labels)
    
    def attack(self, images, labels):
        """
        Perform the Purification-Aware Attack.
        
        Args:
            images: Input images
            labels: True labels
            
        Returns:
            Adversarial examples
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        if self.target_labels is not None:
            self.target_labels = self.target_labels.to(self.device)
            
        # Initialize perturbation
        
#        delta.requires_grad_(True)
        
        # Get class names from dataset
        class_names = get_class_names(self.model.dataset)
        
#        Get blank template embedding for purification simulation
        with torch.no_grad():
            class_embeddings = self.model.get_class_embeddings(class_names)
            template_emb = self.model.get_blank_template_embedding()
            batch_size = images.shape[0]
            template_emb = template_emb.repeat(batch_size, 1).to(self.device)

        delta = torch.zeros_like(images).to(self.device)
        if self.random_start:
            # Random initialization within epsilon ball
            if self.norm == "Linf":
                delta = (torch.rand_like(images) * 2 - 1) * self.epsilon
            elif self.norm == "L2":
                delta = torch.randn_like(images)
                delta_norms = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
                factor = torch.min(torch.ones_like(delta_norms) * self.epsilon / delta_norms, torch.ones_like(delta_norms))
                delta = delta * factor.view(-1, 1, 1, 1)

            delta = torch.clamp(images + delta, 0, 1) - images
            
        delta.requires_grad = True  
          
        # Attack optimization loop
        
        for step in range(self.steps):
            # Zero gradients
            #delta_step = delta.clone()
            #delta_step.requires_grad_(True)
            
#            if delta.grad is not None:
#                delta.grad.zero_()
                
            # Forward pass to get image embeddings
#            with torch.enable_grad():
            adv_images = (images + delta).requires_grad_(True)
            
            try:
                img_emb = self.model.embed_image(adv_images, purify=True)
                
                # Simulate purification process
                purified_emb = self.simulate_purification(img_emb, template_emb)
                
                # Compute logits using purified embeddings
                logits = self.model.compute_logits(purified_emb, class_embeddings)
                
                # Compute attack loss
                loss = self._compute_attack_loss(logits, labels)

                # Debug prints for gradient tracking
                #print(f"Step {step}: delta.requires_grad = {delta.requires_grad}")
                #print(f"Loss value: {loss.item()}")

                loss.backward()
                
                #print("delta.grad:", delta.grad)

                if delta.grad is None:
                    print("ERROR: No gradients computed despite delta requiring gradients")
                    raise RuntimeError("Gradients not computed for delta. Check computational graph.")

#                Update delta with projected gradient descent
                with torch.no_grad():
                    grad = delta.grad           
                
                    if self.norm == "Linf":
                        delta = delta + self.alpha * grad.sign()
                        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                    elif self.norm == "L2":
                        grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
                        scaled_grad = grad / (grad_norm + 1e-10)
                        delta = delta + self.alpha * scaled_grad
                        
                        delta_norm = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
                        factor = torch.min(self.epsilon / delta_norm, torch.ones_like(delta_norm))
                        delta = delta * factor
            
                    delta = torch.clamp(images + delta, 0, 1) - images

                # Reset gradients for next step
                delta = delta.detach()
                delta.requires_grad = True
            
            except Exception as e:
                print(print(f"Error in attack step {step}: {e}"))
                # Continue with next step if there's an error
                continue
            
        return torch.clamp(images+delta, 0, 1)
    
    def __call__(self, images, labels):
        """
        Callable interface for the attack.
        
        Args:
            images: Input images
            labels: True labels
            
        Returns:
            Adversarial examples
        """
        return self.attack(images, labels)