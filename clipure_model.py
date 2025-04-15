import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.functional import cosine_similarity
import open_clip
from torchvision import transforms

class CLIPureModel(nn.Module):
    """
    CLIPure-Cos model for adversarial purification in CLIP's latent space
    by maximizing cosine similarity with a blank template.
    """
    def __init__(self, clip_model_name='ViT-B-32', pretrained='openai', dataset=None, 
                 template_file=None, device='cuda', purification_steps=10, 
                 purification_step_size=30.0, purification_gamma=0.9, 
                 purification_norm="L2", logit_scale=True):
        """
        Initialize the CLIPure-Cos model.
        
        Args:
            clip_model_name: CLIP model name (e.g., 'ViT-B-32', 'ViT-L-14')
            pretrained: Pretrained model source
            templates: List of templates for zero-shot classification
            device: Device to use
            purification_steps: Number of purification steps
            purification_step_size: Step size for purification
            purification_gamma: Momentum factor for purification
            purification_norm: Norm for purification updates ("L2" or "Linf")
            logit_scale: Whether to scale logits with CLIP's logit scale
        """
        super().__init__()
        self.device = device
        self.clip_model_name = clip_model_name
        self.purification_steps = purification_steps
        self.purification_step_size = purification_step_size
        self.purification_gamma = purification_gamma
        self.purification_norm = purification_norm
        self.logit_scale = logit_scale
        
        # Load CLIP model
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        
        # Move model to device and set to eval mode
        self.clip_model = self.clip_model.to(device).eval()
        self.dataset = dataset
        # Initialize templates
        self.blank_template = "a photo of a ."
        if template_file:
            from utils import load_templates
            self.templates = load_templates(template_file, dataset)
            print(f"Loaded {len(self.templates)} templates for {dataset} dataset")

        else:
            # Default templates from CLIP
            self.templates = [
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
        
        # Initialize blank template embeddings
        self._init_template_embeddings()
        
    def _init_template_embeddings(self):
        """
        Initialize null template embeddings for purification.
        """
        # Get blank template embedding
        blank_text_tokens = self.tokenizer([self.blank_template]).to(self.device)
        with torch.no_grad():
            blank_text_features = self.clip_model.encode_text(blank_text_tokens)
            blank_text_features = F.normalize(blank_text_features, dim=-1)
        self.blank_template_embedding = blank_text_features
        
        # Get category template embeddings
        self.text_embeddings = {}
    
    def get_blank_template_embedding(self):
        """
        Get the embedding of the blank template.
        
        Returns:
            Blank template embedding
        """
        return self.blank_template_embedding
    
    def embed_image(self, images):
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False)

        images = images.to(self.device)

        # Enable gradients for image encoder
        was_training = self.clip_model.training
        self.clip_model.train()  # 👈 TEMPORARILY enable train mode to allow gradients

        # Forward pass with gradient computation
        with torch.set_grad_enabled(True):
            image_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

        if not was_training:
            self.clip_model.eval()  # 👈 Restore original mode

            # Ensure gradient tracking is enabled
        if not image_features.requires_grad:
            image_features = image_features.clone().requires_grad_(True)
    
        return image_features

    
    def embed_text(self, texts):
        """
        Encode texts into CLIP's embedding space.
        
        Args:
            texts: Input texts or text tokens
            
        Returns:
            Text embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize if not already tokenized
        if not torch.is_tensor(texts):
            text_tokens = self.tokenizer(texts).to(self.device)
        else:
            text_tokens = texts
        
        # Encode texts
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def get_class_embeddings(self, classes):
        """
        Get embeddings for class names using templates.
        
        Args:
            classes: List of class names
            
        Returns:
            Class embeddings
        """
        if not isinstance(classes, list):
            classes = [classes]
            
        # Check if already computed
        missing_classes = [cls for cls in classes if cls not in self.text_embeddings]
        
        if missing_classes:
            # Generate text tokens for each class with each template
            text_embeddings = []
            for cls in missing_classes:
                cls_text_embeddings = []
                for template in self.templates:
                    text = template.format(c=cls)
                    embedding = self.embed_text([text])[0]
                    cls_text_embeddings.append(embedding)
                
                # Average embeddings across templates
                cls_embedding = torch.stack(cls_text_embeddings).mean(dim=0)
                cls_embedding = F.normalize(cls_embedding, dim=-1)
                self.text_embeddings[cls] = cls_embedding
        
        # Return embeddings for requested classes
        return torch.stack([self.text_embeddings[cls] for cls in classes], dim=1)
    
    def purify_zi(self, img_emb):
        """
        Purification in latent space via CLIPure-Cos method.
        
        Args:
            img_emb: Image embedding tensor
            
        Returns:
            Purified embedding
        """
        batch_size = img_emb.shape[0]
        
        # Ensure img_emb requires gradient
        img_emb = img_emb.clone().detach().requires_grad_(True)
        
        # Initialize blank template embedding
        text_embed = self.blank_template_embedding.repeat(batch_size, 1).to(self.device)
        text_embed.requires_grad_(True)
        # Initialize momentum
        momentum = torch.zeros_like(img_emb)
        
        # Iterative purification
        for i in range(self.purification_steps):
            # Normalize to unit vectors (magnitude and direction)
            r = torch.norm(img_emb, dim=1, keepdim=True)
            u = img_emb / r
            
            # Compute cosine similarity with blank template
            logits_uncond = F.cosine_similarity(u, text_embed, dim=1)
            loss = -logits_uncond.mean()  # Negative because we want to maximize similarity

            # 🧠 Add debug prints here:
            #print("u.requires_grad:", u.requires_grad)
            #print("text_embed.requires_grad:", text_embed.requires_grad)
            #print("logits_uncond.requires_grad:", logits_uncond.requires_grad)
            #print("loss.requires_grad:", loss.requires_grad)

            # Verify gradient tracking is enabled
            if not img_emb.requires_grad:
                raise RuntimeError("img_emb does not require grad — cannot purify")

            grad = torch.autograd.grad(loss, img_emb, create_graph=True)[0]  # create_graph if you need higher-order grads
            
            # Update with momentum
            grad_u = r * grad
            
            if self.purification_norm == "Linf":
                momentum = self.purification_gamma * momentum - (1 - self.purification_gamma) * grad_u / torch.norm(grad_u, p=1, dim=1, keepdim=True)
                u = u + self.purification_step_size * momentum.sign()
            elif self.purification_norm == "L2":
                momentum = self.purification_gamma * momentum - (1 - self.purification_gamma) * grad_u / torch.norm(grad_u, p=2, dim=1, keepdim=True)
                u = u + self.purification_step_size * momentum
            
            # Normalize again
            u = u / torch.norm(u, dim=1, keepdim=True)
            
            # Update embedding
            if i < self.purification_steps - 1:
                img_emb = (r * u).clone().detach().requires_grad_(True)
            else:
                img_emb = r * u
        
        return img_emb
    
    def compute_logits(self, img_emb, class_embeddings=None):
        """
        Compute logits using image embeddings and class embeddings.
        
        Args:
            img_emb: Image embedding tensor
            class_embeddings: Class embeddings (if None, use all classes)
            
        Returns:
            Logits
        """
        if class_embeddings is None:
            # Use all classes if not specified
            raise ValueError("Class embeddings must be provided")

        if class_embeddings.shape[0] != img_emb.shape[-1]:
            class_embeddings = class_embeddings.transpose(0, 1)
        
        # Compute cosine similarity with class embeddings
        logits = img_emb @ class_embeddings
        
        # Apply logit scaling if enabled
        if self.logit_scale:
            logits = logits * self.clip_model.logit_scale.exp()
        
        return logits
    
    def forward(self, images, classes=None, purify=True):
        """
        Forward pass with purification.
        
        Args:
            images: Input images
            classes: List of class names
            
        Returns:
            Logits
        """
        # Encode images
        img_emb = self.embed_image(images)
        
        # Purify embeddings only if flag is True
        if purify:
            img_emb = self.purify_zi(img_emb)
        
        # Get class embeddings
        if classes is not None:
            class_embeddings = self.get_class_embeddings(classes)
        else:
            raise ValueError("Class names must be provided")
        
        # Compute logits
        logits = self.compute_logits(img_emb, class_embeddings)
        
        return logits
    
    def classify(self, images, classes, purify=True):
        """
        Classify images using CLIPure-Cos.
        
        Args:
            images: Input images
            classes: List of class names
            
        Returns:
            Predicted class indices and probabilities
        """
        # Forward pass with purification
        logits = self.forward(images, classes, purify)
        
        # Get predicted class indices and probabilities
        probs = F.softmax(logits, dim=1)
        predicted_indices = torch.argmax(probs, dim=1)
        
        return predicted_indices, probs
