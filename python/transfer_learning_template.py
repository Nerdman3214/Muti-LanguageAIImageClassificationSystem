#!/usr/bin/env python3
"""
Transfer Learning Template
==========================

This template shows how to use your pre-trained models as a starting point
for training on new data.

STEPS FOR TRANSFER LEARNING:
1. Load pre-trained model
2. Freeze early layers (keep learned features)
3. Replace/modify final layers for your task
4. Train on your new data
5. Fine-tune (optionally unfreeze some layers)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

# ============================================================================
# OPTION 1: Transfer Learning from ImageNet (ResNet50)
# ============================================================================

def create_classifier_from_resnet50(num_classes):
    """
    Create a classifier using ResNet50 pre-trained on ImageNet.
    
    WHY RESNET50?
    - Trained on 1.2M images, 1000 classes
    - Learned general visual features (edges, textures, objects)
    - These features transfer well to many vision tasks
    
    Args:
        num_classes: Number of classes in YOUR dataset
    
    Returns:
        Model ready for transfer learning
    """
    # Load pre-trained ResNet50
    # weights='IMAGENET1K_V2' uses the best available weights
    model = models.resnet50(weights='IMAGENET1K_V2')
    
    # FREEZE early layers
    # These layers learned general features that work for most images
    # We don't want to change them (saves training time, prevents overfitting)
    for param in model.parameters():
        param.requires_grad = False  # Don't compute gradients for these
    
    # REPLACE the final layer
    # Original: nn.Linear(2048, 1000) for ImageNet's 1000 classes
    # New: nn.Linear(2048, num_classes) for YOUR classes
    num_features = model.fc.in_features  # Get input size (2048 for ResNet50)
    
    model.fc = nn.Sequential(
        nn.Dropout(0.5),              # Prevent overfitting
        nn.Linear(num_features, 256), # Hidden layer
        nn.ReLU(),                    # Activation
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)   # Output layer (YOUR classes)
    )
    
    return model


def fine_tune_model(model, unfreeze_from='layer4'):
    """
    Unfreeze later layers for fine-tuning.
    
    Call this AFTER initial training to allow the model to adapt
    its learned features to your specific data.
    
    Args:
        model: The model to fine-tune
        unfreeze_from: Which layer to start unfreezing from
    """
    # Unfreeze layers starting from specified layer
    unfreeze = False
    for name, child in model.named_children():
        if name == unfreeze_from:
            unfreeze = True
        if unfreeze:
            for param in child.parameters():
                param.requires_grad = True
    
    return model


# ============================================================================
# OPTION 2: Transfer Learning for RL (Policy Network)
# ============================================================================

class TransferablePolicy(nn.Module):
    """
    A policy network that can be used for transfer learning between
    similar RL environments.
    
    TRANSFER LEARNING IN RL:
    - Pre-train on simple environment (e.g., CartPole)
    - Transfer to harder environment (e.g., Acrobot, MountainCar)
    - The network has learned general control concepts
    """
    
    def __init__(self, state_dim, action_dim, pretrained_path=None):
        super().__init__()
        
        # Shared feature extractor (transfer these layers)
        self.features = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # Task-specific head (replace for new task)
        self.action_head = nn.Linear(64, action_dim)
        
        # Load pre-trained weights if provided
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def forward(self, x):
        features = self.features(x)
        return self.action_head(features)
    
    def load_pretrained(self, path):
        """Load pre-trained weights for the feature extractor only."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Only load feature extractor weights
        feature_state = {k: v for k, v in checkpoint.items() 
                        if k.startswith('features.')}
        self.load_state_dict(feature_state, strict=False)
        
        # Freeze feature extractor
        for param in self.features.parameters():
            param.requires_grad = False
        
        print(f"Loaded pre-trained features from {path}")


# ============================================================================
# TRAINING EXAMPLE
# ============================================================================

def train_transfer_model(model, train_loader, val_loader, epochs=10):
    """
    Training loop for transfer learning.
    
    KEY DIFFERENCES FROM TRAINING FROM SCRATCH:
    1. Lower learning rate (don't disrupt pre-trained weights)
    2. Fewer epochs needed (starting from good initialization)
    3. Different LR for frozen vs unfrozen layers
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use lower learning rate for transfer learning
    # Pre-trained weights are already good, don't change them too much
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': 1e-3},  # New layers: higher LR
    ])
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Val Accuracy: {accuracy:.2f}%")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example: Create a 10-class classifier from ResNet50
    num_my_classes = 10  # e.g., 10 types of food
    
    model = create_classifier_from_resnet50(num_my_classes)
    print(f"Created model with {num_my_classes} output classes")
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
