#!/usr/bin/env python3
"""
============================================================================
save_model_for_transfer_learning.py - Model Export & Transfer Learning Guide
============================================================================

This script helps you:
1. Save your trained models in various formats
2. Set up for transfer learning
3. Optimize models for faster inference

WHAT IS TRANSFER LEARNING?
--------------------------
Transfer learning is reusing a pre-trained model on a new, related task.
Instead of training from scratch (which requires lots of data and time),
you start with a model that already learned useful features.

Example:
- ResNet50 trained on ImageNet learned to recognize edges, textures, shapes
- You can reuse these features to classify medical images, food, etc.
- Only need to train the final layers on your specific data

WHAT IS MODEL OPTIMIZATION?
---------------------------
Making your model faster/smaller for production deployment:
- Quantization: Use INT8 instead of FP32 (4x smaller, faster on CPU)
- Pruning: Remove unimportant weights
- Knowledge Distillation: Train a smaller model to mimic the larger one

Usage:
    python scripts/save_model_for_transfer_learning.py --model cartpole --format all
    python scripts/save_model_for_transfer_learning.py --model resnet50 --quantize
"""

import os
import sys
import argparse
import shutil
from datetime import datetime

# Check if required packages are available
def check_dependencies():
    """
    Check if required packages are installed.
    Returns a dict of package: available boolean
    """
    deps = {}
    
    try:
        import torch
        deps['torch'] = True
    except ImportError:
        deps['torch'] = False
        
    try:
        import onnx
        deps['onnx'] = True
    except ImportError:
        deps['onnx'] = False
        
    try:
        import onnxruntime
        deps['onnxruntime'] = True
    except ImportError:
        deps['onnxruntime'] = False
        
    try:
        import numpy
        deps['numpy'] = True
    except ImportError:
        deps['numpy'] = False
    
    return deps


def save_pytorch_model(model_path, output_dir):
    """
    Save a PyTorch model in multiple formats for transfer learning.
    
    PYTORCH SAVE FORMATS:
    ---------------------
    1. state_dict (recommended for transfer learning)
       - Saves only the learned weights, not the model architecture
       - Smaller file size
       - Requires you to define the model class when loading
       - Use: torch.save(model.state_dict(), 'model_weights.pth')
       
    2. Full model (entire pickle)
       - Saves both architecture and weights
       - Larger file size
       - Can break if you change the code
       - Use: torch.save(model, 'model_full.pth')
       
    3. TorchScript (for production)
       - Compiled model that runs without Python
       - Can be loaded in C++, mobile, etc.
       - Use: torch.jit.script(model) or torch.jit.trace(model, input)
    """
    import torch
    
    print(f"\n📦 Saving PyTorch model from: {model_path}")
    
    # Create versioned output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"pytorch_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Copy original file
    if os.path.exists(model_path):
        shutil.copy(model_path, os.path.join(save_dir, os.path.basename(model_path)))
        print(f"  ✅ Copied original: {os.path.basename(model_path)}")
    
    print(f"\n  📁 Saved to: {save_dir}")
    return save_dir


def save_onnx_model(model_path, output_dir, quantize=False):
    """
    Save and optionally optimize an ONNX model.
    
    ONNX (Open Neural Network Exchange):
    ------------------------------------
    - Universal format supported by many frameworks
    - Can run on: ONNX Runtime, TensorRT, OpenVINO, CoreML, etc.
    - Your current models are already in ONNX format!
    
    QUANTIZATION:
    -------------
    Converts FP32 weights to INT8, making the model:
    - ~4x smaller
    - ~2-4x faster on CPU
    - Slight accuracy loss (usually <1%)
    
    TYPES OF QUANTIZATION:
    - Dynamic: Quantizes weights, activations computed at runtime
    - Static: Both weights and activations pre-quantized (needs calibration data)
    """
    import onnx
    
    print(f"\n📦 Saving ONNX model from: {model_path}")
    
    # Create versioned output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"onnx_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and verify the model
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("  ✅ Model is valid ONNX")
    
    # Get model info
    print("\n  📊 Model Info:")
    print(f"     - IR Version: {model.ir_version}")
    print(f"     - Producer: {model.producer_name}")
    print(f"     - Opset: {model.opset_import[0].version}")
    
    # Save original
    original_path = os.path.join(save_dir, "model_original.onnx")
    shutil.copy(model_path, original_path)
    original_size = os.path.getsize(original_path) / (1024 * 1024)
    print(f"\n  ✅ Saved original: model_original.onnx ({original_size:.2f} MB)")
    
    # Quantization (if requested)
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantized_path = os.path.join(save_dir, "model_quantized_int8.onnx")
            
            # Dynamic quantization (no calibration data needed)
            quantize_dynamic(
                model_input=original_path,
                model_output=quantized_path,
                weight_type=QuantType.QUInt8  # Quantize weights to 8-bit unsigned int
            )
            
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
            compression = (1 - quantized_size/original_size) * 100
            
            print(f"  ✅ Saved quantized: model_quantized_int8.onnx ({quantized_size:.2f} MB)")
            print(f"     - Compression: {compression:.1f}%")
            
        except ImportError:
            print("  ⚠️ Quantization requires: pip install onnxruntime")
        except Exception as e:
            print(f"  ⚠️ Quantization failed: {e}")
    
    # Save model metadata
    metadata_path = os.path.join(save_dir, "model_info.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Saved: {datetime.now().isoformat()}\n")
        f.write(f"IR Version: {model.ir_version}\n")
        f.write(f"Opset: {model.opset_import[0].version}\n")
        f.write(f"Original Size: {original_size:.2f} MB\n")
        
        # List inputs and outputs
        f.write("\nInputs:\n")
        for inp in model.graph.input:
            f.write(f"  - {inp.name}\n")
        f.write("\nOutputs:\n")
        for out in model.graph.output:
            f.write(f"  - {out.name}\n")
    
    print(f"  ✅ Saved metadata: model_info.txt")
    print(f"\n  📁 Saved to: {save_dir}")
    
    return save_dir


def create_transfer_learning_template():
    """
    Create a Python template for transfer learning with your models.
    """
    template = '''#!/usr/bin/env python3
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
'''
    
    return template


def main():
    """Main function to save models and create transfer learning resources."""
    
    parser = argparse.ArgumentParser(
        description="Save models for transfer learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python save_model_for_transfer_learning.py --list
  python save_model_for_transfer_learning.py --model resnet50 --quantize
  python save_model_for_transfer_learning.py --model cartpole
  python save_model_for_transfer_learning.py --create-template
        """
    )
    
    parser.add_argument('--list', action='store_true', 
                       help='List available models')
    parser.add_argument('--model', type=str, choices=['resnet50', 'cartpole', 'all'],
                       help='Which model to save')
    parser.add_argument('--quantize', action='store_true',
                       help='Create quantized (INT8) version')
    parser.add_argument('--output-dir', type=str, default='saved_models',
                       help='Output directory for saved models')
    parser.add_argument('--create-template', action='store_true',
                       help='Create transfer learning Python template')
    
    args = parser.parse_args()
    
    # Check dependencies
    deps = check_dependencies()
    print("📋 Dependencies:")
    for pkg, available in deps.items():
        status = "✅" if available else "❌"
        print(f"   {status} {pkg}")
    
    # Define model paths
    models_dir = "models"
    available_models = {
        'resnet50': os.path.join(models_dir, 'resnet50_imagenet.onnx'),
        'cartpole': os.path.join(models_dir, 'cartpole_policy.onnx'),
    }
    
    # List models
    if args.list:
        print("\n📋 Available Models:")
        for name, path in available_models.items():
            exists = "✅" if os.path.exists(path) else "❌"
            size = ""
            if os.path.exists(path):
                size = f" ({os.path.getsize(path) / (1024*1024):.2f} MB)"
            print(f"   {exists} {name}: {path}{size}")
        return
    
    # Create template
    if args.create_template:
        template = create_transfer_learning_template()
        template_path = os.path.join("python", "transfer_learning_template.py")
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        with open(template_path, 'w') as f:
            f.write(template)
        print(f"\n✅ Created transfer learning template: {template_path}")
        return
    
    # Save models
    if args.model:
        os.makedirs(args.output_dir, exist_ok=True)
        
        models_to_save = [args.model] if args.model != 'all' else list(available_models.keys())
        
        for model_name in models_to_save:
            model_path = available_models.get(model_name)
            
            if not model_path or not os.path.exists(model_path):
                print(f"\n❌ Model not found: {model_name}")
                continue
            
            if deps['onnx']:
                save_onnx_model(model_path, args.output_dir, quantize=args.quantize)
            else:
                print(f"\n⚠️ Install onnx to save models: pip install onnx")
    
    # Print help if no action specified
    if not any([args.list, args.model, args.create_template]):
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python scripts/save_model_for_transfer_learning.py --list")
        print("   python scripts/save_model_for_transfer_learning.py --model cartpole --quantize")


if __name__ == "__main__":
    main()
