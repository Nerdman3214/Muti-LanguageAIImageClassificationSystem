"""
ImageNet Model Exporter
======================

Purpose:
--------
This file loads a PRETRAINED ImageNet model (ResNet-50),
freezes it, and exports it to ONNX so that it can be used
by a high-performance C++ inference engine.

IMPORTANT CONCEPT:
------------------
We are NOT training ImageNet.
Training ImageNet from scratch requires:
- Millions of images
- Weeks of GPU time

Instead:
- We reuse learned weights
- Treat the model as a fixed feature extractor

This mirrors how real production systems work.

Design Patterns Used:
---------------------
1. Pipeline Pattern
   Input -> Preprocessing -> Model -> Export

2. Facade Pattern
   This file hides all ML complexity behind:
       "Give me an ONNX model"

3. Separation of Concerns
   - Training / exporting (Python)
   - Inference (C++)
   - Serving (Java)
"""

import torch
import torchvision.models as models

# -----------------------------
# 1️⃣ LOAD PRETRAINED MODEL
# -----------------------------

"""
ResNet-50 architecture:
- 50 layers deep
- Residual connections (skip connections)
- Solves vanishing gradient problem

Mathematical intuition:
-----------------------
Instead of learning:
    H(x)

ResNet learns:
    F(x) = H(x) - x
So:
    H(x) = F(x) + x

This makes optimization MUCH easier.
"""

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# -----------------------------
# 2️⃣ FREEZE MODEL PARAMETERS
# -----------------------------

"""
Why freeze weights?
-------------------
Because:
- We are not training
- We only want inference
- This prevents accidental gradient computation

In training terms:
------------------
Epochs = 0
No backpropagation
No optimizer
"""

model.eval()  # VERY IMPORTANT for inference mode

for param in model.parameters():
    param.requires_grad = False

# -----------------------------
# 3️⃣ DEFINE DUMMY INPUT
# -----------------------------

"""
ImageNet input format (CRITICAL):
--------------------------------
Shape: [batch_size, channels, height, width]
       [1, 3, 224, 224]

Channels:
- 3 = RGB

Why dummy input?
----------------
ONNX export traces the computation graph.
It needs ONE example input to understand tensor flow.
"""

dummy_input = torch.randn(1, 3, 224, 224)

# -----------------------------
# 4️⃣ EXPORT TO ONNX
# -----------------------------

"""
ONNX = Open Neural Network Exchange

Why ONNX?
---------
- Language agnostic
- Can be loaded by:
    - C++
    - Java
    - Rust
    - Python
- Supports GPU acceleration
- Industry standard for deployment

This ONNX file is what your C++ inference engine will load.
"""

import os

# Determine output path - save to models directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
output_path = os.path.join(project_root, "models", "resnet50_imagenet.onnx")

# Ensure models directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,        # store trained weights
    opset_version=17,          # use newer stable ONNX opset
    do_constant_folding=True,  # optimize graph
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
)

print(f"✅ ResNet-50 ImageNet model exported to {output_path}")
