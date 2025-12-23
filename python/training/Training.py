import os

# ============================================
# GPU Configuration for CUDA/TensorFlow
# ============================================
# Set XLA flags to find libdevice (required for GPU)
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import keras
from keras import layers, models
import numpy as np

# Configure GPU memory growth to prevent OOM errors
try:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"🎮 Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    else:
        print("⚠️ No GPU found, using CPU")
except Exception as e:
    print(f"GPU config warning: {e}")

# Determine output path - save to models directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
model_path = os.path.join(project_root, "models", "digit_model.h5")

# Ensure models directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Load dataset (MNIST)
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize and convert to float32
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Build CNN
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Reshape((28, 28, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training digit classifier...")
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# Save model
model.save(model_path)
print(f"✅ Model saved to {model_path}")
