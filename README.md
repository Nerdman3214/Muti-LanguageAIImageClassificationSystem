# Multi-Language AI Image Classification System

A sophisticated multi-language AI image classification system combining Python training, C++ inference optimization, and Java API exposure through JNI.

## Project Overview

This system provides:
- **Python**: Model training and export capabilities
- **C++**: High-performance inference engine with optimized image processing
- **Java**: API layer for integration into Java-based applications
- **JNI**: Bridge between Java and C++ for native performance

## Project Structure

```
MultiLanguageAIImageSystem/
├── python/          # Python training and utilities
├── cpp/             # C++ inference engine
├── java/            # Java API and controller
├── jni/             # Java Native Interface bindings
├── models/          # Pre-trained models and labels
├── api/             # API specifications
├── scripts/         # Build and deployment scripts
└── .vscode/         # VS Code configuration
```

## Requirements

- Python 3.7+
- C++ 17
- Java 21+
- CMake 3.10+
- GCC/G++ or Clang

## Building

### Python Dependencies

```bash
pip install -r python/requirements.txt
```

### C++ Build

```bash
bash scripts/build_cpp.sh
```

### Java Build

```bash
bash scripts/build_java.sh
```

## Usage

### Python

```python
# Training and export
python python/training/Training.py

# Export to ONNX
python python/export/export_imagenet_resnet50_to_onnx.py
```

### C++

```bash
./cpp/build/InferenceEngine <model_path> <image_path>
```

### Java

```java
AIController controller = new AIController();
// Use native inference through JNI
```

## Development

See `.vscode/settings.json` and `.vscode/launch.json` for debugging configurations.

## License

[Your License Here]
