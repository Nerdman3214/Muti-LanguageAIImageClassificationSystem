# Multi-Language AI Image Classification System

A production-ready multi-language AI image classification system that **trains models in Python, exports to ONNX, runs high-performance inference in C++, and exposes a REST API via Java/JNI**.

## 🎯 Project Overview

This system demonstrates professional multi-language ML engineering:

| Language | Responsibility | Key Files |
|----------|---------------|-----------|
| **Python** | Training & ONNX export | `python/training/`, `python/export/` |
| **C++** | High-performance inference | `cpp/src/InferenceEngine.cpp` |
| **Java** | REST API & application layer | `java/src/.../AIController.java` |
| **JNI** | Native bridge (thin layer) | `jni/InferenceJNI.cpp` |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT                                  │
│                    (curl, browser, app)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP POST /classify
┌─────────────────────────────────────────────────────────────────┐
│                      JAVA REST API                              │
│                    (AIController.java)                          │
│                  • Spark HTTP server                            │
│                  • File upload handling                         │
│                  • JSON response formatting                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ JNI call: nativeInfer(imagePath)
┌─────────────────────────────────────────────────────────────────┐
│                      JNI BRIDGE                                 │
│                   (InferenceJNI.cpp)                            │
│                  • String conversion                            │
│                  • Exception handling                           │
│                  • Singleton engine management                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ C++ method call
┌─────────────────────────────────────────────────────────────────┐
│                   C++ INFERENCE ENGINE                          │
│                  (InferenceEngine.cpp)                          │
│                  • ONNX Runtime session                         │
│                  • OpenCV image preprocessing                   │
│                  • ImageNet normalization                       │
│                  • Softmax (numerically stable)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ ONNX Runtime inference
┌─────────────────────────────────────────────────────────────────┐
│                      ONNX MODEL                                 │
│               (resnet50_imagenet.onnx)                          │
│                  • 1000 ImageNet classes                        │
│                  • 224×224×3 input                              │
│                  • ~25M parameters                              │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Muti-LanguageAIImageClassificationSystem/
├── python/
│   ├── training/Training.py          # Model training
│   ├── export/export_*.py            # ONNX export scripts
│   └── requirements.txt
├── cpp/
│   ├── include/
│   │   ├── InferenceEngine.hpp       # Main engine interface
│   │   ├── Softmax.hpp               # Numerically stable softmax
│   │   └── ImageUtils.hpp            # Image preprocessing
│   ├── src/
│   │   ├── InferenceEngine.cpp       # ONNX Runtime inference
│   │   ├── Softmax.cpp               # Softmax implementation
│   │   ├── ImageUtils.cpp            # OpenCV preprocessing
│   │   └── main.cpp                  # CLI demo
│   └── CMakeLists.txt
├── jni/
│   └── InferenceJNI.cpp              # JNI bridge (thin layer)
├── java/
│   ├── src/main/java/ai/controller/
│   │   └── AIController.java         # REST API server
│   └── pom.xml
├── models/
│   ├── resnet50_imagenet.onnx        # Pre-trained model
│   └── labels_imagenet.txt           # 1000 class labels
├── test_images/                      # Test images
├── scripts/                          # Build scripts
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with TensorFlow (training only)
- **C++17** compiler (g++ 9+)
- **Java 11+** with Maven
- **CMake 3.10+**
- **OpenCV 4.x**
- **ONNX Runtime 1.19+**

### 1. Download ONNX Runtime

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz
tar -xzf onnxruntime-linux-x64-1.19.2.tgz -C /opt/
export ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.19.2
```

### 2. Build C++ Engine

```bash
cd cpp
mkdir build && cd build
ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.19.2 cmake ..
make -j
```

### 3. Build Java API

```bash
cd java
mvn clean package -DskipTests
```

### 4. Run C++ CLI Demo

```bash
LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64-1.19.2/lib:cpp/build \
./cpp/build/InferenceEngine models/resnet50_imagenet.onnx test_images/dog.jpg
```

Expected output:
```
Top-5 Predictions:
  1. Golden Retriever (57.19%)
  2. Kuvasz (10.53%)
  3. Pyrenean Mountain Dog (8.65%)
  4. Labrador Retriever (8.50%)
  5. Cocker Spaniels (6.61%)
```

### 5. Run Java REST API

```bash
LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64-1.19.2/lib:cpp/build \
java -Djava.library.path=cpp/build \
     -jar java/target/MultiLanguageAIImageSystem-1.0.0.jar
```

Test the API:
```bash
curl -X POST -F "image=@test_images/dog.jpg" http://localhost:8080/classify | jq .
```

## 🔬 Technical Details

### Softmax Implementation (Numerically Stable)

```cpp
// Subtract max to prevent overflow
float maxVal = *std::max_element(logits.begin(), logits.end());
for (float& val : shifted) val = std::exp(val - maxVal);
// Normalize
float sum = std::accumulate(shifted.begin(), shifted.end(), 0.0f);
for (float& val : result) val /= sum;
```

### ImageNet Preprocessing

- Resize to 224×224
- Convert BGR→RGB
- Normalize: `(pixel/255 - mean) / std`
  - mean = [0.485, 0.456, 0.406]
  - std = [0.229, 0.224, 0.225]
- Convert HWC→CHW format

### JNI Bridge Pattern

```cpp
extern "C" {
JNIEXPORT jfloatArray JNICALL
Java_ai_controller_AIController_nativeInfer(JNIEnv* env, jobject, jstring imagePath) {
    // 1. Convert Java string → C++ string
    // 2. Call InferenceEngine::classifyImage()
    // 3. Convert C++ vector → Java float[]
    // 4. Handle exceptions → Java exceptions
}
}
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/classify` | Upload and classify an image |
| GET | `/health` | Health check |
| GET | `/info` | Model information |

### Example Response

```json
{
  "status": "success",
  "imageName": "dog.jpg",
  "predictions": [
    {"classIndex": 207, "label": "Golden Retriever", "confidence": 0.5719},
    {"classIndex": 222, "label": "Kuvasz", "confidence": 0.1053}
  ],
  "inferenceTimeMs": 131,
  "modelVersion": "1.0.0"
}
```

## ✅ Submission Checklist

- [x] Python = training only (no inference)
- [x] C++ = inference only (no training)
- [x] Java = API layer (no model loading)
- [x] JNI = thin bridge (no business logic)
- [x] ONNX = single model format
- [x] No TensorFlow in C++
- [x] Softmax sums to 1.0
- [x] Both CLI and REST demos work
- [x] README with architecture diagram

## 📜 License

MIT License
