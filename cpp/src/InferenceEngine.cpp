/**
 * InferenceEngine.cpp - ONNX-based Image Classification Engine
 * 
 * This implementation provides:
 * 1. Model loading (ONNX format)
 * 2. Image preprocessing (resize, normalize)
 * 3. Inference execution
 * 4. Top-K classification results
 * 
 * Design Pattern: PIMPL (Pointer to Implementation)
 * - Hides implementation details
 * - Allows binary compatibility
 * - Reduces compilation dependencies
 */

#include "InferenceEngine.hpp"
#include "Softmax.hpp"
#include "ImageUtils.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>

// Conditional ONNX Runtime support
#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

/**
 * PIMPL Implementation
 * Holds all internal state and ONNX Runtime objects
 */
struct InferenceEngine::Impl {
    std::string modelPath;
    bool initialized = false;
    int numClasses = 1000;  // ImageNet default
    int inputWidth = 224;
    int inputHeight = 224;
    int inputChannels = 3;
    
    std::vector<std::string> labels;
    
#ifdef HAS_ONNXRUNTIME
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "InferenceEngine"};
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif
    
    // Load labels from file
    bool loadLabels(const std::string& labelsPath) {
        std::ifstream file(labelsPath);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not load labels from " << labelsPath << std::endl;
            // Generate placeholder labels
            labels.resize(numClasses);
            for (int i = 0; i < numClasses; ++i) {
                labels[i] = "class_" + std::to_string(i);
            }
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                labels.push_back(line);
            }
        }
        std::cout << "Loaded " << labels.size() << " labels" << std::endl;
        return true;
    }
    
    // Preprocess image data for ImageNet
    std::vector<float> preprocessImage(const std::vector<uint8_t>& imageData, 
                                       int imgWidth, int imgHeight) {
        // ImageNet normalization values
        const float mean[] = {0.485f, 0.456f, 0.406f};  // RGB
        const float std_val[] = {0.229f, 0.224f, 0.225f};
        
        std::vector<float> tensor(inputChannels * inputHeight * inputWidth);
        
        // Simple resize using nearest neighbor (for demonstration)
        // In production, use bilinear interpolation
        for (int c = 0; c < inputChannels; ++c) {
            for (int h = 0; h < inputHeight; ++h) {
                for (int w = 0; w < inputWidth; ++w) {
                    // Map to source coordinates
                    int srcH = h * imgHeight / inputHeight;
                    int srcW = w * imgWidth / inputWidth;
                    
                    // Get pixel value (assuming CHW format input or convert from HWC)
                    int srcIdx = (srcH * imgWidth + srcW) * 3 + c;
                    float pixel = 0.0f;
                    
                    if (srcIdx < static_cast<int>(imageData.size())) {
                        pixel = imageData[srcIdx] / 255.0f;  // Normalize to [0, 1]
                    }
                    
                    // Apply ImageNet normalization
                    pixel = (pixel - mean[c]) / std_val[c];
                    
                    // Store in CHW format (PyTorch convention)
                    int dstIdx = c * inputHeight * inputWidth + h * inputWidth + w;
                    tensor[dstIdx] = pixel;
                }
            }
        }
        
        return tensor;
    }
};

InferenceEngine::InferenceEngine() : pImpl(std::make_unique<Impl>()) {}

InferenceEngine::~InferenceEngine() = default;

bool InferenceEngine::initialize(const std::string& modelPath) {
    try {
        pImpl->modelPath = modelPath;
        
        // Try to load labels from same directory
        std::string labelsPath = modelPath;
        size_t lastSlash = labelsPath.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
            labelsPath = labelsPath.substr(0, lastSlash + 1) + "labels_imagenet.txt";
        } else {
            labelsPath = "labels_imagenet.txt";
        }
        pImpl->loadLabels(labelsPath);
        
#ifdef HAS_ONNXRUNTIME
        // Create ONNX Runtime session with GPU support
        Ort::SessionOptions sessionOptions;
        
        // Try to enable CUDA GPU acceleration
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;  // 2GB limit
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "CUDA execution provider enabled" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "CUDA not available, falling back to CPU: " << e.what() << std::endl;
        }
        
        // Performance tuning
        sessionOptions.SetIntraOpNumThreads(4);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        
        pImpl->session = std::make_unique<Ort::Session>(
            pImpl->env, modelPath.c_str(), sessionOptions
        );
        
        std::cout << "ONNX Runtime session created successfully" << std::endl;
        
        // Get input dimensions from model
        auto inputInfo = pImpl->session->GetInputTypeInfo(0);
        auto tensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
        auto inputShape = tensorInfo.GetShape();
        
        if (inputShape.size() == 4) {
            pImpl->inputChannels = inputShape[1];
            pImpl->inputHeight = inputShape[2];
            pImpl->inputWidth = inputShape[3];
        }
        
        // Get number of output classes
        auto outputInfo = pImpl->session->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputInfo.GetTensorTypeAndShapeInfo();
        auto outputShape = outputTensorInfo.GetShape();
        
        if (outputShape.size() >= 2) {
            pImpl->numClasses = outputShape[1];
        }
#else
        std::cout << "Note: ONNX Runtime not available, using simulation mode" << std::endl;
        std::cout << "To enable real inference, install ONNX Runtime and rebuild with -DHAS_ONNXRUNTIME" << std::endl;
#endif
        
        pImpl->initialized = true;
        
        std::cout << "Model loaded: " << modelPath << std::endl;
        std::cout << "Input dimensions: " << pImpl->inputWidth << "x" 
                  << pImpl->inputHeight << "x" << pImpl->inputChannels << std::endl;
        std::cout << "Number of classes: " << pImpl->numClasses << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize engine: " << e.what() << std::endl;
        return false;
    }
}

// ============================================
// RL Policy Initialization
// ============================================

bool InferenceEngine::initializeRL(const std::string& modelPath) {
    try {
        pImpl->modelPath = modelPath;
        std::cout << "Initializing RL policy engine with model: " << modelPath << std::endl;
        
#ifdef HAS_ONNXRUNTIME
        // Create ONNX Runtime session for RL policy
        Ort::SessionOptions sessionOptions;
        
        // Try to enable CUDA GPU acceleration
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "CUDA execution provider enabled for RL" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "CUDA not available for RL, falling back to CPU: " << e.what() << std::endl;
        }
        
        // Performance tuning
        sessionOptions.SetIntraOpNumThreads(2);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        
        pImpl->session = std::make_unique<Ort::Session>(
            pImpl->env, modelPath.c_str(), sessionOptions
        );
        
        std::cout << "ONNX Runtime RL session created successfully" << std::endl;
        
        // Get input dimensions from model (expecting [batch, state_dim])
        auto inputInfo = pImpl->session->GetInputTypeInfo(0);
        auto tensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
        auto inputShape = tensorInfo.GetShape();
        
        if (inputShape.size() >= 2) {
            // For RL: store state dimension in inputWidth
            pImpl->inputWidth = inputShape[1];  // state_dim
            pImpl->inputHeight = 1;
            pImpl->inputChannels = 1;
        }
        
        // Get number of actions
        auto outputInfo = pImpl->session->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputInfo.GetTensorTypeAndShapeInfo();
        auto outputShape = outputTensorInfo.GetShape();
        
        if (outputShape.size() >= 2) {
            pImpl->numClasses = outputShape[1];  // num_actions
        }
        
        std::cout << "RL Model loaded: " << modelPath << std::endl;
        std::cout << "State dimension: " << pImpl->inputWidth << std::endl;
        std::cout << "Number of actions: " << pImpl->numClasses << std::endl;
#else
        std::cout << "Note: ONNX Runtime not available, using simulation mode for RL" << std::endl;
        pImpl->numClasses = 2;  // CartPole has 2 actions
#endif
        
        pImpl->initialized = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize RL engine: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> InferenceEngine::classifyImage(const std::string& imagePath) {
    if (!pImpl->initialized) {
        throw std::runtime_error("Engine not initialized. Call initialize() first.");
    }
    
    std::cout << "Classifying image: " << imagePath << std::endl;
    
#ifdef HAS_ONNXRUNTIME
    // Load and preprocess image
    std::vector<uint8_t> imageData;
    int imgWidth, imgHeight;
    
    // Try to load image (simplified - in production use OpenCV)
    if (!ImageUtils::loadImage(imagePath, imageData, imgWidth, imgHeight)) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }
    
    // Preprocess
    std::vector<float> inputTensor = pImpl->preprocessImage(imageData, imgWidth, imgHeight);
    
    // Prepare input
    std::array<int64_t, 4> inputShape = {1, pImpl->inputChannels, pImpl->inputHeight, pImpl->inputWidth};
    
    Ort::Value inputOrt = Ort::Value::CreateTensor<float>(
        pImpl->memoryInfo,
        inputTensor.data(),
        inputTensor.size(),
        inputShape.data(),
        inputShape.size()
    );
    
    // Run inference
    const char* inputNames[] = {"input"};
    const char* outputNames[] = {"logits"};
    
    auto outputTensors = pImpl->session->Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputOrt, 1,
        outputNames, 1
    );
    
    // Get output
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t outputSize = outputShape[1];
    
    std::vector<float> logits(outputData, outputData + outputSize);
    
    // Apply softmax
    return Softmax::compute(logits);
#else
    // Simulation mode - return dummy classification
    std::vector<float> results(pImpl->numClasses, 0.0f);
    
    // Simulate some classification results
    // In real mode, this would be actual inference output
    results[0] = 0.15f;   // tench
    results[1] = 0.45f;   // goldfish (highest)
    results[2] = 0.10f;   // great white shark
    results[3] = 0.08f;   // tiger shark
    results[4] = 0.05f;   // hammerhead
    
    // Fill remaining with small values
    float remaining = 1.0f - 0.83f;
    for (int i = 5; i < pImpl->numClasses; ++i) {
        results[i] = remaining / (pImpl->numClasses - 5);
    }
    
    std::cout << "\n=== Classification Results (Simulation Mode) ===" << std::endl;
    
    // Get top-5 predictions
    std::vector<std::pair<int, float>> predictions;
    for (int i = 0; i < pImpl->numClasses; ++i) {
        predictions.emplace_back(i, results[i]);
    }
    
    std::sort(predictions.begin(), predictions.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::cout << "\nTop-5 Predictions:" << std::endl;
    for (int i = 0; i < 5 && i < static_cast<int>(predictions.size()); ++i) {
        int classIdx = predictions[i].first;
        float confidence = predictions[i].second * 100.0f;
        std::string label = (classIdx < static_cast<int>(pImpl->labels.size())) 
                           ? pImpl->labels[classIdx] 
                           : "class_" + std::to_string(classIdx);
        std::cout << "  " << (i + 1) << ". " << label 
                  << " (" << confidence << "%)" << std::endl;
    }
    
    return results;
#endif
}

int InferenceEngine::getNumClasses() const {
    return pImpl->numClasses;
}

void InferenceEngine::getInputDimensions(int& width, int& height, int& channels) const {
    width = pImpl->inputWidth;
    height = pImpl->inputHeight;
    channels = pImpl->inputChannels;
}

std::string InferenceEngine::getVersion() const {
    return "1.0.0";
}

const std::vector<std::string>& InferenceEngine::getLabels() const {
    return pImpl->labels;
}

// Legacy API support (for JNI compatibility)
std::vector<float> InferenceEngine::softmax(const std::vector<float>& logits) {
    return Softmax::compute(logits);
}

int InferenceEngine::predict(const std::vector<float>& logits) {
    auto probs = softmax(logits);
    return static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

// ============================================
// RL Policy Inference (for CartPole, etc.)
// ============================================

std::vector<float> InferenceEngine::inferState(const std::vector<float>& state) {
    std::cout << "Running RL state inference on " << state.size() << "-dim state" << std::endl;
    
#ifdef HAS_ONNXRUNTIME
    if (!pImpl->initialized) {
        throw std::runtime_error("Engine not initialized. Call initialize() first.");
    }
    
    // Create input tensor from state vector
    std::array<int64_t, 2> inputShape = {1, static_cast<int64_t>(state.size())};
    
    std::vector<float> inputData = state;  // Copy for non-const
    
    Ort::Value inputOrt = Ort::Value::CreateTensor<float>(
        pImpl->memoryInfo,
        inputData.data(),
        inputData.size(),
        inputShape.data(),
        inputShape.size()
    );
    
    // Run inference - use "state" and "action_logits" for RL models
    const char* inputNames[] = {"state"};
    const char* outputNames[] = {"action_logits"};
    
    try {
        auto outputTensors = pImpl->session->Run(
            Ort::RunOptions{nullptr},
            inputNames, &inputOrt, 1,
            outputNames, 1
        );
        
        // Get output
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t outputSize = outputShape[1];
        
        std::vector<float> logits(outputData, outputData + outputSize);
        
        std::cout << "  Output logits: [";
        for (size_t i = 0; i < logits.size(); ++i) {
            std::cout << logits[i];
            if (i < logits.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        return logits;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        throw std::runtime_error(std::string("RL inference failed: ") + e.what());
    }
#else
    // Simulation mode - return random action logits for 2 actions (left/right)
    std::cout << "  (Simulation mode - returning dummy logits)" << std::endl;
    
    // Simple simulation: prefer action 1 (right) if pole leaning right (positive angle)
    float angle = (state.size() >= 3) ? state[2] : 0.0f;
    
    std::vector<float> logits(2);
    if (angle > 0) {
        logits[0] = -0.5f;  // left
        logits[1] = 0.5f;   // right
    } else {
        logits[0] = 0.5f;   // left
        logits[1] = -0.5f;  // right
    }
    
    return logits;
#endif
}
