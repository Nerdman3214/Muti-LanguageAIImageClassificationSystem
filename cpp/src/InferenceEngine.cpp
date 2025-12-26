/**
 * =============================================================================
 * InferenceEngine.cpp - ONNX-Based Neural Network Inference Engine
 * =============================================================================
 * 
 * This is the core C++ implementation that performs actual AI inference using
 * ONNX Runtime. It supports both image classification (ResNet50) and
 * reinforcement learning policy inference (CartPole).
 * 
 * ONNX RUNTIME OVERVIEW:
 * ----------------------
 * ONNX (Open Neural Network Exchange) is a standard format for ML models.
 * ONNX Runtime is a high-performance inference engine that can run ONNX models
 * on various hardware (CPU, GPU via CUDA/TensorRT, etc.).
 * 
 * KEY CONCEPTS:
 * - Session: A loaded model ready for inference
 * - Tensor: Multi-dimensional array (the data structure neural networks use)
 * - ExecutionProvider: Backend that runs the computations (CPU, CUDA, etc.)
 * 
 * IMAGE CLASSIFICATION PIPELINE:
 * 1. Load image from disk
 * 2. Resize to model's expected size (224x224 for ResNet50)
 * 3. Normalize pixels using ImageNet statistics
 * 4. Convert to tensor format (NCHW: batch, channels, height, width)
 * 5. Run through neural network
 * 6. Apply softmax to get probabilities
 * 7. Return top-K predictions
 * 
 * REINFORCEMENT LEARNING PIPELINE:
 * 1. Receive state vector (e.g., CartPole's 4 observations)
 * 2. Convert to tensor format (batch, state_dim)
 * 3. Run through policy network
 * 4. Return action logits (converted to probabilities in Java/Python)
 * 
 * DESIGN PATTERN: PIMPL (Pointer to Implementation)
 * - "Pimpl" = "Pointer to implementation"
 * - Hides all implementation details in a private struct
 * - Benefits:
 *   1. Binary compatibility: Can change implementation without recompiling users
 *   2. Faster compilation: Changes to impl don't trigger recompilation of headers
 *   3. Clean interface: Header only shows public API
 * 
 * @file InferenceEngine.cpp
 * @author Multi-Language AI System
 * @version 1.0.0
 */

// ============================================================================
// INCLUDES
// ============================================================================

/**
 * Our public header file.
 * Declares the InferenceEngine class interface that others include.
 */
#include "InferenceEngine.hpp"

/**
 * Softmax utility for converting logits to probabilities.
 * Softmax: P(i) = exp(logit_i) / Σ exp(logit_j)
 */
#include "Softmax.hpp"

/**
 * Image loading utilities (uses OpenCV if available, otherwise fallback).
 */
#include "ImageUtils.hpp"

// Standard C++ library headers
#include <iostream>    // std::cout, std::cerr for console output
#include <fstream>     // std::ifstream for file reading (labels file)
#include <sstream>     // std::stringstream for string manipulation
#include <stdexcept>   // std::runtime_error for exception handling
#include <algorithm>   // std::sort, std::max_element for array operations
#include <numeric>     // Numeric algorithms (not currently used but useful)
#include <cmath>       // Math functions like exp(), sqrt() (used in softmax)

/**
 * CONDITIONAL COMPILATION:
 * HAS_ONNXRUNTIME is defined by CMake if ONNX Runtime was found.
 * If not defined, we compile in "simulation mode" for testing without ONNX.
 */
#ifdef HAS_ONNXRUNTIME
/**
 * ONNX Runtime C++ API header.
 * This provides C++ wrapper classes for the C API, making it safer and easier to use.
 */
#include <onnxruntime_cxx_api.h>
#endif

// ============================================================================
// PIMPL IMPLEMENTATION CLASS
// ============================================================================

/**
 * Private implementation structure for InferenceEngine.
 * 
 * This struct contains ALL private data and helper functions.
 * It's only visible in this .cpp file, not in the header.
 * 
 * WHY PIMPL?
 * Consider this scenario without PIMPL:
 *   - Header includes <onnxruntime_cxx_api.h>
 *   - Every file that includes InferenceEngine.hpp now also includes ONNX headers
 *   - Compile time explodes, and users need ONNX headers installed
 * 
 * With PIMPL:
 *   - Header just declares: struct Impl; std::unique_ptr<Impl> pImpl;
 *   - ONNX headers only included in .cpp file
 *   - Users don't need ONNX headers to use InferenceEngine
 */
struct InferenceEngine::Impl {
    // ========================================================================
    // CONFIGURATION STATE
    // ========================================================================
    
    /** Path to the loaded ONNX model file */
    std::string modelPath;
    
    /** Whether the engine is ready for inference */
    bool initialized = false;
    
    /**
     * Number of output classes.
     * - For ImageNet: 1000 (the standard ImageNet classes)
     * - For CartPole: 2 (push left, push right)
     */
    int numClasses = 1000;  // ImageNet default
    
    /**
     * Expected input image dimensions.
     * ResNet50 expects 224x224 RGB images.
     * These values are read from the model's metadata at load time.
     */
    int inputWidth = 224;
    int inputHeight = 224;
    int inputChannels = 3;  // RGB = 3 channels
    
    /**
     * Human-readable class labels.
     * For ImageNet, these are things like "goldfish", "great white shark", etc.
     * Loaded from labels_imagenet.txt file.
     */
    std::vector<std::string> labels;
    
    // ========================================================================
    // ONNX RUNTIME OBJECTS
    // ========================================================================
    
#ifdef HAS_ONNXRUNTIME
    /**
     * ONNX Runtime Environment.
     * 
     * The Env object holds global state for ONNX Runtime:
     * - Logging configuration
     * - Thread pools
     * - Memory allocators
     * 
     * One Env can be shared across multiple Sessions.
     * We initialize it with WARNING level logging to reduce noise.
     */
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "InferenceEngine"};
    
    /**
     * ONNX Runtime Session.
     * 
     * A Session is a loaded model ready for inference.
     * Creating a session:
     * 1. Reads the ONNX file
     * 2. Parses the model graph
     * 3. Optimizes the graph for the target hardware
     * 4. Allocates memory for intermediate tensors
     * 
     * This is expensive, which is why we do it once and reuse.
     * 
     * std::unique_ptr: Smart pointer that auto-deletes when Impl is destroyed.
     */
    std::unique_ptr<Ort::Session> session;
    
    /**
     * Memory information for creating input tensors.
     * 
     * Tensors need to know where their memory is located:
     * - OrtArenaAllocator: Uses a memory pool for efficiency
     * - OrtMemTypeDefault: Standard CPU memory
     * 
     * For GPU inference, this would be configured differently.
     */
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif
    
    // ========================================================================
    // HELPER METHODS
    // ========================================================================
    
    /**
     * Loads class labels from a text file.
     * 
     * The labels file should have one label per line:
     *   tench
     *   goldfish
     *   great white shark
     *   ...
     * 
     * @param labelsPath Path to the labels text file
     * @return true if labels were loaded successfully, false otherwise
     */
    bool loadLabels(const std::string& labelsPath) {
        // Open file using ifstream (input file stream)
        std::ifstream file(labelsPath);
        
        // Check if file opened successfully
        if (!file.is_open()) {
            std::cerr << "Warning: Could not load labels from " << labelsPath << std::endl;
            
            // Generate placeholder labels if file not found
            // This allows the engine to work even without a labels file
            labels.resize(numClasses);
            for (int i = 0; i < numClasses; ++i) {
                labels[i] = "class_" + std::to_string(i);
            }
            return false;
        }
        
        // Read file line by line
        std::string line;
        while (std::getline(file, line)) {
            // Skip empty lines
            if (!line.empty()) {
                labels.push_back(line);
            }
        }
        
        std::cout << "Loaded " << labels.size() << " labels" << std::endl;
        return true;
    }
    
    /**
     * Preprocesses raw image data for neural network input.
     * 
     * PREPROCESSING STEPS (ImageNet standard):
     * 1. Resize to 224x224 (model's expected input size)
     * 2. Convert to float and normalize to [0, 1]
     * 3. Subtract ImageNet mean: [0.485, 0.456, 0.406] (RGB)
     * 4. Divide by ImageNet std: [0.229, 0.224, 0.225] (RGB)
     * 5. Rearrange from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
     * 
     * WHY NORMALIZE?
     * Neural networks train faster when inputs have zero mean and unit variance.
     * ImageNet models were trained with these specific normalization values,
     * so we must use the same values at inference time.
     * 
     * WHY CHW FORMAT?
     * PyTorch models (like ResNet50) expect CHW format.
     * This is because convolutions are more efficient when channels are contiguous.
     * 
     * @param imageData Raw pixel data (RGB, HWC format, values 0-255)
     * @param imgWidth  Original image width
     * @param imgHeight Original image height
     * @return Preprocessed tensor data ready for ONNX Runtime
     */
    std::vector<float> preprocessImage(const std::vector<uint8_t>& imageData, 
                                       int imgWidth, int imgHeight) {
        // ImageNet normalization values (computed from training dataset)
        // These are in RGB order
        const float mean[] = {0.485f, 0.456f, 0.406f};      // Mean pixel values
        const float std_val[] = {0.229f, 0.224f, 0.225f};   // Standard deviations
        
        // Allocate output tensor: C × H × W floats
        std::vector<float> tensor(inputChannels * inputHeight * inputWidth);
        
        /**
         * RESIZE + NORMALIZE + REARRANGE in one pass.
         * 
         * We iterate over the output tensor dimensions and sample from the input.
         * This is "nearest neighbor" interpolation - simple but not optimal.
         * Production code should use bilinear interpolation for better quality.
         */
        for (int c = 0; c < inputChannels; ++c) {          // For each channel (R, G, B)
            for (int h = 0; h < inputHeight; ++h) {        // For each output row
                for (int w = 0; w < inputWidth; ++w) {     // For each output column
                    
                    // Map output coordinates to input coordinates (nearest neighbor)
                    // This effectively scales the image to fit 224x224
                    int srcH = h * imgHeight / inputHeight;
                    int srcW = w * imgWidth / inputWidth;
                    
                    // Calculate index in source HWC image
                    // HWC means: row * width * 3 + column * 3 + channel
                    int srcIdx = (srcH * imgWidth + srcW) * 3 + c;
                    
                    float pixel = 0.0f;
                    
                    // Bounds check (prevent buffer overrun)
                    if (srcIdx < static_cast<int>(imageData.size())) {
                        // Convert from uint8 [0, 255] to float [0, 1]
                        pixel = imageData[srcIdx] / 255.0f;
                    }
                    
                    // Apply ImageNet normalization: (pixel - mean) / std
                    pixel = (pixel - mean[c]) / std_val[c];
                    
                    // Calculate index in output CHW tensor
                    // CHW means: channel * (height * width) + row * width + column
                    int dstIdx = c * inputHeight * inputWidth + h * inputWidth + w;
                    tensor[dstIdx] = pixel;
                }
            }
        }
        
        return tensor;
    }
};

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

/**
 * Default constructor - creates the PIMPL implementation object.
 * 
 * std::make_unique creates a new Impl on the heap and wraps it in unique_ptr.
 * This is the only place Impl is allocated.
 */
InferenceEngine::InferenceEngine() : pImpl(std::make_unique<Impl>()) {}

/**
 * Destructor - uses default implementation.
 * 
 * We need to declare this in the .cpp file (not header) because:
 * - unique_ptr destructor needs to know the complete type of Impl
 * - In the header, Impl is only forward-declared
 * - Here in .cpp, Impl is fully defined, so destruction works
 * 
 * "= default" tells compiler to generate the standard destructor behavior.
 */
InferenceEngine::~InferenceEngine() = default;

// ============================================================================
// IMAGE CLASSIFICATION INITIALIZATION
// ============================================================================

/**
 * Initializes the engine with an image classification model.
 * 
 * This function:
 * 1. Loads labels from accompanying text file
 * 2. Creates ONNX Runtime session with GPU support (if available)
 * 3. Reads input/output dimensions from model metadata
 * 
 * @param modelPath Path to the .onnx model file
 * @return true if initialization succeeded, false otherwise
 */
bool InferenceEngine::initialize(const std::string& modelPath) {
    try {
        // Store model path for later reference
        pImpl->modelPath = modelPath;
        
        // ====================================================================
        // LOAD CLASS LABELS
        // ====================================================================
        
        /**
         * Try to find labels file in the same directory as the model.
         * Example: models/resnet50_imagenet.onnx → models/labels_imagenet.txt
         */
        std::string labelsPath = modelPath;
        
        // Find the last path separator (/ or \)
        size_t lastSlash = labelsPath.find_last_of("/\\");
        
        if (lastSlash != std::string::npos) {
            // Extract directory and append labels filename
            labelsPath = labelsPath.substr(0, lastSlash + 1) + "labels_imagenet.txt";
        } else {
            // No directory separator - model is in current directory
            labelsPath = "labels_imagenet.txt";
        }
        
        // Load labels (will use placeholders if file not found)
        pImpl->loadLabels(labelsPath);
        
        // ====================================================================
        // CREATE ONNX RUNTIME SESSION
        // ====================================================================
        
#ifdef HAS_ONNXRUNTIME
        /**
         * SessionOptions control how the model is loaded and executed.
         */
        Ort::SessionOptions sessionOptions;
        
        /**
         * TRY TO ENABLE CUDA GPU ACCELERATION
         * 
         * GPU inference is much faster than CPU for neural networks.
         * We wrap this in try-catch because CUDA might not be available.
         */
        try {
            // Configure CUDA execution provider
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;                          // Use first GPU
            cuda_options.arena_extend_strategy = 0;              // Memory allocation strategy
            cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;  // 2GB GPU memory limit
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;  // Best conv algorithm
            cuda_options.do_copy_in_default_stream = 1;          // Sync behavior
            
            // Add CUDA provider to session options
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "CUDA execution provider enabled" << std::endl;
            
        } catch (const std::exception& e) {
            // CUDA not available - this is okay, will fall back to CPU
            std::cerr << "CUDA not available, falling back to CPU: " << e.what() << std::endl;
        }
        
        /**
         * PERFORMANCE TUNING OPTIONS
         */
        sessionOptions.SetIntraOpNumThreads(4);  // Threads for parallel operations within one op
        
        // Enable all graph optimizations (constant folding, operator fusion, etc.)
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Sequential execution (vs parallel) - often faster for single inference
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        
        /**
         * Create the session - this loads and compiles the model.
         * This is the slow part - can take seconds for large models.
         */
        pImpl->session = std::make_unique<Ort::Session>(
            pImpl->env,          // ONNX Runtime environment
            modelPath.c_str(),   // Path to .onnx file
            sessionOptions       // Configuration options
        );
        
        std::cout << "ONNX Runtime session created successfully" << std::endl;
        
        // ====================================================================
        // READ MODEL METADATA
        // ====================================================================
        
        /**
         * Get input tensor information from the model.
         * This tells us what size images the model expects.
         */
        auto inputInfo = pImpl->session->GetInputTypeInfo(0);  // First (only) input
        auto tensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
        auto inputShape = tensorInfo.GetShape();
        
        // Parse shape: [batch_size, channels, height, width]
        if (inputShape.size() == 4) {
            pImpl->inputChannels = inputShape[1];  // Usually 3 (RGB)
            pImpl->inputHeight = inputShape[2];     // Usually 224
            pImpl->inputWidth = inputShape[3];      // Usually 224
        }
        
        /**
         * Get output tensor information.
         * This tells us how many classes the model predicts.
         */
        auto outputInfo = pImpl->session->GetOutputTypeInfo(0);  // First output
        auto outputTensorInfo = outputInfo.GetTensorTypeAndShapeInfo();
        auto outputShape = outputTensorInfo.GetShape();
        
        // Parse shape: [batch_size, num_classes]
        if (outputShape.size() >= 2) {
            pImpl->numClasses = outputShape[1];  // Usually 1000 for ImageNet
        }
        
#else
        // No ONNX Runtime - run in simulation mode
        std::cout << "Note: ONNX Runtime not available, using simulation mode" << std::endl;
        std::cout << "To enable real inference, install ONNX Runtime and rebuild with -DHAS_ONNXRUNTIME" << std::endl;
#endif
        
        // Mark as initialized
        pImpl->initialized = true;
        
        // Log configuration
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

// ============================================================================
// REINFORCEMENT LEARNING INITIALIZATION
// ============================================================================

/**
 * Initializes the engine with an RL policy model.
 * 
 * RL models differ from classification models:
 * - Input: State vector (e.g., 4 floats for CartPole)
 * - Output: Action logits (e.g., 2 floats for left/right)
 * 
 * The model architecture is simpler (fully-connected layers vs convolutions)
 * so inference is very fast.
 * 
 * @param modelPath Path to the RL policy .onnx model file
 * @return true if initialization succeeded, false otherwise
 */
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
        
        // Performance tuning - RL models are small, fewer threads is fine
        sessionOptions.SetIntraOpNumThreads(2);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        
        // Create session
        pImpl->session = std::make_unique<Ort::Session>(
            pImpl->env, modelPath.c_str(), sessionOptions
        );
        
        std::cout << "ONNX Runtime RL session created successfully" << std::endl;
        
        /**
         * Read RL model dimensions.
         * RL models have different I/O shapes than image models:
         * - Input: [batch, state_dim] e.g., [1, 4] for CartPole
         * - Output: [batch, num_actions] e.g., [1, 2] for CartPole
         */
        auto inputInfo = pImpl->session->GetInputTypeInfo(0);
        auto tensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
        auto inputShape = tensorInfo.GetShape();
        
        if (inputShape.size() >= 2) {
            // For RL: store state dimension in inputWidth
            // We repurpose these fields since RL doesn't use images
            pImpl->inputWidth = inputShape[1];  // state_dim
            pImpl->inputHeight = 1;
            pImpl->inputChannels = 1;
        }
        
        // Get number of actions from output shape
        auto outputInfo = pImpl->session->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputInfo.GetTensorTypeAndShapeInfo();
        auto outputShape = outputTensorInfo.GetShape();
        
        if (outputShape.size() >= 2) {
            pImpl->numClasses = outputShape[1];  // Repurpose as num_actions
        }
        
        std::cout << "RL Model loaded: " << modelPath << std::endl;
        std::cout << "State dimension: " << pImpl->inputWidth << std::endl;
        std::cout << "Number of actions: " << pImpl->numClasses << std::endl;
        
#else
        // Simulation mode for RL
        std::cout << "Note: ONNX Runtime not available, using simulation mode for RL" << std::endl;
        pImpl->numClasses = 2;  // CartPole has 2 actions: left and right
#endif
        
        pImpl->initialized = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize RL engine: " << e.what() << std::endl;
        return false;
    }
}
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

// ============================================================================
// IMAGE CLASSIFICATION INFERENCE
// ============================================================================

/**
 * Classifies an image and returns class probabilities.
 * 
 * INFERENCE PIPELINE:
 * 1. Load image from file (using OpenCV or fallback loader)
 * 2. Preprocess: resize, normalize, convert to CHW tensor
 * 3. Create ONNX Runtime input tensor
 * 4. Run forward pass through neural network
 * 5. Apply softmax to convert logits to probabilities
 * 6. Return probability distribution over all classes
 * 
 * @param imagePath Path to the image file (supports JPEG, PNG, etc.)
 * @return Vector of probabilities (length = numClasses, sums to ~1.0)
 * @throws std::runtime_error if not initialized or image can't be loaded
 */
std::vector<float> InferenceEngine::classifyImage(const std::string& imagePath) {
    // Verify engine is initialized
    if (!pImpl->initialized) {
        throw std::runtime_error("Engine not initialized. Call initialize() first.");
    }
    
    std::cout << "Classifying image: " << imagePath << std::endl;
    
#ifdef HAS_ONNXRUNTIME
    // ========================================================================
    // LOAD AND PREPROCESS IMAGE
    // ========================================================================
    
    // Image data buffer (raw RGB pixels)
    std::vector<uint8_t> imageData;
    int imgWidth, imgHeight;
    
    /**
     * Load image using ImageUtils.
     * This uses OpenCV if available, otherwise a simple fallback loader.
     * Returns raw RGB pixel data in HWC format.
     */
    if (!ImageUtils::loadImage(imagePath, imageData, imgWidth, imgHeight)) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }
    
    /**
     * Preprocess the image:
     * - Resize to 224x224
     * - Normalize with ImageNet mean/std
     * - Convert from HWC to CHW format
     */
    std::vector<float> inputTensor = pImpl->preprocessImage(imageData, imgWidth, imgHeight);
    
    // ========================================================================
    // CREATE ONNX RUNTIME INPUT TENSOR
    // ========================================================================
    
    /**
     * Define input shape: [batch_size, channels, height, width]
     * We process one image at a time, so batch_size = 1.
     * 
     * std::array is used because ONNX Runtime needs a contiguous array of dimensions.
     */
    std::array<int64_t, 4> inputShape = {
        1,                       // batch size
        pImpl->inputChannels,    // channels (3 for RGB)
        pImpl->inputHeight,      // height (224)
        pImpl->inputWidth        // width (224)
    };
    
    /**
     * Create an ONNX Runtime tensor that wraps our data.
     * 
     * CreateTensor takes:
     * - memoryInfo: Where the data lives (CPU in our case)
     * - data pointer: Our preprocessed float array
     * - data size: Total number of floats
     * - shape array: The dimensions
     * - shape size: Number of dimensions (4 for images)
     * 
     * IMPORTANT: This doesn't copy the data! The tensor references our vector.
     * The vector must stay alive until after Run() completes.
     */
    Ort::Value inputOrt = Ort::Value::CreateTensor<float>(
        pImpl->memoryInfo,
        inputTensor.data(),       // Pointer to our data
        inputTensor.size(),       // Number of elements
        inputShape.data(),        // Pointer to shape array
        inputShape.size()         // Number of dimensions
    );
    
    // ========================================================================
    // RUN INFERENCE
    // ========================================================================
    
    /**
     * Input and output tensor names.
     * These must match the names in the ONNX model.
     * You can inspect these using: python -c "import onnx; m=onnx.load('model.onnx'); print([i.name for i in m.graph.input])"
     */
    const char* inputNames[] = {"input"};    // ResNet50's input tensor name
    const char* outputNames[] = {"logits"};  // ResNet50's output tensor name
    
    /**
     * Run the model!
     * 
     * This is where all the computation happens:
     * - Convolutions, batch norms, ReLUs, pooling, fully-connected layers
     * - For ResNet50: ~50 layers, ~25 million parameters
     * - Takes ~10-50ms on CPU, ~1-5ms on GPU
     */
    auto outputTensors = pImpl->session->Run(
        Ort::RunOptions{nullptr},   // Run options (nullptr = defaults)
        inputNames, &inputOrt, 1,   // Input names, values, count
        outputNames, 1              // Output names, count
    );
    
    // ========================================================================
    // EXTRACT OUTPUT
    // ========================================================================
    
    /**
     * Get pointer to output data.
     * The output tensor contains 1000 logits (one per ImageNet class).
     */
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    
    // Get output size from tensor shape
    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t outputSize = outputShape[1];  // Second dimension is num_classes
    
    // Copy to vector (logits are raw, unbounded scores)
    std::vector<float> logits(outputData, outputData + outputSize);
    
    /**
     * Apply softmax to convert logits to probabilities.
     * After softmax:
     * - All values are in [0, 1]
     * - All values sum to 1.0
     * - Higher value = more confident in that class
     */
    return Softmax::compute(logits);
    
#else
    // ========================================================================
    // SIMULATION MODE (No ONNX Runtime)
    // ========================================================================
    
    /**
     * When ONNX Runtime is not available, return simulated results.
     * This is useful for testing the pipeline without requiring ONNX Runtime.
     */
    std::vector<float> results(pImpl->numClasses, 0.0f);
    
    // Simulate some classification results (these are made up)
    results[0] = 0.15f;   // tench (a type of fish)
    results[1] = 0.45f;   // goldfish (highest confidence)
    results[2] = 0.10f;   // great white shark
    results[3] = 0.08f;   // tiger shark
    results[4] = 0.05f;   // hammerhead shark
    
    // Fill remaining classes with small uniform probabilities
    float remaining = 1.0f - 0.83f;
    for (int i = 5; i < pImpl->numClasses; ++i) {
        results[i] = remaining / (pImpl->numClasses - 5);
    }
    
    std::cout << "\n=== Classification Results (Simulation Mode) ===" << std::endl;
    
    // ========================================================================
    // DISPLAY TOP-5 PREDICTIONS
    // ========================================================================
    
    /**
     * Create pairs of (class_index, probability) for sorting.
     * We want to find the 5 classes with highest probability.
     */
    std::vector<std::pair<int, float>> predictions;
    for (int i = 0; i < pImpl->numClasses; ++i) {
        predictions.emplace_back(i, results[i]);
    }
    
    /**
     * Sort by probability (descending).
     * std::sort with custom comparator using lambda function.
     * Lambda: [](params) { body } - anonymous function
     */
    std::sort(predictions.begin(), predictions.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Display top 5
    std::cout << "\nTop-5 Predictions:" << std::endl;
    for (int i = 0; i < 5 && i < static_cast<int>(predictions.size()); ++i) {
        int classIdx = predictions[i].first;
        float confidence = predictions[i].second * 100.0f;  // Convert to percentage
        
        // Get label if available, otherwise use generic name
        std::string label = (classIdx < static_cast<int>(pImpl->labels.size())) 
                           ? pImpl->labels[classIdx] 
                           : "class_" + std::to_string(classIdx);
                           
        std::cout << "  " << (i + 1) << ". " << label 
                  << " (" << confidence << "%)" << std::endl;
    }
    
    return results;
#endif
}

// ============================================================================
// ACCESSOR METHODS
// ============================================================================

/**
 * Gets the number of classes the model can classify.
 * @return Number of output classes (1000 for ImageNet, 2 for CartPole)
 */
int InferenceEngine::getNumClasses() const {
    return pImpl->numClasses;
}

/**
 * Gets the expected input dimensions.
 * @param[out] width  Image width (224 for ResNet50)
 * @param[out] height Image height (224 for ResNet50)  
 * @param[out] channels Number of channels (3 for RGB)
 */
void InferenceEngine::getInputDimensions(int& width, int& height, int& channels) const {
    width = pImpl->inputWidth;
    height = pImpl->inputHeight;
    channels = pImpl->inputChannels;
}

/**
 * Gets the engine version string.
 * @return Version string (e.g., "1.0.0")
 */
std::string InferenceEngine::getVersion() const {
    return "1.0.0";
}

/**
 * Gets the loaded class labels.
 * @return Reference to the labels vector
 */
const std::vector<std::string>& InferenceEngine::getLabels() const {
    return pImpl->labels;
}

// ============================================================================
// LEGACY API SUPPORT
// ============================================================================

/**
 * Applies softmax to a vector of logits.
 * 
 * This is a legacy API - prefer using Softmax::compute() directly.
 * Kept for backward compatibility with JNI code.
 * 
 * @param logits Raw model outputs
 * @return Probabilities (sum to 1.0)
 */
std::vector<float> InferenceEngine::softmax(const std::vector<float>& logits) {
    return Softmax::compute(logits);
}

/**
 * Returns the class index with highest probability.
 * 
 * Legacy API - combines softmax + argmax.
 * 
 * @param logits Raw model outputs
 * @return Index of the predicted class
 */
int InferenceEngine::predict(const std::vector<float>& logits) {
    auto probs = softmax(logits);
    // std::max_element returns iterator to maximum element
    // Subtracting begin() gives the index
    return static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

// ============================================================================
// REINFORCEMENT LEARNING INFERENCE
// ============================================================================

/**
 * Runs RL policy inference on a state vector.
 * 
 * This function takes the current environment state and returns action logits.
 * The Java/Python layer converts these logits to probabilities using softmax,
 * then either samples from the distribution (during training) or takes the
 * argmax (during evaluation).
 * 
 * CARTPOLE STATE VECTOR:
 * - state[0]: Cart position (meters, typically -4.8 to 4.8)
 * - state[1]: Cart velocity (m/s)
 * - state[2]: Pole angle (radians, typically -0.42 to 0.42)
 * - state[3]: Pole angular velocity (rad/s)
 * 
 * OUTPUT LOGITS:
 * - logits[0]: Score for action 0 (push left)
 * - logits[1]: Score for action 1 (push right)
 * 
 * Higher logit = more preferred action.
 * 
 * @param state Environment state vector
 * @return Action logits (unnormalized log-probabilities)
 * @throws std::runtime_error if inference fails
 */
std::vector<float> InferenceEngine::inferState(const std::vector<float>& state) {
    std::cout << "Running RL state inference on " << state.size() << "-dim state" << std::endl;
    
#ifdef HAS_ONNXRUNTIME
    // Verify engine is ready
    if (!pImpl->initialized) {
        throw std::runtime_error("Engine not initialized. Call initialize() first.");
    }
    
    // ========================================================================
    // CREATE INPUT TENSOR
    // ========================================================================
    
    /**
     * RL input shape: [batch_size, state_dim]
     * Unlike images (4D), RL states are simple 2D tensors.
     */
    std::array<int64_t, 2> inputShape = {
        1,                                        // batch size
        static_cast<int64_t>(state.size())        // state dimension
    };
    
    // Make a mutable copy (CreateTensor needs non-const pointer)
    std::vector<float> inputData = state;
    
    // Create ONNX tensor wrapping our state data
    Ort::Value inputOrt = Ort::Value::CreateTensor<float>(
        pImpl->memoryInfo,
        inputData.data(),
        inputData.size(),
        inputShape.data(),
        inputShape.size()
    );
    
    // ========================================================================
    // RUN INFERENCE
    // ========================================================================
    
    /**
     * RL model uses different I/O names than image classification.
     * These match what we defined in train_cartpole.py when exporting to ONNX.
     */
    const char* inputNames[] = {"state"};          // Input tensor name
    const char* outputNames[] = {"action_logits"}; // Output tensor name
    
    try {
        // Run the policy network
        auto outputTensors = pImpl->session->Run(
            Ort::RunOptions{nullptr},
            inputNames, &inputOrt, 1,
            outputNames, 1
        );
        
        // ====================================================================
        // EXTRACT OUTPUT LOGITS
        // ====================================================================
        
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t outputSize = outputShape[1];  // Number of actions
        
        std::vector<float> logits(outputData, outputData + outputSize);
        
        // Debug output
        std::cout << "  Output logits: [";
        for (size_t i = 0; i < logits.size(); ++i) {
            std::cout << logits[i];
            if (i < logits.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        return logits;
        
    } catch (const Ort::Exception& e) {
        // ONNX Runtime specific exception
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        throw std::runtime_error(std::string("RL inference failed: ") + e.what());
    }
    
#else
    // ========================================================================
    // SIMULATION MODE
    // ========================================================================
    
    /**
     * Without ONNX Runtime, use a simple heuristic:
     * If pole is leaning right (positive angle), prefer pushing right.
     * If pole is leaning left (negative angle), prefer pushing left.
     * 
     * This is NOT a trained policy - just a simple rule for testing.
     */
    std::cout << "  (Simulation mode - returning dummy logits)" << std::endl;
    
    // Extract pole angle from state (index 2 in CartPole)
    float angle = (state.size() >= 3) ? state[2] : 0.0f;
    
    // Create action logits based on heuristic
    std::vector<float> logits(2);
    if (angle > 0) {
        // Pole leaning right → prefer action 1 (push right)
        logits[0] = -0.5f;  // left action logit
        logits[1] = 0.5f;   // right action logit
    } else {
        // Pole leaning left → prefer action 0 (push left)
        logits[0] = 0.5f;   // left action logit
        logits[1] = -0.5f;  // right action logit
    }
    
    return logits;
#endif
}
