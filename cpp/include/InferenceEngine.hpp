/**
 * =============================================================================
 * InferenceEngine.hpp - Public Interface for the AI Inference Engine
 * =============================================================================
 * 
 * This is a HEADER FILE - it declares the interface (what the class looks like)
 * without providing the implementation (how it works).
 * 
 * HEADER FILES (.h, .hpp) vs SOURCE FILES (.cpp):
 * - Headers: Declarations, included by multiple files
 * - Sources: Implementations, compiled once
 * - Headers should be minimal to reduce compile times
 * 
 * WHY SEPARATE HEADER AND SOURCE?
 * 1. Compilation: C++ compiles each .cpp independently, then links
 * 2. Encapsulation: Users only see the public interface
 * 3. Recompilation: Changing .cpp doesn't require recompiling users
 * 
 * INCLUDE GUARDS (#ifndef / #define / #endif):
 * - Prevent the header from being processed multiple times
 * - Without guards: if A.cpp includes B.hpp and C.hpp, and both
 *   B.hpp and C.hpp include this file, we'd get duplicate definitions
 * - Modern alternative: #pragma once (not standard but widely supported)
 * 
 * @file InferenceEngine.hpp
 * @author Multi-Language AI System
 * @version 1.0.0
 */

#ifndef INFERENCE_ENGINE_HPP   // If INFERENCE_ENGINE_HPP is NOT defined...
#define INFERENCE_ENGINE_HPP   // ...define it (prevents re-inclusion)

// ============================================================================
// STANDARD LIBRARY INCLUDES
// ============================================================================

/**
 * We include only what's needed for the PUBLIC interface.
 * Private implementation details go in the .cpp file.
 * 
 * This is called "minimal includes" - reduces compile dependencies.
 */

#include <string>   // std::string for file paths
#include <vector>   // std::vector for arrays of floats/strings
#include <memory>   // std::unique_ptr for PIMPL pattern

// ============================================================================
// MAIN CLASS DECLARATION
// ============================================================================

/**
 * InferenceEngine - High-level interface for neural network inference.
 * 
 * This class wraps ONNX Runtime to provide:
 * - Image classification using ResNet50/ImageNet
 * - Reinforcement learning policy inference (e.g., CartPole)
 * 
 * DESIGN PATTERN: PIMPL (Pointer to Implementation)
 * -------------------------------------------------
 * Notice there's no ONNX Runtime includes in this header.
 * All ONNX-specific code is hidden in the "Impl" struct (defined in .cpp).
 * 
 * This provides:
 * - Binary stability: Can change implementation without recompiling users
 * - Reduced dependencies: Users don't need ONNX Runtime headers
 * - Faster compilation: Less code to parse in each translation unit
 * 
 * USAGE EXAMPLE:
 * ```cpp
 * InferenceEngine engine;
 * 
 * // For image classification:
 * engine.initialize("models/resnet50.onnx");
 * std::vector<float> probs = engine.classifyImage("cat.jpg");
 * 
 * // For RL policy:
 * engine.initializeRL("models/policy.onnx");
 * std::vector<float> state = {0.0, 0.0, 0.1, -0.2};  // CartPole state
 * std::vector<float> logits = engine.inferState(state);
 * ```
 */
class InferenceEngine {
public:
    // ========================================================================
    // CONSTRUCTORS & DESTRUCTOR
    // ========================================================================
    
    /**
     * Default constructor.
     * Creates an uninitialized engine. Call initialize() or initializeRL()
     * before using other methods.
     */
    InferenceEngine();
    
    /**
     * Destructor.
     * 
     * Declared here, defined in .cpp because unique_ptr<Impl> needs to
     * know the complete type of Impl to destroy it. In this header,
     * Impl is only forward-declared, so the compiler can't generate
     * the destructor here.
     */
    ~InferenceEngine();

    // ========================================================================
    // INITIALIZATION METHODS
    // ========================================================================

    /**
     * Initialize the engine with an image classification model.
     * 
     * This loads an ONNX model designed for image classification.
     * The model should accept input shape [1, 3, 224, 224] (NCHW format)
     * and output class logits/probabilities.
     * 
     * @param modelPath Path to the ONNX model file (e.g., "models/resnet50.onnx")
     * @return true if model loaded successfully, false otherwise
     * 
     * @note Call only once. To change models, create a new InferenceEngine.
     * @note Will attempt to use GPU (CUDA) if available, falls back to CPU.
     */
    bool initialize(const std::string& modelPath);
    
    /**
     * Initialize the engine with a reinforcement learning policy model.
     * 
     * RL models differ from classification models:
     * - Input: State vector [batch, state_dim]
     * - Output: Action logits [batch, num_actions]
     * 
     * @param modelPath Path to the ONNX policy model
     * @return true if model loaded successfully
     * 
     * @note Uses different I/O tensor names than image models:
     *       Input: "state", Output: "action_logits"
     */
    bool initializeRL(const std::string& modelPath);

    // ========================================================================
    // INFERENCE METHODS
    // ========================================================================

    /**
     * Classify an image and return class probabilities.
     * 
     * PREPROCESSING (done internally):
     * 1. Load image from file
     * 2. Resize to model's expected size (typically 224x224)
     * 3. Normalize using ImageNet mean/std
     * 4. Convert to CHW tensor format
     * 
     * @param imagePath Path to the image file (JPEG, PNG, etc.)
     * @return Vector of probabilities, one per class (sums to ~1.0)
     * 
     * @throws std::runtime_error if engine not initialized or image can't be loaded
     * 
     * @example
     * ```cpp
     * auto probs = engine.classifyImage("dog.jpg");
     * int predictedClass = std::max_element(probs.begin(), probs.end()) - probs.begin();
     * std::cout << "Predicted: " << labels[predictedClass] << std::endl;
     * ```
     */
    std::vector<float> classifyImage(const std::string& imagePath);
    
    /**
     * Run policy inference on a state vector.
     * 
     * Used for reinforcement learning to select actions based on environment state.
     * 
     * @param state State vector (e.g., CartPole: [position, velocity, angle, angular_vel])
     * @return Action logits (unnormalized scores). Apply softmax to get probabilities.
     * 
     * @example
     * ```cpp
     * std::vector<float> state = {0.05, 0.02, 0.1, -0.05};  // CartPole state
     * auto logits = engine.inferState(state);
     * // logits[0] = left action score, logits[1] = right action score
     * int action = logits[0] > logits[1] ? 0 : 1;  // Greedy action selection
     * ```
     */
    std::vector<float> inferState(const std::vector<float>& state);

    // ========================================================================
    // ACCESSOR METHODS
    // ========================================================================

    /**
     * Get the number of output classes/actions.
     * @return For ImageNet: 1000. For CartPole: 2.
     */
    int getNumClasses() const;

    /**
     * Get expected input image dimensions.
     * 
     * @param[out] width  Image width (typically 224)
     * @param[out] height Image height (typically 224)
     * @param[out] channels Number of color channels (typically 3 for RGB)
     * 
     * @note For RL models, these values are repurposed:
     *       width = state_dim, height = 1, channels = 1
     */
    void getInputDimensions(int& width, int& height, int& channels) const;

    /**
     * Get the engine version string.
     * @return Version string (e.g., "1.0.0")
     */
    std::string getVersion() const;
    
    /**
     * Get the loaded class labels.
     * 
     * For ImageNet, these are the 1000 class names:
     * "tench", "goldfish", "great white shark", ...
     * 
     * @return Reference to internal labels vector
     */
    const std::vector<std::string>& getLabels() const;
    
    // ========================================================================
    // LEGACY API (for JNI and backward compatibility)
    // ========================================================================
    
    /**
     * Apply softmax function to convert logits to probabilities.
     * 
     * SOFTMAX: P(i) = exp(logit_i) / Σ exp(logit_j)
     * 
     * After softmax:
     * - All outputs are in range (0, 1)
     * - All outputs sum to 1.0
     * - Order is preserved (highest logit → highest probability)
     * 
     * @param logits Raw model outputs (unbounded real numbers)
     * @return Probabilities (positive, sum to 1)
     * 
     * @note Prefer using Softmax::compute() directly. This is kept for JNI compatibility.
     */
    std::vector<float> softmax(const std::vector<float>& logits);
    
    /**
     * Get the predicted class index.
     * 
     * Combines softmax + argmax to find the most likely class.
     * 
     * @param logits Raw model outputs
     * @return Index of the class with highest probability (0 to numClasses-1)
     */
    int predict(const std::vector<float>& logits);

private:
    // ========================================================================
    // PIMPL PATTERN - PRIVATE IMPLEMENTATION
    // ========================================================================
    
    /**
     * Forward declaration of the implementation struct.
     * 
     * FORWARD DECLARATION:
     * "struct Impl;" tells the compiler "Impl is a struct, but I won't tell
     * you what's in it yet."
     * 
     * This is enough for the compiler to know:
     * - We can have a pointer to Impl
     * - We can't access Impl's members (they're unknown here)
     * 
     * The full definition is in InferenceEngine.cpp.
     */
    struct Impl;
    
    /**
     * Smart pointer to the implementation.
     * 
     * std::unique_ptr<Impl> holds a pointer to our Impl struct.
     * When InferenceEngine is destroyed, unique_ptr automatically deletes Impl.
     * 
     * WHY unique_ptr INSTEAD OF raw Impl*?
     * - Automatic memory management (no memory leaks)
     * - Clear ownership semantics (InferenceEngine owns Impl)
     * - Exception-safe (destroyed even if exception thrown)
     */
    std::unique_ptr<Impl> pImpl;
};

// ============================================================================
// MACRO CLEANUP
// ============================================================================

/**
 * Some system headers (especially Windows headers) define macros with common names
 * that conflict with our method names. We undefine them here to be safe.
 * 
 * Example: Windows SDK defines "initialize" in some contexts.
 */
#ifdef initialize
#undef initialize
#endif

#endif // INFERENCE_ENGINE_HPP
