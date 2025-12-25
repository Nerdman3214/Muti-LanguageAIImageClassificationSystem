#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>

/**
 * InferenceEngine - Main C++ class for AI model inference
 * 
 * Supports:
 * - ONNX model loading
 * - ImageNet classification
 * - Top-K predictions
 * - Legacy JNI compatibility
 */
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    /**
     * Initialize the engine with a model
     * @param modelPath Path to the ONNX model file
     * @return true if successful
     */
    bool initialize(const std::string& modelPath);

    /**
     * Classify an image
     * @param imagePath Path to the image file
     * @return Vector of probabilities for each class
     */
    std::vector<float> classifyImage(const std::string& imagePath);

    /**
     * Get number of classes
     * @return Number of output classes (1000 for ImageNet)
     */
    int getNumClasses() const;

    /**
     * Get input image dimensions
     * @param width Output width
     * @param height Output height
     * @param channels Output number of channels
     */
    void getInputDimensions(int& width, int& height, int& channels) const;

    /**
     * Get engine version
     * @return Version string
     */
    std::string getVersion() const;
    
    // ============================================
    // Legacy API (for JNI compatibility)
    // ============================================
    
    /**
     * Apply softmax to logits
     * @param logits Raw model output
     * @return Probabilities
     */
    std::vector<float> softmax(const std::vector<float>& logits);
    
    /**
     * Get predicted class from logits
     * @param logits Raw model output
     * @return Index of highest probability class
     */
    int predict(const std::vector<float>& logits);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

// Undefine any conflicting macros that might be defined by system headers
#ifdef initialize
#undef initialize
#endif

#endif // INFERENCE_ENGINE_HPP
