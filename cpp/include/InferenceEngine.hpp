#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>

/**
 * InferenceEngine - Main C++ class for AI model inference
 */
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    /**
     * Initialize the engine with a model
     * @param modelPath Path to the model file
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
     * @return Number of output classes
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

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // INFERENCE_ENGINE_HPP
