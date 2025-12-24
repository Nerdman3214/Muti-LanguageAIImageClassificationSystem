#pragma once
#include <string>
#include <vector>
#include <cstdint>  // required for uint8_t

/**
 * Image processing utilities
 */
namespace ImageUtils {

/**
 * Load and preprocess image for ImageNet models
 * Returns CHW tensor (Channels x Height x Width)
 * 
 * ImageNet normalization:
 *   mean = [0.485, 0.456, 0.406] (RGB)
 *   std  = [0.229, 0.224, 0.225]
 * 
 * Math: x' = (x/255 - mean) / std
 */
std::vector<float> loadAndPreprocessImage(
    const std::string& imagePath, 
    int targetWidth, 
    int targetHeight
);

/**
 * Raw image loader (returns HWC format)
 * Used internally by preprocessing pipeline
 */
bool loadImage(
    const std::string& imagePath,
    std::vector<uint8_t>& imageData,
    int& width,
    int& height
);

} // namespace ImageUtils
