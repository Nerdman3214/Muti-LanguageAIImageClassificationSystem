/**
 * =============================================================================
 * ImageUtils.hpp - Image Loading and Preprocessing Utilities
 * =============================================================================
 * 
 * This file provides utilities for loading images from disk and preprocessing
 * them for neural network inference.
 * 
 * IMAGE PREPROCESSING FOR NEURAL NETWORKS:
 * ----------------------------------------
 * Neural networks expect input in a very specific format:
 * 
 * 1. SIZE: Images must be a fixed size (e.g., 224x224 for ResNet50)
 *    - Most networks are designed for a specific input size
 *    - Images are resized (not cropped) to fit
 * 
 * 2. NORMALIZATION: Pixel values must be normalized
 *    - Raw pixels are 0-255 (uint8)
 *    - Networks expect approximately [-2, 2] (float32)
 *    - We apply: normalized = (pixel/255 - mean) / std
 * 
 * 3. FORMAT: Specific tensor layout
 *    - Images are stored as HWC (Height, Width, Channels)
 *    - PyTorch networks expect NCHW (Batch, Channels, Height, Width)
 *    - We must rearrange the data
 * 
 * IMAGENET NORMALIZATION VALUES:
 * - These were computed from the ImageNet training dataset
 * - Mean: [0.485, 0.456, 0.406] for RGB
 * - Std:  [0.229, 0.224, 0.225] for RGB
 * - All ImageNet-trained models use these values!
 * 
 * COLOR CHANNEL ORDER:
 * - OpenCV loads images as BGR (Blue, Green, Red) - for historical reasons
 * - Neural networks expect RGB (Red, Green, Blue)
 * - We must swap channels during preprocessing
 * 
 * @file ImageUtils.hpp
 * @author Multi-Language AI System
 * @version 1.0.0
 */

#pragma once  // Modern include guard (same effect as #ifndef/#define/#endif)

#include <string>    // std::string for file paths
#include <vector>    // std::vector for dynamic arrays
#include <cstdint>   // uint8_t - unsigned 8-bit integer (0-255, perfect for pixels)

/**
 * Namespace for image processing utilities.
 */
namespace ImageUtils {

/**
 * Load an image and preprocess it for ImageNet models.
 * 
 * This function combines loading and preprocessing into one call:
 * 1. Load image from file (JPEG, PNG, etc.)
 * 2. Resize to targetWidth × targetHeight
 * 3. Convert from BGR to RGB (if OpenCV is used)
 * 4. Normalize using ImageNet mean/std
 * 5. Convert from HWC to CHW format
 * 
 * OUTPUT FORMAT:
 * The returned tensor is in CHW (Channels, Height, Width) format,
 * which is what PyTorch models expect.
 * 
 * Memory layout: [R0, R1, ..., R223*224, G0, G1, ..., B0, B1, ...]
 *                 ↑ All red pixels     ↑ All green   ↑ All blue
 * 
 * @param imagePath    Path to the image file
 * @param targetWidth  Desired output width (typically 224)
 * @param targetHeight Desired output height (typically 224)
 * @return Preprocessed tensor as flat vector of floats
 *         Size: targetWidth × targetHeight × 3
 * 
 * @throws std::runtime_error if image cannot be loaded
 * 
 * @example
 * // Load and preprocess for ResNet50
 * auto tensor = ImageUtils::loadAndPreprocessImage("cat.jpg", 224, 224);
 * // tensor.size() == 224 * 224 * 3 == 150528
 */
std::vector<float> loadAndPreprocessImage(
    const std::string& imagePath, 
    int targetWidth, 
    int targetHeight
);

/**
 * Load raw image data from file.
 * 
 * This is a lower-level function that just loads pixel data without
 * preprocessing. Useful when you want to do custom preprocessing.
 * 
 * OUTPUT FORMAT:
 * Returns HWC (Height, Width, Channels) RGB data as unsigned bytes.
 * Each pixel is 3 consecutive bytes: [R, G, B]
 * 
 * Memory layout: [R0,G0,B0, R1,G1,B1, R2,G2,B2, ...]
 *                 ↑ pixel 0 ↑ pixel 1 ↑ pixel 2
 * 
 * @param imagePath    Path to the image file
 * @param[out] imageData Output vector for pixel data (HWC format, RGB)
 * @param[out] width     Output: image width in pixels
 * @param[out] height    Output: image height in pixels
 * @return true if image was loaded successfully, false otherwise
 * 
 * @note Uses OpenCV if available (HAS_OPENCV defined), otherwise falls
 *       back to a simple PPM/placeholder loader.
 * 
 * @example
 * std::vector<uint8_t> pixels;
 * int w, h;
 * if (ImageUtils::loadImage("photo.jpg", pixels, w, h)) {
 *     // pixels.size() == w * h * 3
 *     // First pixel's red value: pixels[0]
 *     // First pixel's green value: pixels[1]
 *     // First pixel's blue value: pixels[2]
 * }
 */
bool loadImage(
    const std::string& imagePath,
    std::vector<uint8_t>& imageData,
    int& width,
    int& height
);

} // namespace ImageUtils
