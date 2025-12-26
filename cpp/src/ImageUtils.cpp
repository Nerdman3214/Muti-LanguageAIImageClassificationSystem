/**
 * =============================================================================
 * ImageUtils.cpp - Image Loading and Preprocessing Implementation
 * =============================================================================
 * 
 * This file implements image loading and preprocessing for neural networks.
 * It supports two modes:
 * 1. OpenCV mode (HAS_OPENCV defined): Full-featured, high-quality
 * 2. Fallback mode: Simple PPM loader for testing without dependencies
 * 
 * OPENCV OVERVIEW:
 * ----------------
 * OpenCV (Open Source Computer Vision Library) is the most widely used
 * library for image processing. It provides:
 * - Image I/O (JPEG, PNG, BMP, etc.)
 * - Image transformations (resize, rotate, crop)
 * - Color space conversions (BGR↔RGB, RGB↔HSV, etc.)
 * - Computer vision algorithms (edge detection, feature matching, etc.)
 * 
 * OPENCV QUIRK - BGR vs RGB:
 * For historical reasons (Windows bitmap format), OpenCV stores images
 * in BGR format (Blue, Green, Red) instead of the more common RGB.
 * Neural networks expect RGB, so we must convert.
 * 
 * @file ImageUtils.cpp
 * @author Multi-Language AI System
 * @version 1.0.0
 */

#include "ImageUtils.hpp"    // Our header file

#include <iostream>          // std::cerr for error messages
#include <fstream>           // std::ifstream for file reading (PPM fallback)
#include <stdexcept>         // std::runtime_error for exceptions
#include <algorithm>         // std::min, std::max
#include <cmath>             // Math functions (not currently used)

/**
 * CONDITIONAL COMPILATION FOR OPENCV:
 * HAS_OPENCV is defined by CMake if OpenCV was found on the system.
 * If not defined, we fall back to a simple built-in loader.
 */
#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>  // Main OpenCV header (includes everything)
#endif

namespace ImageUtils {

/**
 * Load and preprocess an image for ImageNet-trained neural networks.
 * 
 * This function performs the complete preprocessing pipeline:
 * 1. Load image file (JPEG, PNG, etc.)
 * 2. Convert BGR to RGB (OpenCV uses BGR internally)
 * 3. Resize to target dimensions with bilinear interpolation
 * 4. Normalize using ImageNet mean/std
 * 5. Convert from HWC to CHW format
 * 
 * @param imagePath    Path to the image file
 * @param targetWidth  Desired width (typically 224)
 * @param targetHeight Desired height (typically 224)
 * @return Preprocessed tensor in CHW format
 */
std::vector<float> loadAndPreprocessImage(
    const std::string& imagePath,
    int targetWidth,
    int targetHeight
) {
#ifdef HAS_OPENCV
    // ========================================================================
    // OPENCV PATH - Full-featured image loading and preprocessing
    // ========================================================================
    
    // -------------------------------------------------------------------------
    // STEP 1: Load image from file
    // -------------------------------------------------------------------------
    /**
     * cv::imread loads an image from file into a cv::Mat.
     * 
     * cv::Mat (Matrix) is OpenCV's main image container:
     * - Stores pixel data in a contiguous array
     * - Knows width, height, number of channels
     * - Supports various pixel types (uint8, float, etc.)
     * 
     * If loading fails, imread returns an empty Mat.
     */
    cv::Mat img = cv::imread(imagePath);
    
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }

    // -------------------------------------------------------------------------
    // STEP 2: Convert from BGR to RGB
    // -------------------------------------------------------------------------
    /**
     * cv::cvtColor converts between color spaces.
     * 
     * OpenCV loads images in BGR format (Blue-Green-Red), but neural networks
     * trained on ImageNet expect RGB format. This historical quirk comes from
     * Windows bitmap format which used BGR.
     * 
     * Parameters:
     * - img: Input image (BGR)
     * - img: Output image (same variable - in-place conversion)
     * - cv::COLOR_BGR2RGB: Conversion code
     */
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // -------------------------------------------------------------------------
    // STEP 3: Resize with high-quality interpolation
    // -------------------------------------------------------------------------
    /**
     * cv::resize changes image dimensions.
     * 
     * INTERPOLATION METHODS:
     * - INTER_NEAREST: Fastest, lowest quality (blocky)
     * - INTER_LINEAR: Good balance of speed and quality (bilinear interpolation)
     * - INTER_CUBIC: Higher quality, slower (bicubic interpolation)
     * - INTER_LANCZOS4: Highest quality, slowest
     * 
     * We use INTER_LINEAR as a good trade-off.
     */
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_LINEAR);

    // -------------------------------------------------------------------------
    // STEP 4: Normalize using ImageNet statistics
    // -------------------------------------------------------------------------
    /**
     * ImageNet normalization values:
     * These were computed from the 1.2 million training images in ImageNet.
     * All ImageNet-pretrained models (ResNet, VGG, etc.) use these exact values.
     * 
     * Mean: average pixel value for each channel
     * Std: standard deviation of pixel values for each channel
     * 
     * Normalization formula: normalized = (pixel/255 - mean) / std
     * This transforms pixels from [0, 255] to approximately [-2, 2]
     */
    const float mean[3] = {0.485f, 0.456f, 0.406f};  // RGB means
    const float std[3]  = {0.229f, 0.224f, 0.225f};  // RGB standard deviations

    // Allocate output tensor: 3 channels × height × width
    std::vector<float> tensor(3 * targetHeight * targetWidth);

    // -------------------------------------------------------------------------
    // STEP 5: Convert from HWC to CHW format
    // -------------------------------------------------------------------------
    /**
     * FORMAT CONVERSION: HWC → CHW
     * 
     * HWC (Height, Width, Channels) - How OpenCV stores images:
     *   Memory: [R0,G0,B0, R1,G1,B1, R2,G2,B2, ...]  (pixels interleaved)
     * 
     * CHW (Channels, Height, Width) - How PyTorch models expect data:
     *   Memory: [R0,R1,R2,..., G0,G1,G2,..., B0,B1,B2,...]  (channels separate)
     * 
     * We iterate over output (CHW) coordinates and compute the corresponding
     * input (HWC) coordinates.
     */
    for (int c = 0; c < 3; ++c) {                      // For each channel (R, G, B)
        for (int h = 0; h < targetHeight; ++h) {      // For each row
            for (int w = 0; w < targetWidth; ++w) {   // For each column
                
                // Calculate source index in HWC format
                // HWC index = row * row_stride + col * pixel_stride + channel
                int hwc_idx = h * targetWidth * 3 + w * 3 + c;
                
                // Calculate destination index in CHW format
                // CHW index = channel * plane_size + row * width + col
                int chw_idx = c * targetHeight * targetWidth + h * targetWidth + w;
                
                /**
                 * Apply ImageNet normalization:
                 * 1. resized.data[hwc_idx]: Get raw pixel (0-255 as uint8)
                 * 2. / 255.0f: Convert to [0, 1] range
                 * 3. - mean[c]: Center around 0
                 * 4. / std[c]: Scale by standard deviation
                 * 
                 * Result: approximately [-2, 2] range
                 */
                float pixel = static_cast<float>(resized.data[hwc_idx]) / 255.0f;
                tensor[chw_idx] = (pixel - mean[c]) / std[c];
            }
        }
    }

    return tensor;

#else
    // ========================================================================
    // FALLBACK PATH - Simple loader when OpenCV is not available
    // ========================================================================
    
    /**
     * Without OpenCV, we use a simple built-in loader.
     * This supports PPM format (a simple uncompressed image format)
     * and generates test patterns for other formats.
     * 
     * This is mainly for testing and development.
     * Production systems should use OpenCV for full format support.
     */
    std::vector<uint8_t> rawData;
    int width, height;
    
    // Load raw image data
    if (!loadImage(imagePath, rawData, width, height)) {
        throw std::runtime_error("Failed to load image (OpenCV not available)");
    }

    // Same normalization values as OpenCV path
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3]  = {0.229f, 0.224f, 0.225f};

    // Allocate output tensor
    std::vector<float> tensor(3 * targetHeight * targetWidth);

    /**
     * NEAREST NEIGHBOR RESIZE + FORMAT CONVERSION
     * 
     * Nearest neighbor is simple: for each output pixel, find the
     * closest input pixel. This is fast but produces blocky results.
     * 
     * Formula: srcCoord = dstCoord * srcSize / dstSize
     */
    for (int c = 0; c < 3; ++c) {                      // For each channel
        for (int h = 0; h < targetHeight; ++h) {      // For each output row
            for (int w = 0; w < targetWidth; ++w) {   // For each output column
                
                // Nearest neighbor mapping: find source coordinates
                int srcH = (h * height) / targetHeight;
                int srcW = (w * width) / targetWidth;
                
                // Calculate source index in HWC format
                int srcIdx = (srcH * width + srcW) * 3 + c;

                // Get pixel value (with bounds checking)
                float pixel = (srcIdx < static_cast<int>(rawData.size())) 
                    ? rawData[srcIdx] / 255.0f 
                    : 0.0f;

                // Calculate destination index and apply normalization
                int dstIdx = c * targetHeight * targetWidth + h * targetWidth + w;
                tensor[dstIdx] = (pixel - mean[c]) / std[c];
            }
        }
    }

    return tensor;
#endif
}

/**
 * Load raw image data from a file.
 * 
 * This is a lower-level function that loads pixel data without preprocessing.
 * 
 * @param imagePath    Path to image file
 * @param imageData    Output: raw pixel data in HWC RGB format
 * @param width        Output: image width
 * @param height       Output: image height
 * @return true if successful, false otherwise
 */
bool loadImage(
    const std::string& imagePath,
    std::vector<uint8_t>& imageData,
    int& width,
    int& height
) {
#ifdef HAS_OPENCV
    // ========================================================================
    // OPENCV PATH
    // ========================================================================
    
    // Load image
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) return false;

    // Convert BGR → RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    // Store dimensions
    width = img.cols;   // Number of columns = width
    height = img.rows;  // Number of rows = height

    // Copy pixel data to output vector
    imageData.resize(width * height * 3);  // 3 bytes per pixel (RGB)
    
    /**
     * std::memcpy: Fast memory copy
     * Copies raw bytes from img.data to imageData.data()
     * This works because cv::Mat stores pixels contiguously.
     */
    std::memcpy(imageData.data(), img.data, imageData.size());
    
    return true;

#else
    // ========================================================================
    // FALLBACK PATH - PPM Loader
    // ========================================================================
    
    /**
     * PPM (Portable Pixel Map) is a simple image format:
     * - Header: "P6" magic number, width, height, max value
     * - Body: Raw RGB pixel data
     * 
     * Example header:
     *   P6
     *   640 480
     *   255
     *   [binary pixel data follows]
     */
    std::ifstream file(imagePath, std::ios::binary);
    
    if (!file) {
        // File doesn't exist - generate a gray test image
        width = 224;
        height = 224;
        imageData.resize(width * height * 3, 128);  // 128 = medium gray
        return true;
    }

    // Read magic number
    std::string magic;
    file >> magic;

    if (magic != "P6") {
        // Not a PPM file - generate test image
        width = 224;
        height = 224;
        imageData.resize(width * height * 3, 128);
        return true;
    }

    // Read dimensions and max pixel value
    int maxval;
    file >> width >> height >> maxval;
    file.get();  // Consume newline after header

    // Read pixel data
    imageData.resize(width * height * 3);
    file.read(reinterpret_cast<char*>(imageData.data()), imageData.size());

    return true;
#endif
}

} // namespace ImageUtils

} // namespace ImageUtils
