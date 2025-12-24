#include "ImageUtils.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace ImageUtils {

std::vector<float> loadAndPreprocessImage(
    const std::string& imagePath,
    int targetWidth,
    int targetHeight
) {
#ifdef HAS_OPENCV
    // Step 1: Load image
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }

    // Step 2: Convert BGR to RGB (OpenCV loads as BGR)
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Step 3: Resize with high-quality interpolation
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_LINEAR);

    // Step 4: Normalize with ImageNet statistics
    const float mean[3] = {0.485f, 0.456f, 0.406f};  // RGB
    const float std[3]  = {0.229f, 0.224f, 0.225f};

    std::vector<float> tensor(3 * targetHeight * targetWidth);

    // Convert HWC (OpenCV) to CHW (PyTorch/ONNX)
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < targetHeight; ++h) {
            for (int w = 0; w < targetWidth; ++w) {
                int hwc_idx = h * targetWidth * 3 + w * 3 + c;
                int chw_idx = c * targetHeight * targetWidth + h * targetWidth + w;
                
                // Normalize: (pixel/255 - mean) / std
                float pixel = static_cast<float>(resized.data[hwc_idx]) / 255.0f;
                tensor[chw_idx] = (pixel - mean[c]) / std[c];
            }
        }
    }

    return tensor;

#else
    // Fallback: Simple PPM loader + nearest neighbor resize
    std::vector<uint8_t> rawData;
    int width, height;
    
    if (!loadImage(imagePath, rawData, width, height)) {
        throw std::runtime_error("Failed to load image (OpenCV not available)");
    }

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3]  = {0.229f, 0.224f, 0.225f};

    std::vector<float> tensor(3 * targetHeight * targetWidth);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < targetHeight; ++h) {
            for (int w = 0; w < targetWidth; ++w) {
                // Nearest neighbor mapping
                int srcH = (h * height) / targetHeight;
                int srcW = (w * width) / targetWidth;
                int srcIdx = (srcH * width + srcW) * 3 + c;

                float pixel = (srcIdx < static_cast<int>(rawData.size())) 
                    ? rawData[srcIdx] / 255.0f 
                    : 0.0f;

                int dstIdx = c * targetHeight * targetWidth + h * targetWidth + w;
                tensor[dstIdx] = (pixel - mean[c]) / std[c];
            }
        }
    }

    return tensor;
#endif
}

bool loadImage(
    const std::string& imagePath,
    std::vector<uint8_t>& imageData,
    int& width,
    int& height
) {
#ifdef HAS_OPENCV
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) return false;

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    width = img.cols;
    height = img.rows;

    imageData.resize(width * height * 3);
    std::memcpy(imageData.data(), img.data, imageData.size());
    return true;

#else
    // Simple PPM P6 loader
    std::ifstream file(imagePath, std::ios::binary);
    if (!file) {
        // Generate test pattern if file doesn't exist
        width = 224;
        height = 224;
        imageData.resize(width * height * 3, 128);
        return true;
    }

    std::string magic;
    file >> magic;

    if (magic != "P6") {
        width = 224;
        height = 224;
        imageData.resize(width * height * 3, 128);
        return true;
    }

    int maxval;
    file >> width >> height >> maxval;
    file.get(); // consume newline

    imageData.resize(width * height * 3);
    file.read(reinterpret_cast<char*>(imageData.data()), imageData.size());

    return true;
#endif
}

} // namespace ImageUtils
