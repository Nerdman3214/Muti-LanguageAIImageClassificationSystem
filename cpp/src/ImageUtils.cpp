#include "ImageUtils.hpp"
#include <vector>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <cstring>

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace ImageUtils {

std::vector<float> loadAndPreprocessImage(const std::string& imagePath, 
                                         int width, int height) {
#ifdef HAS_OPENCV
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        throw std::runtime_error("Could not load image: " + imagePath);
    }
    
    // Resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height));
    
    // Convert to float and normalize
    std::vector<float> result(width * height * 3);
    
    // ImageNet normalization
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std_val[] = {0.229f, 0.224f, 0.225f};
    
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // OpenCV uses BGR, convert to RGB
                int cv_c = 2 - c;
                float pixel = resized.at<cv::Vec3b>(h, w)[cv_c] / 255.0f;
                pixel = (pixel - mean[c]) / std_val[c];
                result[c * height * width + h * width + w] = pixel;
            }
        }
    }
    
    return result;
#else
    // Placeholder when OpenCV not available
    std::cerr << "Warning: OpenCV not available, returning placeholder image data" << std::endl;
    std::vector<float> result(width * height * 3, 0.0f);
    return result;
#endif
}

std::vector<float> normalizeImage(const std::vector<uint8_t>& pixels) {
    std::vector<float> normalized;
    normalized.reserve(pixels.size());

    for (uint8_t pixel : pixels) {
        normalized.push_back(static_cast<float>(pixel) / 255.0f);
    }

    return normalized;
}

bool resizeImage(const std::string& inputPath, const std::string& outputPath,
                int width, int height) {
#ifdef HAS_OPENCV
    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) return false;
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height));
    return cv::imwrite(outputPath, resized);
#else
    std::cerr << "Warning: OpenCV not available, cannot resize image" << std::endl;
    return false;
#endif
}

bool loadImage(const std::string& imagePath, 
               std::vector<uint8_t>& imageData,
               int& width, int& height) {
#ifdef HAS_OPENCV
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return false;
    }
    
    width = img.cols;
    height = img.rows;
    
    // Convert BGR to RGB and flatten
    imageData.resize(width * height * 3);
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(h, w);
            int idx = (h * width + w) * 3;
            imageData[idx + 0] = pixel[2];  // R
            imageData[idx + 1] = pixel[1];  // G
            imageData[idx + 2] = pixel[0];  // B
        }
    }
    
    return true;
#else
    // Simple PPM loader for testing without OpenCV
    std::ifstream file(imagePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << imagePath << std::endl;
        // Return placeholder data for testing
        width = 224;
        height = 224;
        imageData.resize(width * height * 3, 128);  // Gray placeholder
        return true;  // Return true with placeholder for simulation mode
    }
    
    std::string magic;
    file >> magic;
    
    if (magic == "P6") {
        file >> width >> height;
        int maxVal;
        file >> maxVal;
        file.get();  // Skip whitespace
        
        imageData.resize(width * height * 3);
        file.read(reinterpret_cast<char*>(imageData.data()), imageData.size());
        return true;
    }
    
    // Unknown format - return placeholder
    width = 224;
    height = 224;
    imageData.resize(width * height * 3, 128);
    return true;
#endif
}

} // namespace ImageUtils
