#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <string>
#include <vector>

/**
 * Image processing utilities
 */
namespace ImageUtils {
    /**
     * Load and preprocess image
     * @param imagePath Path to the image file
     * @param width Target width
     * @param height Target height
     * @return Vector of normalized pixel values
     */
    std::vector<float> loadAndPreprocessImage(const std::string& imagePath, 
                                             int width, int height);

    /**
     * Normalize pixel values to [0, 1]
     * @param pixels Raw pixel data
     * @return Normalized pixel values
     */
    std::vector<float> normalizeImage(const std::vector<uint8_t>& pixels);

    /**
     * Resize image to target dimensions
     * @param inputPath Path to input image
     * @param outputPath Path to save resized image
     * @param width Target width
     * @param height Target height
     * @return true if successful
     */
    bool resizeImage(const std::string& inputPath, const std::string& outputPath,
                    int width, int height);
}

#endif // IMAGE_UTILS_HPP
