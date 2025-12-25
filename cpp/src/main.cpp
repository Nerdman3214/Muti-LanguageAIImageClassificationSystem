#include "InferenceEngine.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];

    try {
        std::cout << "=== Multi-Language AI Image Classification ===" << std::endl;
        std::cout << "Initializing inference engine..." << std::endl;
        
        InferenceEngine engine;

        if (!engine.initialize(modelPath)) {
            std::cerr << "Failed to initialize inference engine" << std::endl;
            return 1;
        }

        std::cout << "\nClassifying image..." << std::endl;
        
        // Measure inference time
        auto start = std::chrono::high_resolution_clock::now();
        auto results = engine.classifyImage(imagePath);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
        std::cout << "Number of classes: " << engine.getNumClasses() << std::endl;

        // Print top-5 predictions
        if (!results.empty()) {
            // Create index vector and sort by probability
            std::vector<size_t> indices(results.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&results](size_t a, size_t b) {
                return results[a] > results[b];
            });

            std::cout << "\nTop-5 Predictions:" << std::endl;
            const auto& labels = engine.getLabels();
            for (int i = 0; i < 5 && i < static_cast<int>(indices.size()); ++i) {
                size_t idx = indices[i];
                std::string label = (idx < labels.size()) ? labels[idx] : "Unknown";
                std::cout << "  " << (i + 1) << ". " << label 
                          << " (" << (results[idx] * 100.0f) << "%)" << std::endl;
            }
        }

        std::cout << "\nEngine version: " << engine.getVersion() << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
