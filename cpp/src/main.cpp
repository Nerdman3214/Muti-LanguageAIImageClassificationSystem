#include "InferenceEngine.hpp"
#include <iostream>
#include <chrono>

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
        std::cout << "Engine version: " << engine.getVersion() << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
