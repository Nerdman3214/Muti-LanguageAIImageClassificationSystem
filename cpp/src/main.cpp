#include "InferenceEngine.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];

    try {
        InferenceEngine engine;

        if (!engine.initialize(modelPath)) {
            std::cerr << "Failed to initialize inference engine" << std::endl;
            return 1;
        }

        auto results = engine.classifyImage(imagePath);

        std::cout << "Classification results for: " << imagePath << std::endl;
        std::cout << "Number of classes: " << engine.getNumClasses() << std::endl;
        std::cout << "Engine version: " << engine.getVersion() << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
