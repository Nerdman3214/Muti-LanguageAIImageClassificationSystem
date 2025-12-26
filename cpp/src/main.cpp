/**
 * =============================================================================
 * main.cpp - Command-Line Interface for the Inference Engine
 * =============================================================================
 * 
 * This file provides a standalone command-line interface to the InferenceEngine.
 * It's useful for:
 * - Testing the C++ inference pipeline independently
 * - Benchmarking inference speed
 * - Quick image classification without starting the full Java server
 * 
 * USAGE:
 *   ./InferenceEngine <model_path> <image_path>
 * 
 * EXAMPLE:
 *   ./InferenceEngine models/resnet50_imagenet.onnx test_images/cat.jpg
 * 
 * OUTPUT:
 *   - Inference time in milliseconds
 *   - Top-5 predictions with class names and confidence percentages
 * 
 * @file main.cpp
 * @author Multi-Language AI System
 * @version 1.0.0
 */

// ============================================================================
// INCLUDES
// ============================================================================

#include "InferenceEngine.hpp"  // Our inference engine class

#include <iostream>    // std::cout, std::cerr for console I/O
#include <chrono>      // High-resolution timing for benchmarks
#include <algorithm>   // std::sort for sorting predictions
#include <numeric>     // std::iota for generating index sequences

/**
 * Main entry point for the command-line application.
 * 
 * COMMAND-LINE ARGUMENTS IN C++:
 * - argc: "argument count" - number of arguments including program name
 * - argv: "argument vector" - array of C-strings
 * 
 * Example: ./program arg1 arg2
 *   argc = 3
 *   argv[0] = "./program"
 *   argv[1] = "arg1"
 *   argv[2] = "arg2"
 * 
 * @param argc Number of command-line arguments
 * @param argv Array of argument strings
 * @return 0 on success, non-zero on error
 */
int main(int argc, char* argv[]) {
    // ========================================================================
    // ARGUMENT VALIDATION
    // ========================================================================
    
    /**
     * Check that we have enough arguments.
     * We need at least 3: program name, model path, image path
     */
    if (argc < 3) {
        // Print usage message to stderr (standard error stream)
        // argv[0] is the program name as invoked
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;  // Non-zero return indicates error
    }

    // Extract arguments into more readable variables
    std::string modelPath = argv[1];  // Path to .onnx model file
    std::string imagePath = argv[2];  // Path to image file

    // ========================================================================
    // MAIN PROCESSING (wrapped in try-catch for error handling)
    // ========================================================================
    
    /**
     * try-catch for exception handling:
     * If any code inside 'try' throws an exception, execution jumps to 'catch'.
     * This prevents crashes and allows graceful error reporting.
     */
    try {
        // Print banner
        std::cout << "=== Multi-Language AI Image Classification ===" << std::endl;
        std::cout << "Initializing inference engine..." << std::endl;
        
        // ====================================================================
        // ENGINE INITIALIZATION
        // ====================================================================
        
        /**
         * Create an InferenceEngine instance.
         * This allocates memory and sets up the PIMPL implementation,
         * but doesn't load a model yet.
         */
        InferenceEngine engine;

        /**
         * Initialize with the specified model.
         * This:
         * 1. Loads the ONNX model file
         * 2. Creates an ONNX Runtime session
         * 3. Enables GPU (CUDA) if available
         * 4. Loads class labels from accompanying file
         */
        if (!engine.initialize(modelPath)) {
            std::cerr << "Failed to initialize inference engine" << std::endl;
            return 1;  // Exit with error
        }

        std::cout << "\nClassifying image..." << std::endl;
        
        // ====================================================================
        // INFERENCE WITH TIMING
        // ====================================================================
        
        /**
         * High-resolution timing using std::chrono.
         * 
         * std::chrono is C++11's timing library:
         * - high_resolution_clock: Most precise clock available
         * - time_point: A moment in time
         * - duration: Difference between two time_points
         */
        auto start = std::chrono::high_resolution_clock::now();
        
        /**
         * Run classification!
         * This:
         * 1. Loads the image from disk
         * 2. Preprocesses (resize, normalize, convert to tensor)
         * 3. Runs neural network forward pass
         * 4. Returns probability distribution over all 1000 classes
         */
        auto results = engine.classifyImage(imagePath);
        
        auto end = std::chrono::high_resolution_clock::now();
        
        /**
         * Calculate elapsed time.
         * duration_cast converts the raw duration to milliseconds.
         */
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // ====================================================================
        // DISPLAY RESULTS
        // ====================================================================
        
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
        std::cout << "Number of classes: " << engine.getNumClasses() << std::endl;

        /**
         * Find and display top-5 predictions.
         * This is similar to how the Java API presents results.
         */
        if (!results.empty()) {
            // Create index vector: [0, 1, 2, ..., n-1]
            // We'll sort this by corresponding probability
            std::vector<size_t> indices(results.size());
            
            /**
             * std::iota fills a range with sequentially increasing values.
             * After this: indices = [0, 1, 2, 3, ..., 999]
             * 
             * "iota" comes from the Greek letter ι, used in APL programming
             * language to generate integer sequences.
             */
            std::iota(indices.begin(), indices.end(), 0);
            
            /**
             * Sort indices by their corresponding probability (descending).
             * 
             * Lambda function [&results](size_t a, size_t b):
             * - [&results]: Capture 'results' by reference
             * - (size_t a, size_t b): Two indices to compare
             * - returns true if results[a] > results[b] (descending order)
             */
            std::sort(indices.begin(), indices.end(), [&results](size_t a, size_t b) {
                return results[a] > results[b];
            });

            std::cout << "\nTop-5 Predictions:" << std::endl;
            
            // Get labels for readable output
            const auto& labels = engine.getLabels();
            
            // Display top 5 predictions
            for (int i = 0; i < 5 && i < static_cast<int>(indices.size()); ++i) {
                size_t idx = indices[i];
                
                // Get label if available, otherwise show "Unknown"
                std::string label = (idx < labels.size()) ? labels[idx] : "Unknown";
                
                // Display with percentage (probability * 100)
                std::cout << "  " << (i + 1) << ". " << label 
                          << " (" << (results[idx] * 100.0f) << "%)" << std::endl;
            }
        }

        // Show engine version for debugging
        std::cout << "\nEngine version: " << engine.getVersion() << std::endl;

        return 0;  // Success!
        
    } catch (const std::exception& e) {
        // ====================================================================
        // ERROR HANDLING
        // ====================================================================
        
        /**
         * Catch any std::exception (and derived classes).
         * e.what() returns the error message.
         */
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;  // Return non-zero to indicate failure
    }
    
    /**
     * Note: If a non-std::exception is thrown (rare), it won't be caught
     * and the program will terminate. In production, you might add:
     *   catch (...) { std::cerr << "Unknown error"; return 1; }
     */
}
