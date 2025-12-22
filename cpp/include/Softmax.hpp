#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <vector>

/**
 * Softmax utilities for neural network outputs
 */
namespace SoftmaxUtils {
    /**
     * Apply softmax normalization to raw scores
     * @param scores Raw output scores
     * @return Probabilities after softmax normalization
     */
    std::vector<float> softmax(const std::vector<float>& scores);

    /**
     * Get top-k predictions
     * @param scores Output probabilities
     * @param k Number of top predictions
     * @return Vector of indices of top-k predictions
     */
    std::vector<int> topk(const std::vector<float>& scores, int k);
}

#endif // SOFTMAX_HPP
