/**
 * =============================================================================
 * Softmax.hpp - Neural Network Output Normalization Utilities
 * =============================================================================
 * 
 * This file provides functions to convert neural network outputs (logits)
 * into probabilities using the softmax function.
 * 
 * SOFTMAX FUNCTION EXPLAINED:
 * ---------------------------
 * Neural networks output "logits" - raw, unbounded scores.
 * Example: A model might output [2.5, -1.0, 0.5] for 3 classes.
 * 
 * We need to convert these to probabilities that:
 * 1. Are all positive (can't have negative probability!)
 * 2. Sum to 1.0 (total probability must be 100%)
 * 3. Preserve the relative ordering (highest logit → highest probability)
 * 
 * Softmax achieves this:
 * 
 *   P(class_i) = exp(logit_i) / Σ exp(logit_j)
 * 
 * Properties:
 * - exp() makes everything positive
 * - Division by sum normalizes to 1.0
 * - Larger differences in logits → more confident predictions
 * 
 * NUMERICAL EXAMPLE:
 * Logits: [2.5, -1.0, 0.5]
 * After exp: [12.18, 0.37, 1.65]  (all positive now!)
 * Sum: 14.2
 * After normalization: [0.86, 0.026, 0.116]  (sums to 1.0)
 * 
 * @file Softmax.hpp
 * @author Multi-Language AI System
 * @version 1.0.0
 */

#ifndef SOFTMAX_HPP    // Include guard - prevents double inclusion
#define SOFTMAX_HPP

#include <vector>      // std::vector for dynamic arrays

/**
 * Primary namespace for softmax utilities.
 * 
 * NAMESPACE:
 * A namespace groups related functions/classes to prevent name conflicts.
 * Without namespaces, two libraries might both define "softmax()" and collide.
 */
namespace SoftmaxUtils {
    
    /**
     * Apply softmax normalization to convert logits to probabilities.
     * 
     * ALGORITHM:
     * 1. Find max(logits) for numerical stability
     * 2. Compute exp(logit - max) for each logit
     * 3. Divide by sum of exponentials
     * 
     * NUMERICAL STABILITY:
     * Without subtracting max, exp(1000) would overflow to infinity.
     * By subtracting max, the largest exponential is exp(0) = 1.
     * This trick doesn't change the result but prevents overflow.
     * 
     * @param scores Input logits (raw neural network outputs)
     * @return Probabilities (all positive, sum to 1.0)
     * 
     * @example
     * std::vector<float> logits = {2.5f, -1.0f, 0.5f};
     * auto probs = SoftmaxUtils::softmax(logits);
     * // probs ≈ [0.86, 0.026, 0.116]
     */
    std::vector<float> softmax(const std::vector<float>& scores);

    /**
     * Get indices of top-k highest scoring elements.
     * 
     * Used to find the k most likely classes after classification.
     * For ImageNet, we often want "Top-5" accuracy (correct if true
     * class is in top 5 predictions).
     * 
     * ALGORITHM:
     * Uses std::partial_sort which is O(n log k) - more efficient than
     * full sort O(n log n) when k << n.
     * 
     * @param scores Array of scores (typically probabilities)
     * @param k Number of top indices to return
     * @return Vector of indices, sorted by score (highest first)
     * 
     * @example
     * std::vector<float> probs = {0.1, 0.3, 0.05, 0.5, 0.05};
     * auto top3 = SoftmaxUtils::topk(probs, 3);
     * // top3 = [3, 1, 0]  (indices with probs 0.5, 0.3, 0.1)
     */
    std::vector<int> topk(const std::vector<float>& scores, int k);
}

/**
 * Convenience namespace with simpler function names.
 * 
 * Aliases SoftmaxUtils functions for cleaner calling code.
 * Example: Softmax::compute() instead of SoftmaxUtils::softmax()
 * 
 * INLINE FUNCTIONS:
 * 'inline' suggests the compiler replace the function call with the
 * function body at each call site. For trivial wrapper functions like
 * these, this eliminates function call overhead.
 */
namespace Softmax {
    /**
     * Compute softmax probabilities from logits.
     * Alias for SoftmaxUtils::softmax().
     */
    inline std::vector<float> compute(const std::vector<float>& logits) {
        return SoftmaxUtils::softmax(logits);
    }
    
    /**
     * Get top-K prediction indices.
     * Alias for SoftmaxUtils::topk().
     */
    inline std::vector<int> topK(const std::vector<float>& scores, int k) {
        return SoftmaxUtils::topk(scores, k);
    }
}

#endif // SOFTMAX_HPP
