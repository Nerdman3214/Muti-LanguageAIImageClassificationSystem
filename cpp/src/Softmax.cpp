/**
 * =============================================================================
 * Softmax.cpp - Implementation of Softmax and Top-K Functions
 * =============================================================================
 * 
 * This file implements the softmax function and top-k selection algorithm.
 * These are fundamental operations in neural network inference.
 * 
 * @file Softmax.cpp
 * @author Multi-Language AI System
 * @version 1.0.0
 */

#include "Softmax.hpp"   // Our header file with declarations

#include <cmath>         // std::exp() - exponential function
#include <algorithm>     // std::max_element, std::partial_sort
#include <numeric>       // std::iota - fills vector with sequential values

/**
 * Implementation namespace.
 * Matches the namespace declared in Softmax.hpp.
 */
namespace SoftmaxUtils {

/**
 * Softmax implementation with numerical stability.
 * 
 * SOFTMAX FORMULA:
 *   softmax(x_i) = exp(x_i) / Σ exp(x_j)
 * 
 * NUMERICAL STABILITY TRICK:
 * Computing exp(1000) overflows to infinity.
 * Solution: Subtract max before exponentiating.
 * 
 *   softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)
 * 
 * This is mathematically equivalent (the max cancels out in the division)
 * but prevents overflow because exp(0) = 1 is the largest value.
 * 
 * PROOF OF EQUIVALENCE:
 *   exp(x_i - max) / Σ exp(x_j - max)
 * = exp(x_i) * exp(-max) / Σ (exp(x_j) * exp(-max))
 * = exp(x_i) * exp(-max) / (exp(-max) * Σ exp(x_j))
 * = exp(x_i) / Σ exp(x_j)  ← original formula!
 */
std::vector<float> softmax(const std::vector<float>& scores) {
    // Handle empty input gracefully
    if (scores.empty()) {
        return {};  // Return empty vector
    }

    // ========================================================================
    // STEP 1: Find maximum value for numerical stability
    // ========================================================================
    
    /**
     * std::max_element returns an iterator to the largest element.
     * The * operator dereferences the iterator to get the actual value.
     */
    float maxScore = *std::max_element(scores.begin(), scores.end());

    // ========================================================================
    // STEP 2: Compute exponentials and running sum
    // ========================================================================
    
    std::vector<float> probabilities;  // Will store exp(score - max)
    float sum = 0.0f;                  // Running sum for normalization

    /**
     * Range-based for loop: Iterates over each element in scores.
     * 'float score' is a copy of each element (not a reference).
     * For small types like float, copying is fine (no reference needed).
     */
    for (float score : scores) {
        /**
         * std::exp computes e^x where e ≈ 2.71828.
         * We subtract maxScore to prevent overflow.
         * If score == maxScore, exp_val = exp(0) = 1 (maximum possible).
         */
        float exp_val = std::exp(score - maxScore);
        
        // Add to result vector and accumulate sum
        probabilities.push_back(exp_val);
        sum += exp_val;
    }

    // ========================================================================
    // STEP 3: Normalize by sum so probabilities sum to 1.0
    // ========================================================================
    
    /**
     * Range-based for loop with reference ('&').
     * This modifies the actual elements in the vector.
     * Without '&', we'd modify copies (useless).
     */
    for (auto& prob : probabilities) {
        prob /= sum;  // Divide by sum: prob = prob / sum
    }

    return probabilities;
}

/**
 * Find the indices of the k largest elements.
 * 
 * ALGORITHM: Partial Sort
 * - Create index array [0, 1, 2, ..., n-1]
 * - Partially sort to put k largest at the front
 * - Return those k indices
 * 
 * TIME COMPLEXITY:
 * - std::partial_sort: O(n log k)
 * - This is better than full sort O(n log n) when k << n
 * - For top-5 of 1000 classes, this is ~5x faster
 * 
 * @param scores Vector of scores (higher = better)
 * @param k Number of top indices to find
 * @return Indices of k highest-scoring elements (sorted by score, descending)
 */
std::vector<int> topk(const std::vector<float>& scores, int k) {
    // ========================================================================
    // STEP 1: Create index array [0, 1, 2, ..., n-1]
    // ========================================================================
    
    /**
     * Create vector of size scores.size().
     * std::iota fills it with sequential values starting from 0.
     * 
     * Example: If scores has 5 elements:
     *   indices = [0, 1, 2, 3, 4]
     */
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);  // Fill with 0, 1, 2, ...

    // ========================================================================
    // STEP 2: Partial sort to move top-k indices to front
    // ========================================================================
    
    /**
     * std::partial_sort rearranges elements so that:
     * - First k elements are the k smallest (or largest with custom comparator)
     * - First k elements are sorted
     * - Remaining elements are in unspecified order
     * 
     * Parameters:
     * - indices.begin(): Start of range
     * - indices.begin() + min(k, size): Middle point (where sorted part ends)
     * - indices.end(): End of range
     * - Lambda comparator: Custom comparison function
     */
    std::partial_sort(
        indices.begin(),
        indices.begin() + std::min(k, static_cast<int>(scores.size())),
        indices.end(),
        /**
         * Lambda function (anonymous function) for comparison.
         * 
         * [&scores]: Capture scores by reference (so we can access it)
         * (int a, int b): Parameters - two indices to compare
         * { return scores[a] > scores[b]; }: Body - compare scores at those indices
         * 
         * Returns true if scores[a] > scores[b], meaning a should come first.
         * This gives us descending order (highest scores first).
         */
        [&scores](int a, int b) { return scores[a] > scores[b]; }
    );

    // ========================================================================
    // STEP 3: Resize to keep only top-k
    // ========================================================================
    
    /**
     * Resize shrinks the vector to keep only the first k elements.
     * std::min handles the case where k > scores.size().
     */
    indices.resize(std::min(k, static_cast<int>(scores.size())));
    
    return indices;
}

} // namespace SoftmaxUtils
