#include "Softmax.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace SoftmaxUtils {

std::vector<float> softmax(const std::vector<float>& scores) {
    if (scores.empty()) {
        return {};
    }

    // Find max for numerical stability
    float maxScore = *std::max_element(scores.begin(), scores.end());

    std::vector<float> probabilities;
    float sum = 0.0f;

    // Calculate exponentials and sum
    for (float score : scores) {
        float exp_val = std::exp(score - maxScore);
        probabilities.push_back(exp_val);
        sum += exp_val;
    }

    // Normalize
    for (auto& prob : probabilities) {
        prob /= sum;
    }

    return probabilities;
}

std::vector<int> topk(const std::vector<float>& scores, int k) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by score values
    std::partial_sort(
        indices.begin(),
        indices.begin() + std::min(k, static_cast<int>(scores.size())),
        indices.end(),
        [&scores](int a, int b) { return scores[a] > scores[b]; }
    );

    indices.resize(std::min(k, static_cast<int>(scores.size())));
    return indices;
}

} // namespace SoftmaxUtils
