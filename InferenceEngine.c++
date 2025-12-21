#include "InferenceEngine.h"

#include <algorithm>    // std::max_element
#include <cmath>        // std::exp
#include <vector>

// Implementations of InferenceEngine methods declared in InferenceEngine.h

std::vector<float> InferenceEngine::softmax(const std::vector<float>& logits) {
    std::vector<float> probs;
    if (logits.empty()) return probs;

    float maxLogit = *std::max_element(logits.begin(), logits.end());

    probs.resize(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        double e = std::exp(static_cast<double>(logits[i] - maxLogit));
        probs[i] = static_cast<float>(e);
        sum += e;
    }

    if (sum == 0.0) return probs;

    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] = static_cast<float>(static_cast<double>(probs[i]) / sum);
    }

    return probs;
}

int InferenceEngine::predict(const std::vector<float>& logits) {
    if (logits.empty()) return -1;
    auto probs = softmax(logits);
    auto maxIt = std::max_element(probs.begin(), probs.end());
    return static_cast<int>(std::distance(probs.begin(), maxIt));
}