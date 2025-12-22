#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <vector>

class InferenceEngine {
public:
    std::vector<float> softmax(const std::vector<float>& logits);
    int predict(const std::vector<float>& logits);
};

#endif
