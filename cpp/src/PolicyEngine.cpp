#include "PolicyEngine.hpp"
#include "Softmax.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>

PolicyEngine::PolicyEngine() 
    : numActions(0)
    , stateDim(0)
    , epsilon(0.0f)
    , rng(std::random_device{}())
    , uniformDist(0.0f, 1.0f)
{
}

PolicyEngine::~PolicyEngine() = default;

bool PolicyEngine::initialize(const std::string& modelPath) {
    if (!engine.initialize(modelPath)) {
        return false;
    }
    
    // For CartPole: state_dim=4, action_dim=2
    // These could be read from model metadata in a more sophisticated implementation
    stateDim = 4;
    numActions = 2;
    
    std::cout << "PolicyEngine initialized" << std::endl;
    std::cout << "  State dim: " << stateDim << std::endl;
    std::cout << "  Action dim: " << numActions << std::endl;
    
    return true;
}

std::vector<float> PolicyEngine::getActionProbabilities(const std::vector<float>& state) {
    if (state.size() != static_cast<size_t>(stateDim)) {
        throw std::runtime_error("State dimension mismatch");
    }
    
    // Run inference to get logits
    // Note: For RL, we'd typically have a separate inference path
    // For now, we assume the model outputs action logits directly
    
    // Placeholder: In a full implementation, this would call ONNX Runtime
    // with the state vector as input
    std::vector<float> logits(numActions);
    
    // Apply softmax to get probabilities
    return Softmax::compute(logits);
}

int PolicyEngine::sampleAction(const std::vector<float>& state) {
    // Epsilon-greedy exploration
    if (uniformDist(rng) < epsilon) {
        // Random action
        std::uniform_int_distribution<int> actionDist(0, numActions - 1);
        return actionDist(rng);
    }
    
    // Get action probabilities
    std::vector<float> probs = getActionProbabilities(state);
    
    // Sample from categorical distribution
    float r = uniformDist(rng);
    float cumProb = 0.0f;
    
    for (int i = 0; i < numActions; ++i) {
        cumProb += probs[i];
        if (r <= cumProb) {
            return i;
        }
    }
    
    return numActions - 1;  // Fallback
}

int PolicyEngine::getBestAction(const std::vector<float>& state) {
    std::vector<float> probs = getActionProbabilities(state);
    return static_cast<int>(
        std::max_element(probs.begin(), probs.end()) - probs.begin()
    );
}

std::vector<float> PolicyEngine::getQValues(const std::vector<float>& state) {
    // For Q-learning, the model outputs Q-values directly (no softmax)
    if (state.size() != static_cast<size_t>(stateDim)) {
        throw std::runtime_error("State dimension mismatch");
    }
    
    // Placeholder: Would call ONNX Runtime with state as input
    std::vector<float> qValues(numActions, 0.0f);
    
    return qValues;
}
