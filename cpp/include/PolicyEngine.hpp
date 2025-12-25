#ifndef POLICY_ENGINE_HPP
#define POLICY_ENGINE_HPP

#include "InferenceEngine.hpp"
#include <string>
#include <vector>
#include <random>

/**
 * PolicyEngine - Extends InferenceEngine for Reinforcement Learning
 * 
 * This class adapts the general InferenceEngine for RL policy inference.
 * 
 * Key differences from classification:
 * - Input: State vector (not image)
 * - Output: Action probabilities or Q-values
 * - Action selection: Sampling or argmax
 * 
 * Supported RL algorithms:
 * - Policy Gradient (softmax output)
 * - Q-Learning (argmax output)
 * - Actor-Critic (separate value head)
 */
class PolicyEngine {
public:
    PolicyEngine();
    ~PolicyEngine();
    
    /**
     * Initialize with an ONNX policy model
     * @param modelPath Path to .onnx policy file
     * @return true if successful
     */
    bool initialize(const std::string& modelPath);
    
    /**
     * Get action probabilities from state
     * @param state State vector (e.g., [cart_pos, cart_vel, pole_angle, pole_vel])
     * @return Action probabilities (sum to 1.0)
     */
    std::vector<float> getActionProbabilities(const std::vector<float>& state);
    
    /**
     * Sample an action from the policy (for training/exploration)
     * @param state State vector
     * @return Sampled action index
     */
    int sampleAction(const std::vector<float>& state);
    
    /**
     * Get best action (argmax) from policy (for evaluation/deployment)
     * @param state State vector
     * @return Best action index
     */
    int getBestAction(const std::vector<float>& state);
    
    /**
     * Get Q-values for all actions (for Q-learning)
     * @param state State vector
     * @return Q-values for each action
     */
    std::vector<float> getQValues(const std::vector<float>& state);
    
    /**
     * Get number of possible actions
     */
    int getNumActions() const { return numActions; }
    
    /**
     * Get state dimension
     */
    int getStateDim() const { return stateDim; }
    
    /**
     * Set exploration epsilon for epsilon-greedy action selection
     */
    void setEpsilon(float eps) { epsilon = eps; }
    
private:
    InferenceEngine engine;
    int numActions;
    int stateDim;
    float epsilon;  // For epsilon-greedy exploration
    
    std::mt19937 rng;
    std::uniform_real_distribution<float> uniformDist;
};

#endif // POLICY_ENGINE_HPP
