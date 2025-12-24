#pragma once
#include <vector>
#include <string>

/**
 * Reinforcement Learning Policy Interface
 * 
 * Converts classification outputs to RL actions
 * 
 * Classification: image → probabilities
 * RL: state → action
 * 
 * The ONNX model doesn't change - only interpretation
 */
class RLPolicy {
public:
    struct Action {
        int actionId;
        float confidence;
        std::string name;
    };

    /**
     * Select action using ε-greedy strategy
     * 
     * Math:
     *   a = argmax Q(s,a)  with probability 1-ε
     *   a = random         with probability ε
     */
    static Action selectAction(
        const std::vector<float>& qValues,
        float epsilon = 0.1
    );

    /**
     * Map class index to action name
     * For autonomous agents
     */
    static std::string getActionName(int actionId);
};