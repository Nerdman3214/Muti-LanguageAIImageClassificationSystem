#include "RLPolicy.hpp"
#include <algorithm>
#include <random>

RLPolicy::Action RLPolicy::selectAction(
    const std::vector<float>& qValues,
    float epsilon
) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);

    Action action;

    // ε-greedy: explore vs exploit
    if (dis(gen) < epsilon) {
        // Explore: random action
        std::uniform_int_distribution<> actionDis(0, qValues.size() - 1);
        action.actionId = actionDis(gen);
        action.confidence = qValues[action.actionId];
    } else {
        // Exploit: best action
        auto maxIt = std::max_element(qValues.begin(), qValues.end());
        action.actionId = std::distance(qValues.begin(), maxIt);
        action.confidence = *maxIt;
    }

    action.name = getActionName(action.actionId);
    return action;
}

std::string RLPolicy::getActionName(int actionId) {
    // Example mapping for autonomous driving
    // In production, load from config file
    static const std::string actions[] = {
        "steer_left",
        "steer_right", 
        "accelerate",
        "brake",
        "maintain"
    };

    if (actionId >= 0 && actionId < 5) {
        return actions[actionId];
    }
    return "unknown_action_" + std::to_string(actionId);
}