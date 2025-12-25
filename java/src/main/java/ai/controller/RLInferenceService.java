package ai.controller;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * RLInferenceService - Business logic for RL policy inference
 * 
 * This service handles:
 * - State → Action policy inference via ONNX
 * - Q-value queries
 * - Action sampling (exploration)
 * 
 * Design: Same pattern as InferenceService, but for RL
 * Pipeline: Java → JNI → C++ ONNX Runtime → cartpole_policy.onnx
 */
public class RLInferenceService {
    
    private static final Gson gson = new GsonBuilder().setPrettyPrinting().create();
    private static RLInferenceService instance;
    
    // JNI bridge reference
    private AIController controller;
    private boolean nativeAvailable = false;
    
    // Policy info
    private String policyModel = "cartpole_policy.onnx";
    private int stateDim = 4;
    private int actionDim = 2;
    
    // Action labels for CartPole
    private static final String[] ACTION_LABELS = {"push_left", "push_right"};
    
    public static synchronized RLInferenceService getInstance() {
        if (instance == null) {
            instance = new RLInferenceService();
        }
        return instance;
    }
    
    private RLInferenceService() {
        // Try to access JNI
        try {
            controller = new AIController();
            nativeAvailable = true;
            System.out.println("✅ RLInferenceService initialized with ONNX backend");
        } catch (Exception e) {
            nativeAvailable = false;
            System.out.println("⚠️ RLInferenceService initialized (simulation mode)");
        }
    }
    
    /**
     * Get action from policy given state
     * 
     * @param state State vector [cart_pos, cart_vel, pole_angle, pole_vel]
     * @return Policy result with action and probabilities
     */
    public PolicyResult getAction(float[] state) {
        long startTime = System.currentTimeMillis();
        
        PolicyResult result = new PolicyResult();
        result.model = policyModel;
        result.stateDim = stateDim;
        result.actionDim = actionDim;
        
        if (state == null || state.length != stateDim) {
            result.success = false;
            result.error = "Invalid state dimension. Expected " + stateDim + " values.";
            return result;
        }
        
        // Store input state
        result.inputState = state;
        
        try {
            float[] actionLogits;
            
            if (nativeAvailable && controller != null) {
                // Real ONNX inference via JNI
                actionLogits = controller.nativeInferState(state);
                result.simulated = false;
            } else {
                // Fallback simulation
                actionLogits = simulatePolicyLogits(state);
                result.simulated = true;
            }
            
            // Convert logits to probabilities via softmax
            result.actionProbabilities = softmax(actionLogits);
            result.selectedAction = argmax(result.actionProbabilities);
            result.actionLabel = ACTION_LABELS[result.selectedAction];
            result.confidence = result.actionProbabilities[result.selectedAction];
            result.success = true;
            
        } catch (Exception e) {
            // Fallback to simulation on error
            float[] actionLogits = simulatePolicyLogits(state);
            result.actionProbabilities = softmax(actionLogits);
            result.selectedAction = argmax(result.actionProbabilities);
            result.actionLabel = ACTION_LABELS[result.selectedAction];
            result.confidence = result.actionProbabilities[result.selectedAction];
            result.success = true;
            result.simulated = true;
            result.error = "Native inference failed: " + e.getMessage();
        }
        
        result.latencyMs = System.currentTimeMillis() - startTime;
        return result;
    }
    
    /**
     * Softmax function to convert logits to probabilities
     */
    private float[] softmax(float[] logits) {
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            maxLogit = Math.max(maxLogit, logit);
        }
        
        float sumExp = 0.0f;
        float[] probs = new float[logits.length];
        
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - maxLogit);
            sumExp += probs[i];
        }
        
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sumExp;
        }
        
        return probs;
    }
    
    /**
     * Simulate logits (for fallback)
     */
    private float[] simulatePolicyLogits(float[] state) {
        float poleAngle = state[2];
        float poleVel = state[3];
        
        // Simple heuristic returning logits
        if (poleAngle + 0.1f * poleVel > 0) {
            return new float[]{-0.5f, 0.5f};  // prefer right
        } else {
            return new float[]{0.5f, -0.5f};  // prefer left
        }
    }
    
    /**
     * Get Q-values for all actions (for Q-learning style queries)
     */
    public QValueResult getQValues(float[] state) {
        QValueResult result = new QValueResult();
        result.model = policyModel + "_qvalues";
        
        if (state == null || state.length != stateDim) {
            result.success = false;
            result.error = "Invalid state dimension";
            return result;
        }
        
        // For REINFORCE policy, we return logits as pseudo-Q-values
        try {
            float[] logits;
            
            if (nativeAvailable && controller != null) {
                logits = controller.nativeInferState(state);
            } else {
                logits = simulatePolicyLogits(state);
            }
            
            result.qValues = logits;
            result.bestAction = argmax(logits);
            result.actionLabel = ACTION_LABELS[result.bestAction];
            result.success = true;
            
        } catch (Exception e) {
            // Fallback
            float poleAngle = state[2];
            result.qValues = new float[]{
                -Math.abs(poleAngle) - 0.1f,  // Q(s, left)
                -Math.abs(poleAngle) + 0.1f   // Q(s, right)
            };
            result.bestAction = argmax(result.qValues);
            result.actionLabel = ACTION_LABELS[result.bestAction];
            result.success = true;
        }
        
        return result;
    }
    
    private int argmax(float[] arr) {
        int maxIdx = 0;
        float maxVal = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    public boolean isPolicyLoaded() {
        return nativeAvailable;
    }
    
    public String[] getActionLabels() {
        return ACTION_LABELS;
    }
    
    // ========================================
    // Result classes
    // ========================================
    
    public static class PolicyResult {
        public boolean success;
        public boolean simulated;
        public String model;
        public int stateDim;
        public int actionDim;
        public float[] inputState;
        public float[] actionProbabilities;
        public int selectedAction;
        public String actionLabel;
        public float confidence;
        public long latencyMs;
        public String error;
    }
    
    public static class QValueResult {
        public boolean success;
        public String model;
        public float[] qValues;
        public int bestAction;
        public String actionLabel;
        public String error;
    }
}
