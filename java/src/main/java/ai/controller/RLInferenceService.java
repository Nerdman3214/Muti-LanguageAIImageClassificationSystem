/**
 * ============================================================================
 * RLInferenceService.java - Reinforcement Learning Policy Inference Service
 * ============================================================================
 * 
 * This file handles business logic for RL (Reinforcement Learning) policy inference.
 * It follows the Service Layer pattern, separating business logic from the REST controller.
 * 
 * REINFORCEMENT LEARNING OVERVIEW:
 * --------------------------------
 * RL is a type of machine learning where an agent learns to make decisions by
 * interacting with an environment:
 * 
 *   1. Agent observes the current STATE of the environment
 *   2. Agent chooses an ACTION based on its POLICY
 *   3. Environment transitions to a new state and gives a REWARD
 *   4. Agent learns to maximize total reward over time
 * 
 * CARTPOLE PROBLEM:
 * -----------------
 * CartPole is a classic RL benchmark:
 * - A pole is attached to a cart that moves on a track
 * - Goal: Keep the pole balanced upright by moving the cart
 * - State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
 * - Actions: Push cart left (0) or right (1)
 * - Reward: +1 for each timestep the pole stays upright
 * - Episode ends when pole falls or cart goes off track
 * 
 * DESIGN PATTERNS USED:
 * - Singleton: Only one instance of this service exists
 * - Service Layer: Encapsulates business logic away from controllers
 * - Graceful Degradation: Falls back to simulation if native engine unavailable
 * 
 * @author Multi-Language AI System
 * @version 1.0.0
 */

// Package declaration - this class belongs to ai.controller package
package ai.controller;

// Gson for JSON serialization (used for debug/logging)
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * Service class for RL policy inference.
 * 
 * This service:
 * 1. Receives state vectors from the REST controller
 * 2. Calls the native C++ engine via JNI to run the ONNX policy model
 * 3. Converts raw logits to action probabilities
 * 4. Returns structured results to the controller
 */
public class RLInferenceService {
    
    // ========================================================================
    // STATIC FIELDS
    // ========================================================================
    
    /**
     * Gson instance for JSON operations.
     * Not currently used but available for debugging/logging.
     * 
     * WHY UNUSED? Java compiler will warn, but we keep it for potential
     * future use in logging or debugging policy decisions.
     */
    @SuppressWarnings("unused")  // Suppress "unused" warning
    private static final Gson gson = new GsonBuilder()
            .setPrettyPrinting()  // Pretty-print JSON output
            .create();            // Build the Gson instance
    
    /**
     * Singleton instance - only one RLInferenceService exists in the application.
     * 
     * SINGLETON PATTERN:
     * - Ensures only one instance of this class is created
     * - Provides global access point via getInstance()
     * - Useful for: database connections, configuration, services with state
     * 
     * WHY SINGLETON HERE?
     * - The native engine is expensive to initialize
     * - We want to reuse the same engine across all requests
     * - Avoids loading the ONNX model multiple times
     */
    private static RLInferenceService instance;
    
    // ========================================================================
    // INSTANCE FIELDS
    // ========================================================================
    
    /**
     * Reference to AIController for calling native JNI methods.
     * 
     * We need an AIController instance because native methods (nativeInferState)
     * are defined there. In Java, native methods must be called on an instance
     * of the class that declares them.
     */
    private AIController controller;
    
    /**
     * Flag indicating if native C++ engine is available.
     * If false, we fall back to simulation mode.
     */
    private boolean nativeAvailable = false;
    
    /**
     * Name of the ONNX model file being used.
     * Displayed in API responses for debugging.
     */
    private String policyModel = "cartpole_policy.onnx";
    
    /**
     * Dimensionality of the state vector.
     * CartPole has 4 state dimensions:
     *   0: cart position (meters from center)
     *   1: cart velocity (meters/second)
     *   2: pole angle (radians from vertical)
     *   3: pole angular velocity (radians/second)
     */
    private int stateDim = 4;
    
    /**
     * Number of possible actions.
     * CartPole has 2 actions:
     *   0: Push cart to the LEFT
     *   1: Push cart to the RIGHT
     */
    private int actionDim = 2;
    
    /**
     * Human-readable labels for each action.
     * Used in API responses to make them more understandable.
     * 
     * STATIC FINAL ARRAY:
     * - static: shared across all instances
     * - final: the array reference can't change (but contents could)
     */
    private static final String[] ACTION_LABELS = {
        "push_left",   // Action 0
        "push_right"   // Action 1
    };
    
    // ========================================================================
    // SINGLETON GETTER
    // ========================================================================
    
    /**
     * Gets the singleton instance of RLInferenceService.
     * 
     * THREAD SAFETY:
     * - The 'synchronized' keyword ensures only one thread can execute
     *   this method at a time
     * - Without it, two threads could both create instances simultaneously
     * - This is called "double-checked locking" (simplified version)
     * 
     * LAZY INITIALIZATION:
     * - The instance is only created when first requested
     * - Saves resources if the RL service is never used
     * 
     * @return The singleton RLInferenceService instance
     */
    public static synchronized RLInferenceService getInstance() {
        // Check if instance already exists
        if (instance == null) {
            // First call - create the singleton instance
            instance = new RLInferenceService();
        }
        // Return the (possibly newly created) instance
        return instance;
    }
    
    // ========================================================================
    // CONSTRUCTOR
    // ========================================================================
    
    /**
     * Private constructor - prevents external instantiation.
     * 
     * WHY PRIVATE?
     * - Enforces the Singleton pattern
     * - Only getInstance() can create an instance
     * - Prevents accidental creation of multiple instances
     * 
     * This constructor tries to initialize the native engine.
     * If that fails, we continue in simulation mode.
     */
    private RLInferenceService() {
        try {
            // Create an AIController instance to access native methods
            // This will trigger the static block in AIController that loads
            // the native library (libinference_engine.so)
            controller = new AIController();
            
            // If we get here without exception, native is available
            nativeAvailable = true;
            
            System.out.println("✅ RLInferenceService initialized with ONNX backend");
            
        } catch (Exception e) {
            // Native library failed to load - fall back to simulation
            nativeAvailable = false;
            System.out.println("⚠️ RLInferenceService initialized (simulation mode)");
        }
    }
    
    // ========================================================================
    // MAIN INFERENCE METHOD
    // ========================================================================
    
    /**
     * Gets an action from the policy given the current state.
     * 
     * FLOW:
     * 1. Validate input state
     * 2. Call native engine (or simulate if unavailable)
     * 3. Convert raw logits to probabilities via softmax
     * 4. Select action with highest probability
     * 5. Return structured result
     * 
     * @param state The environment state vector [cart_pos, cart_vel, pole_angle, pole_vel]
     * @return PolicyResult containing the selected action and probabilities
     */
    public PolicyResult getAction(float[] state) {
        // Record start time for latency measurement
        // System.currentTimeMillis() returns milliseconds since Unix epoch (Jan 1, 1970)
        long startTime = System.currentTimeMillis();
        
        // Create result object to populate
        PolicyResult result = new PolicyResult();
        result.model = policyModel;      // Which model we're using
        result.stateDim = stateDim;      // Expected state dimension
        result.actionDim = actionDim;    // Number of actions
        
        // ====================================================================
        // INPUT VALIDATION
        // ====================================================================
        
        // Check if state is null or wrong dimension
        if (state == null || state.length != stateDim) {
            result.success = false;
            result.error = "Invalid state dimension. Expected " + stateDim + " values.";
            return result;
        }
        
        // Store input state in result (useful for debugging)
        result.inputState = state;
        
        // ====================================================================
        // INFERENCE
        // ====================================================================
        
        try {
            // Array to hold raw model outputs (logits)
            float[] actionLogits;
            
            if (nativeAvailable && controller != null) {
                // ============================================================
                // REAL INFERENCE PATH
                // ============================================================
                // Call the native C++ method via JNI
                // This runs the ONNX model using ONNX Runtime in C++
                actionLogits = controller.nativeInferState(state);
                
                // Mark result as NOT simulated (real inference)
                result.simulated = false;
                
            } else {
                // ============================================================
                // SIMULATION PATH
                // ============================================================
                // Native engine not available - use simple heuristic
                actionLogits = simulatePolicyLogits(state);
                
                // Mark result as simulated
                result.simulated = true;
            }
            
            // ================================================================
            // POST-PROCESSING
            // ================================================================
            
            // Convert logits to probabilities using softmax
            // Softmax: P(action_i) = exp(logit_i) / sum(exp(logit_j))
            result.actionProbabilities = softmax(actionLogits);
            
            // Select action with highest probability
            // argmax returns the INDEX of the maximum value
            result.selectedAction = argmax(result.actionProbabilities);
            
            // Get human-readable label for selected action
            result.actionLabel = ACTION_LABELS[result.selectedAction];
            
            // Confidence is the probability of the selected action
            result.confidence = result.actionProbabilities[result.selectedAction];
            
            // Mark as successful
            result.success = true;
            
        } catch (Exception e) {
            // ================================================================
            // ERROR FALLBACK
            // ================================================================
            // If native inference threw an exception, fall back to simulation
            
            float[] actionLogits = simulatePolicyLogits(state);
            result.actionProbabilities = softmax(actionLogits);
            result.selectedAction = argmax(result.actionProbabilities);
            result.actionLabel = ACTION_LABELS[result.selectedAction];
            result.confidence = result.actionProbabilities[result.selectedAction];
            result.success = true;
            result.simulated = true;
            result.error = "Native inference failed: " + e.getMessage();
        }
        
        // Calculate how long this request took
        result.latencyMs = System.currentTimeMillis() - startTime;
        
        return result;
    }
    
    // ========================================================================
    // SOFTMAX FUNCTION
    // ========================================================================
    
    /**
     * Applies softmax function to convert logits to probabilities.
     * 
     * SOFTMAX EXPLAINED:
     * ------------------
     * Neural networks often output "logits" - raw, unbounded scores.
     * Softmax converts these to probabilities that:
     * 1. Are all positive (>0)
     * 2. Sum to 1.0
     * 3. Preserve the relative ordering
     * 
     * FORMULA:
     *   P(i) = exp(logit_i) / Σ exp(logit_j)
     * 
     * NUMERICAL STABILITY:
     * - Exponentials can overflow (exp(1000) = infinity)
     * - We subtract max(logits) first: exp(x - max) is always ≤ 1
     * - This doesn't change the result because:
     *   exp(a - max) / Σ exp(b - max) = exp(a) / Σ exp(b)
     * 
     * @param logits Raw model outputs (can be any real numbers)
     * @return Probabilities (all positive, sum to 1.0)
     */
    private float[] softmax(float[] logits) {
        // Step 1: Find maximum logit for numerical stability
        float maxLogit = Float.NEGATIVE_INFINITY;  // Start with smallest possible value
        
        for (float logit : logits) {
            // Math.max returns the larger of two values
            maxLogit = Math.max(maxLogit, logit);
        }
        
        // Step 2: Compute exp(logit - max) and sum
        float sumExp = 0.0f;
        float[] probs = new float[logits.length];
        
        for (int i = 0; i < logits.length; i++) {
            // Subtract max for stability, then exponentiate
            // Math.exp computes e^x
            probs[i] = (float) Math.exp(logits[i] - maxLogit);
            sumExp += probs[i];
        }
        
        // Step 3: Normalize by sum (so probabilities sum to 1)
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sumExp;
        }
        
        return probs;
    }
    
    // ========================================================================
    // SIMULATION FALLBACK
    // ========================================================================
    
    /**
     * Simulates policy output when native engine is unavailable.
     * 
     * This is a simple heuristic, NOT a trained policy:
     * - If pole is tilting right (positive angle), prefer pushing right
     * - If pole is tilting left (negative angle), prefer pushing left
     * 
     * The real trained policy is much more sophisticated, considering
     * velocities and learning optimal control through experience.
     * 
     * @param state Current environment state
     * @return Simulated action logits
     */
    private float[] simulatePolicyLogits(float[] state) {
        // Extract pole angle from state
        // State layout: [cart_pos, cart_vel, pole_angle, pole_vel]
        float poleAngle = state[2];    // Index 2 is pole angle
        float poleVel = state[3];      // Index 3 is angular velocity
        
        // Simple heuristic: consider both angle and velocity
        // If angle + 0.1*velocity > 0, pole is "effectively" tilting right
        if (poleAngle + 0.1f * poleVel > 0) {
            // Tilting right → prefer pushing right (action 1)
            return new float[]{-0.5f, 0.5f};  // [left_logit, right_logit]
        } else {
            // Tilting left → prefer pushing left (action 0)
            return new float[]{0.5f, -0.5f};  // [left_logit, right_logit]
        }
    }
    
    // ========================================================================
    // Q-VALUE QUERY
    // ========================================================================
    
    /**
     * Gets Q-values for all actions given a state.
     * 
     * Q-VALUES EXPLAINED:
     * -------------------
     * Q(s, a) represents the expected total future reward if we:
     * 1. Start in state s
     * 2. Take action a
     * 3. Then follow our policy forever
     * 
     * Higher Q-value = better action in that state.
     * 
     * NOTE: Our REINFORCE policy doesn't explicitly compute Q-values.
     * We return logits as "pseudo-Q-values" for this endpoint.
     * True Q-learning uses different architectures (like DQN).
     * 
     * @param state Current environment state
     * @return QValueResult containing Q-values for all actions
     */
    public QValueResult getQValues(float[] state) {
        // Create result object
        QValueResult result = new QValueResult();
        result.model = policyModel + "_qvalues";
        
        // Validate input
        if (state == null || state.length != stateDim) {
            result.success = false;
            result.error = "Invalid state dimension";
            return result;
        }
        
        try {
            float[] logits;
            
            // Get logits from policy (either native or simulated)
            if (nativeAvailable && controller != null) {
                logits = controller.nativeInferState(state);
            } else {
                logits = simulatePolicyLogits(state);
            }
            
            // Return logits as pseudo-Q-values
            result.qValues = logits;
            result.bestAction = argmax(logits);
            result.actionLabel = ACTION_LABELS[result.bestAction];
            result.success = true;
            
        } catch (Exception e) {
            // Fallback to simple heuristic
            float poleAngle = state[2];
            result.qValues = new float[]{
                -Math.abs(poleAngle) - 0.1f,  // Q(s, left)
                -Math.abs(poleAngle) + 0.1f   // Q(s, right) - slightly prefer right
            };
            result.bestAction = argmax(result.qValues);
            result.actionLabel = ACTION_LABELS[result.bestAction];
            result.success = true;
        }
        
        return result;
    }
    
    // ========================================================================
    // UTILITY FUNCTIONS
    // ========================================================================
    
    /**
     * Returns the index of the maximum value in an array.
     * 
     * ARGMAX EXPLAINED:
     * - "arg" = "argument" (input that produces a result)
     * - "max" = maximum
     * - argmax(array) = index of the maximum element
     * 
     * Example: argmax([0.2, 0.7, 0.1]) = 1 (because 0.7 is largest)
     * 
     * Used to select the action with highest probability.
     * 
     * @param arr Array of values
     * @return Index of maximum value
     */
    private int argmax(float[] arr) {
        int maxIdx = 0;           // Assume first element is max
        float maxVal = arr[0];    // Its value
        
        // Scan through array looking for larger values
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];  // Found new maximum
                maxIdx = i;       // Remember its index
            }
        }
        
        return maxIdx;
    }
    
    // ========================================================================
    // GETTERS FOR STATUS
    // ========================================================================
    
    /**
     * Checks if the policy model is loaded and ready.
     * @return true if native ONNX engine is available
     */
    public boolean isPolicyLoaded() {
        return nativeAvailable;
    }
    
    /**
     * Gets the human-readable action labels.
     * @return Array of action names ["push_left", "push_right"]
     */
    public String[] getActionLabels() {
        return ACTION_LABELS;
    }
    
    // ========================================================================
    // RESULT DATA CLASSES
    // ========================================================================
    /**
     * These inner classes define the structure of responses returned by this service.
     * Gson serializes them directly to JSON for the REST API.
     */
    
    /**
     * Result structure for policy action queries.
     * 
     * Contains all information about the policy's decision:
     * - What action was selected
     * - How confident the policy is
     * - Whether real inference or simulation was used
     */
    public static class PolicyResult {
        /** Whether the inference succeeded */
        public boolean success;
        
        /** Whether this result is from simulation (true) or real ONNX inference (false) */
        public boolean simulated;
        
        /** Name of the model used */
        public String model;
        
        /** Expected state dimension */
        public int stateDim;
        
        /** Number of possible actions */
        public int actionDim;
        
        /** The input state that was evaluated */
        public float[] inputState;
        
        /** Probability distribution over actions (sums to 1.0) */
        public float[] actionProbabilities;
        
        /** Index of the selected action (0 or 1 for CartPole) */
        public int selectedAction;
        
        /** Human-readable name of selected action */
        public String actionLabel;
        
        /** Probability of the selected action (confidence score) */
        public float confidence;
        
        /** How long inference took in milliseconds */
        public long latencyMs;
        
        /** Error message if something went wrong */
        public String error;
    }
    
    /**
     * Result structure for Q-value queries.
     * 
     * Q-values represent the "quality" of each action in the given state.
     */
    public static class QValueResult {
        /** Whether the query succeeded */
        public boolean success;
        
        /** Name of the model used */
        public String model;
        
        /** Q-value for each action */
        public float[] qValues;
        
        /** Index of action with highest Q-value */
        public int bestAction;
        
        /** Human-readable name of best action */
        public String actionLabel;
        
        /** Error message if something went wrong */
        public String error;
    }
}
