package ai.controller;

/**
 * JNI wrapper for C++ inference engine
 */
public class NativeInference {
    
    static {
        try {
            System.loadLibrary("inference_engine");
            System.out.println("✅ Native inference engine loaded");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("⚠️ Could not load native library: " + e.getMessage());
        }
    }
    
    /**
     * Initialize the inference engine
     * @param modelPath Path to ONNX model file
     * @return true if successful
     */
    public native boolean initialize(String modelPath);
    
    /**
     * Classify an image
     * @param imagePath Path to image file
     * @return JSON string with classification results
     */
    public native String classifyImage(String imagePath);
    
    /**
     * Cleanup resources
     */
    public native void cleanup();
}
