package ai.controller;

/**
 * NativeInference - JNI wrapper for C++ inference engine
 */
public class NativeInference {
    static {
        // Load the native library
        try {
            System.loadLibrary("inference_engine");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native library not found: " + e.getMessage());
        }
    }

    /**
     * Initialize the inference engine
     * @param modelPath Path to the model file
     * @return true if initialization successful
     */
    public native boolean initialize(String modelPath);

    /**
     * Classify an image
     * @param imagePath Path to the image file
     * @return Classification result as JSON string
     */
    public native String classifyImage(String imagePath);

    /**
     * Cleanup and release resources
     */
    public native void cleanup();

    /**
     * Get engine version
     * @return Version string
     */
    public native String getVersion();
}
