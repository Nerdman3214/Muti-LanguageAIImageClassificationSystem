/**
 * ============================================================================
 * AIController.java - REST API Controller for Multi-Language AI System
 * ============================================================================
 * 
 * This file is the main entry point for the Java REST API server.
 * It uses the Spark framework to handle HTTP requests and delegates
 * AI inference to a native C++ engine via JNI (Java Native Interface).
 * 
 * ARCHITECTURE OVERVIEW:
 * ----------------------
 * [HTTP Client] → [Java REST API] → [JNI Bridge] → [C++ ONNX Runtime] → [AI Model]
 * 
 * KEY CONCEPTS COVERED:
 * - REST API design patterns
 * - JNI (Java Native Interface) for calling C++ code
 * - JSON serialization with Gson
 * - File upload handling (multipart/form-data)
 * - Error handling and graceful degradation
 * - Static initialization blocks
 * - Inner classes for data structures
 * 
 * @author Multi-Language AI System
 * @version 1.0.0
 */

// ============================================================================
// PACKAGE DECLARATION
// ============================================================================
// Packages organize Java classes into namespaces to avoid naming conflicts
// This class belongs to the "ai.controller" package
package ai.controller;

// ============================================================================
// IMPORT STATEMENTS
// ============================================================================
// Imports bring external classes into scope so we can use them without
// typing their full package names every time

// Gson is Google's JSON library - converts Java objects to/from JSON strings
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;  // Builder pattern for configuring Gson

// Spark framework classes for handling HTTP requests and responses
import spark.Request;   // Represents incoming HTTP request (headers, body, params)
import spark.Response;  // Represents outgoing HTTP response (status, headers, body)

// Servlet API for handling file uploads (multipart/form-data)
import javax.servlet.MultipartConfigElement;  // Configures multipart parsing
import javax.servlet.http.Part;               // Represents one part of multipart request

// Java I/O classes for reading/writing files and streams
import java.io.*;                              // InputStream, File, etc.
import java.nio.file.Files;                   // Modern file operations (Java 7+)
import java.nio.file.Path;                    // Represents a file system path
import java.nio.file.StandardCopyOption;      // Options for file copy operations

// Java collections for lists, maps, arrays
import java.util.*;  // List, Map, ArrayList, Arrays, LinkedHashMap, etc.

// Static import - allows calling Spark methods without "Spark." prefix
// e.g., instead of Spark.get(), we can just write get()
import static spark.Spark.*;

/**
 * Main REST API Controller class.
 * 
 * DESIGN PATTERN: Controller Pattern (from MVC architecture)
 * - Receives HTTP requests
 * - Validates input data
 * - Delegates business logic to service classes
 * - Formats and returns responses
 * 
 * This class is also responsible for:
 * - Loading the native C++ library via JNI
 * - Defining REST endpoints
 * - Handling errors gracefully
 */
public class AIController {
    
    // ========================================================================
    // STATIC FIELDS (Class-level constants and shared state)
    // ========================================================================
    
    /**
     * Gson instance for JSON serialization/deserialization.
     * 
     * WHY STATIC FINAL?
     * - static: One instance shared across all AIController objects (memory efficient)
     * - final: Cannot be reassigned after initialization (thread-safe, immutable reference)
     * 
     * GsonBuilder().setPrettyPrinting().create():
     * - GsonBuilder: Builder pattern for creating configured Gson instances
     * - setPrettyPrinting(): Formats JSON with indentation (easier to read in responses)
     * - create(): Builds and returns the final Gson object
     */
    private static final Gson gson = new GsonBuilder()  // Create a builder
            .setPrettyPrinting()                        // Enable indented JSON output
            .create();                                  // Build the Gson instance
    
    /**
     * Server port number.
     * Common ports: 80 (HTTP), 443 (HTTPS), 8080 (development), 3000 (Node.js convention)
     * We use 8080 because ports below 1024 require root/admin privileges.
     */
    private static final int PORT = 8080;
    
    /**
     * Directory for storing uploaded files temporarily.
     * Files are saved here during classification, then deleted after processing.
     */
    private static final String UPLOAD_DIR = "uploads";
    
    /**
     * Flag indicating whether the native C++ library was loaded successfully.
     * If false, the system falls back to "simulation mode" with fake predictions.
     */
    private static boolean nativeLibraryLoaded = false;
    
    // ========================================================================
    // STATIC INITIALIZATION BLOCK #1 - Native Library Loading
    // ========================================================================
    /**
     * Static blocks run once when the class is first loaded by the JVM.
     * This block attempts to load the native C++ inference engine.
     * 
     * JNI (Java Native Interface) EXPLAINED:
     * - JNI allows Java code to call functions written in C/C++
     * - The C++ code is compiled into a shared library (.so on Linux, .dll on Windows)
     * - System.loadLibrary("name") loads "libname.so" from java.library.path
     * - Once loaded, we can call native methods declared in this class
     * 
     * WHY USE NATIVE CODE?
     * - Performance: C++ is faster for number crunching and GPU operations
     * - Existing libraries: ONNX Runtime, TensorFlow, etc. are written in C++
     * - Hardware access: Direct GPU/CUDA access is easier from C++
     */
    static {
        try {
            // Attempt to load the native library
            // Java will look for "libinference_engine.so" in paths specified by:
            // -Djava.library.path=<path> JVM argument
            System.loadLibrary("inference_engine");
            
            // If we reach here, loading succeeded
            nativeLibraryLoaded = true;
            
            // Print success message (✅ is a Unicode emoji)
            System.out.println("✅ Native inference engine loaded successfully");
            
        } catch (UnsatisfiedLinkError e) {
            // UnsatisfiedLinkError is thrown when:
            // 1. The library file doesn't exist
            // 2. The library has missing dependencies
            // 3. Architecture mismatch (32-bit vs 64-bit)
            
            System.err.println("⚠️ Warning: Could not load native library: " + e.getMessage());
            System.err.println("   Running in simulation mode...");
            
            // We don't crash - instead we set the flag to false and continue
            // This is called "graceful degradation"
            nativeLibraryLoaded = false;
        }
    }
    
    // ========================================================================
    // NATIVE METHOD DECLARATIONS (JNI)
    // ========================================================================
    /**
     * Native method for image classification.
     * 
     * The 'native' keyword tells Java this method is implemented in C/C++, not Java.
     * The actual implementation is in jni/InferenceJNI.cpp:
     *   Java_ai_controller_AIController_nativeInfer(JNIEnv*, jobject, jstring)
     * 
     * JNI NAMING CONVENTION:
     *   Java_<package>_<class>_<method>
     *   - Package dots become underscores: ai.controller → ai_controller
     * 
     * @param imagePath Path to the image file on disk
     * @return Array of 1000 floats (probabilities for each ImageNet class)
     */
    public native float[] nativeInfer(String imagePath);
    
    /**
     * Native method for RL (Reinforcement Learning) policy inference.
     * 
     * This method runs the CartPole policy model to decide an action.
     * 
     * @param state The environment state: [cart_pos, cart_vel, pole_angle, pole_vel]
     * @return Array of 2 floats (logits for [push_left, push_right])
     */
    public native float[] nativeInferState(float[] state);
    
    // ========================================================================
    // INNER CLASSES - Data Transfer Objects (DTOs)
    // ========================================================================
    /**
     * Inner classes are classes defined inside another class.
     * Static inner classes don't need an instance of the outer class.
     * 
     * DTOs (Data Transfer Objects) are simple classes that just hold data.
     * Gson converts these to/from JSON automatically using reflection.
     */
    
    /**
     * Represents the result of an image classification.
     * 
     * JSON OUTPUT EXAMPLE:
     * {
     *   "status": "success",
     *   "imageName": "dog.jpg",
     *   "predictions": [
     *     {"classIndex": 207, "label": "Golden Retriever", "confidence": 0.57},
     *     ...
     *   ],
     *   "inferenceTimeMs": 45,
     *   "modelVersion": "1.0.0"
     * }
     */
    public static class ClassificationResult {
        
        // Public fields are directly serialized to JSON by Gson
        // Using public fields (instead of private + getters) is simpler for DTOs
        
        public String status;              // "success" or "success (simulation mode)"
        public String imageName;           // Original filename from upload
        public List<Prediction> predictions;  // List of top predictions
        public long inferenceTimeMs;       // How long inference took in milliseconds
        public String modelVersion;        // Version string for tracking
        
        /**
         * Nested inner class representing a single prediction.
         * Nested classes can access outer class members.
         */
        public static class Prediction {
            public int classIndex;    // Index in ImageNet (0-999)
            public String label;      // Human-readable class name
            public float confidence;  // Probability (0.0 to 1.0)
            
            /**
             * Constructor - initializes all fields.
             * Constructors have the same name as the class and no return type.
             */
            public Prediction(int classIndex, String label, float confidence) {
                this.classIndex = classIndex;  // 'this' refers to the current object
                this.label = label;
                this.confidence = confidence;
            }
        }
    }
    
    /**
     * Response structure for the /health endpoint.
     * Health checks are used by load balancers and monitoring systems.
     */
    public static class HealthResponse {
        public String status;               // "healthy" or "unhealthy"
        public boolean nativeEngineAvailable;  // Is C++ engine loaded?
        public String version;              // API version
        public long timestamp;              // Unix timestamp in milliseconds
    }
    
    /**
     * Generic error response structure.
     * Returning consistent error formats makes client-side error handling easier.
     */
    public static class ErrorResponse {
        public String status = "error";  // Field initializer - default value
        public String message;           // Error description
        
        /**
         * Constructor taking error message.
         * @param message Human-readable error description
         */
        public ErrorResponse(String message) {
            this.message = message;
        }
    }
    
    // ========================================================================
    // ImageNet LABELS
    // ========================================================================
    /**
     * Array of class labels for ImageNet (1000 classes).
     * These labels convert numeric class indices to human-readable names.
     * Example: index 207 → "Golden Retriever"
     */
    private static String[] IMAGENET_LABELS;
    
    // ========================================================================
    // STATIC INITIALIZATION BLOCK #2 - Load Labels
    // ========================================================================
    /**
     * Static block to load ImageNet labels from file.
     * Multiple static blocks run in order of appearance in the source code.
     */
    static {
        try {
            // Files.readAllLines() reads a text file into a List of Strings
            // Each line becomes one element in the list
            // Paths.get() converts a String path to a Path object
            java.util.List<String> lines = java.nio.file.Files.readAllLines(
                java.nio.file.Paths.get("models/labels_imagenet.txt")
            );
            
            // Convert List<String> to String[] array
            // toArray(new String[0]) is the modern idiom - the array size doesn't matter
            IMAGENET_LABELS = lines.toArray(new String[0]);
            
            System.out.println("✅ Loaded " + IMAGENET_LABELS.length + " ImageNet labels");
            
        } catch (Exception e) {
            // If file loading fails, use a small fallback array
            // This ensures the application can still run (graceful degradation)
            System.err.println("⚠️ Could not load labels file: " + e.getMessage());
            
            // Fallback labels - just the first 10 ImageNet classes
            IMAGENET_LABELS = new String[]{
                "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
                "electric ray", "stingray", "rooster", "hen", "ostrich"
            };
        }
    }
    
    // ========================================================================
    // MAIN METHOD - Application Entry Point
    // ========================================================================
    /**
     * Main method - where program execution begins.
     * 
     * In Java, the main method must have this exact signature:
     * - public: accessible from outside the class
     * - static: can be called without creating an object
     * - void: doesn't return anything
     * - String[] args: command-line arguments
     * 
     * @param args Command-line arguments (not used in this application)
     */
    public static void main(String[] args) {
        // Create an instance of this class
        // We need an instance to call non-static methods like startServer()
        AIController controller = new AIController();
        
        // Start the REST API server
        controller.startServer();
    }
    
    // ========================================================================
    // SERVER INITIALIZATION
    // ========================================================================
    /**
     * Initializes and starts the REST API server.
     * 
     * This method:
     * 1. Configures the server port
     * 2. Creates necessary directories
     * 3. Enables CORS (Cross-Origin Resource Sharing)
     * 4. Registers all REST endpoints
     * 5. Sets up error handling
     */
    public void startServer() {
        
        // ====================================================================
        // SERVER CONFIGURATION
        // ====================================================================
        
        // Set the port Spark will listen on
        // This is a static method from Spark, available because of our static import
        port(PORT);
        
        // Create the upload directory if it doesn't exist
        // new File(path).mkdirs() creates the directory and any parent directories
        new File(UPLOAD_DIR).mkdirs();
        
        // ====================================================================
        // CORS (Cross-Origin Resource Sharing) CONFIGURATION
        // ====================================================================
        /**
         * CORS is a security feature in browsers.
         * By default, web pages can only make requests to their own domain.
         * CORS headers tell browsers it's OK for other domains to access this API.
         * 
         * Without CORS headers:
         *   - API works fine from curl, Postman, mobile apps
         *   - API BLOCKED from web pages on different domains
         */
        
        // before() is a "filter" that runs BEFORE every request
        // Lambda syntax: (parameters) -> { code }
        before((request, response) -> {
            // Allow requests from any origin (domain)
            // In production, you'd specify allowed domains: "https://myapp.com"
            response.header("Access-Control-Allow-Origin", "*");
            
            // Allow these HTTP methods
            response.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            
            // Allow these headers in requests
            response.header("Access-Control-Allow-Headers", "Content-Type");
        });
        
        // Handle OPTIONS requests (preflight requests)
        // Browsers send OPTIONS before actual requests to check CORS permissions
        options("/*", (request, response) -> {
            // Just return OK - the before() filter already set the headers
            return "OK";
        });
        
        // ====================================================================
        // IMAGE CLASSIFICATION ENDPOINTS
        // ====================================================================
        
        /**
         * POST /classify - Upload and classify an image
         * 
         * post() registers a POST endpoint
         * Parameters:
         * - "/classify": The URL path
         * - "multipart/form-data": Expected content type (for file uploads)
         * - this::handleClassify: Method reference to the handler
         * 
         * Method references (::) are shorthand for lambdas:
         *   this::handleClassify is equivalent to (req, res) -> this.handleClassify(req, res)
         */
        post("/classify", "multipart/form-data", this::handleClassify);
        
        /**
         * GET /health - Health check endpoint
         * 
         * Health endpoints are used by:
         * - Load balancers to check if server is alive
         * - Kubernetes/Docker for container health
         * - Monitoring systems
         */
        get("/health", this::handleHealth);
        
        /**
         * GET /info - Get model and system information
         * 
         * Info endpoints help API consumers understand capabilities
         */
        get("/info", this::handleInfo);
        
        // ====================================================================
        // REINFORCEMENT LEARNING ENDPOINTS
        // ====================================================================
        
        /**
         * POST /rl/action - Get action from RL policy
         * 
         * Request body (JSON):
         *   {"state": [cart_pos, cart_vel, pole_angle, pole_vel]}
         * 
         * Response: The policy's recommended action
         */
        post("/rl/action", this::handleRLAction);
        
        /**
         * POST /rl/qvalues - Get Q-values for all actions
         * 
         * Q-values represent the "quality" of each action in a given state
         * Higher Q-value = better expected future reward
         */
        post("/rl/qvalues", this::handleRLQValues);
        
        /**
         * GET /rl/info - Get RL policy information
         */
        get("/rl/info", this::handleRLInfo);
        
        // ====================================================================
        // WELCOME PAGE
        // ====================================================================
        
        /**
         * GET / - Welcome page with HTML UI
         * 
         * Lambda expression: (req, res) -> { ... }
         * - req: the Request object
         * - res: the Response object
         * - Code block returns the response body
         */
        get("/", (req, res) -> {
            // Set content type to HTML (default is text/plain)
            res.type("text/html");
            
            // Return HTML content
            return getWelcomePage();
        });
        
        // ====================================================================
        // GLOBAL ERROR HANDLING
        // ====================================================================
        
        /**
         * exception() registers a global exception handler.
         * 
         * When ANY endpoint throws an exception:
         * 1. This handler catches it
         * 2. Sets HTTP 500 status (Internal Server Error)
         * 3. Returns a JSON error response
         * 
         * This prevents ugly stack traces from being sent to clients
         */
        exception(Exception.class, (e, req, res) -> {
            res.status(500);                              // Set HTTP status code
            res.type("application/json");                 // Set content type
            res.body(gson.toJson(new ErrorResponse(e.getMessage())));  // Set response body
        });
        
        // ====================================================================
        // STARTUP MESSAGE
        // ====================================================================
        
        // Print a nice banner showing the server is ready
        // Using Unicode box-drawing characters for the border
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║   🚀 Multi-Language AI API Server Started                  ║");
        System.out.println("║                                                            ║");
        System.out.println("║   Image Classification Endpoints:                          ║");
        System.out.println("║   • POST /classify  - Upload and classify an image         ║");
        System.out.println("║   • GET  /health    - Health check                         ║");
        System.out.println("║   • GET  /info      - Model information                    ║");
        System.out.println("║                                                            ║");
        System.out.println("║   Reinforcement Learning Endpoints:                        ║");
        System.out.println("║   • POST /rl/action   - Get action from policy             ║");
        System.out.println("║   • POST /rl/qvalues  - Get Q-values for state             ║");
        System.out.println("║   • GET  /rl/info     - RL policy information              ║");
        System.out.println("║                                                            ║");
        System.out.println("║   Server running on: http://localhost:" + PORT + "                ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
    }
    
    // ========================================================================
    // REQUEST HANDLERS
    // ========================================================================
    
    /**
     * Handles POST /classify requests.
     * 
     * Multipart File Upload Process:
     * 1. Configure multipart parsing
     * 2. Extract the uploaded file from the request
     * 3. Save it to a temporary file
     * 4. Run classification
     * 5. Delete the temporary file
     * 6. Return JSON results
     * 
     * @param req The HTTP request object
     * @param res The HTTP response object
     * @return JSON string with classification results
     */
    private String handleClassify(Request req, Response res) {
        // Set response content type to JSON
        res.type("application/json");
        
        try {
            // ================================================================
            // STEP 1: Configure Multipart Parsing
            // ================================================================
            /**
             * Multipart requests contain multiple "parts" separated by boundaries.
             * Example raw request:
             * 
             * Content-Type: multipart/form-data; boundary=----WebKitFormBoundary
             * 
             * ------WebKitFormBoundary
             * Content-Disposition: form-data; name="image"; filename="dog.jpg"
             * Content-Type: image/jpeg
             * 
             * [binary image data]
             * ------WebKitFormBoundary--
             * 
             * MultipartConfigElement configures:
             * - Location for temporary files
             * - Maximum file size (10MB)
             * - Maximum request size (10MB)
             * - Threshold before writing to disk (1KB)
             */
            req.attribute("org.eclipse.jetty.multipartConfig",
                new MultipartConfigElement(
                    UPLOAD_DIR,       // Location for temp files
                    10_000_000,       // Max file size: 10MB (underscores are digit separators in Java)
                    10_000_000,       // Max request size: 10MB
                    1024              // File size threshold: 1KB
                ));
            
            // ================================================================
            // STEP 2: Extract Uploaded File
            // ================================================================
            
            // req.raw() gets the underlying javax.servlet.HttpServletRequest
            // getPart("image") gets the part with name="image" from the form
            Part filePart = req.raw().getPart("image");
            
            // Check if file was provided
            if (filePart == null) {
                // Set HTTP 400 Bad Request status
                res.status(400);
                // Return error as JSON
                return gson.toJson(new ErrorResponse("No image file provided. Use 'image' form field."));
            }
            
            // Get the original filename from the upload
            String fileName = getFileName(filePart);
            
            // ================================================================
            // STEP 3: Save to Temporary File
            // ================================================================
            
            // Files.createTempFile creates a unique temporary file
            // Path.of() converts string to Path (Java 11+)
            // Prefix: "upload_", Suffix: "_" + original filename
            Path tempFile = Files.createTempFile(
                Path.of(UPLOAD_DIR),  // Directory
                "upload_",            // Prefix
                "_" + fileName        // Suffix
            );
            
            // try-with-resources: automatically closes the InputStream when done
            // Even if an exception occurs, the stream will be closed
            try (InputStream is = filePart.getInputStream()) {
                // Copy input stream to file, replacing if exists
                Files.copy(is, tempFile, StandardCopyOption.REPLACE_EXISTING);
            }
            
            // ================================================================
            // STEP 4: Run Classification
            // ================================================================
            
            // Record start time for measuring inference duration
            // System.currentTimeMillis() returns milliseconds since Jan 1, 1970 (Unix epoch)
            long startTime = System.currentTimeMillis();
            
            // Call our classification method
            // tempFile.toString() converts Path to String (file path)
            ClassificationResult result = classifyImageFile(tempFile.toString(), fileName);
            
            // Calculate how long inference took
            result.inferenceTimeMs = System.currentTimeMillis() - startTime;
            
            // ================================================================
            // STEP 5: Clean Up
            // ================================================================
            
            // Delete the temporary file to save disk space
            // deleteIfExists doesn't throw if file already deleted
            Files.deleteIfExists(tempFile);
            
            // ================================================================
            // STEP 6: Return Results
            // ================================================================
            
            // gson.toJson() converts Java object to JSON string
            return gson.toJson(result);
            
        } catch (Exception e) {
            // If anything goes wrong, return a 500 error with the message
            res.status(500);  // Internal Server Error
            return gson.toJson(new ErrorResponse("Classification failed: " + e.getMessage()));
        }
    }
    
    /**
     * Performs image classification using the native engine or simulation.
     * 
     * This method demonstrates the "fallback" or "graceful degradation" pattern:
     * 1. Try the preferred method (native C++ inference)
     * 2. If that fails, fall back to a simpler method (simulation)
     * 
     * @param imagePath Full path to the image file
     * @param fileName Original filename for display
     * @return ClassificationResult with predictions
     */
    private ClassificationResult classifyImageFile(String imagePath, String fileName) {
        // Create and populate the result object
        ClassificationResult result = new ClassificationResult();
        result.status = "success";
        result.imageName = fileName;
        result.modelVersion = "1.0.0";
        result.predictions = new ArrayList<>();  // Initialize empty list
        
        // Check if native library is available
        if (nativeLibraryLoaded) {
            try {
                // ============================================================
                // NATIVE INFERENCE PATH
                // ============================================================
                
                // Call the native C++ method via JNI
                // This executes the ONNX model using the C++ inference engine
                float[] probs = nativeInfer(imagePath);
                
                // Check if we got valid results
                if (probs != null) {
                    // Get indices of top 5 predictions
                    // probs is a 1000-element array of probabilities
                    List<int[]> topK = getTopK(probs, 5);
                    
                    // Convert each top prediction to a Prediction object
                    for (int[] item : topK) {
                        int idx = item[0];  // Class index
                        
                        // Get label, with fallback if index is out of bounds
                        // Ternary operator: condition ? valueIfTrue : valueIfFalse
                        String label = idx < IMAGENET_LABELS.length ? 
                            IMAGENET_LABELS[idx] :   // Normal case: use label
                            "class_" + idx;          // Fallback: use index as label
                        
                        // Add prediction to list
                        result.predictions.add(
                            new ClassificationResult.Prediction(idx, label, probs[idx])
                        );
                    }
                }
                
            } catch (Exception e) {
                // Log the error but don't crash
                // We'll fall through to simulation mode
                System.err.println("Native inference failed: " + e.getMessage());
            }
        }
        
        // ====================================================================
        // SIMULATION MODE FALLBACK
        // ====================================================================
        
        // If we have no predictions (native not loaded or failed), use simulation
        if (result.predictions.isEmpty()) {
            // Add some fake predictions for demonstration
            result.predictions.add(new ClassificationResult.Prediction(1, "goldfish", 0.45f));
            result.predictions.add(new ClassificationResult.Prediction(0, "tench", 0.15f));
            result.predictions.add(new ClassificationResult.Prediction(2, "great white shark", 0.10f));
            result.predictions.add(new ClassificationResult.Prediction(3, "tiger shark", 0.08f));
            result.predictions.add(new ClassificationResult.Prediction(4, "hammerhead shark", 0.05f));
            
            // Update status to indicate simulation mode
            result.status = "success (simulation mode)";
        }
        
        return result;
    }
    
    /**
     * Gets the top K indices from a probability array.
     * 
     * ALGORITHM:
     * 1. Create a list of (index, probability) pairs
     * 2. Sort by probability descending
     * 3. Return top K entries
     * 
     * TIME COMPLEXITY: O(n log n) due to sorting
     * 
     * @param probs Array of probabilities (one per class)
     * @param k Number of top predictions to return
     * @return List of [index, probability-as-int-bits] arrays
     */
    private List<int[]> getTopK(float[] probs, int k) {
        // Create list to hold (index, probability) pairs
        // We store probability as int bits because we're using int[] arrays
        List<int[]> indexed = new ArrayList<>();
        
        // Populate the list with all indices and their probabilities
        for (int i = 0; i < probs.length; i++) {
            // Float.floatToIntBits converts float to its IEEE 754 bit representation
            // This allows storing float in an int without loss of precision
            indexed.add(new int[]{i, Float.floatToIntBits(probs[i])});
        }
        
        // Sort by probability descending (highest first)
        // Lambda comparator: compare element b to a (reversed for descending)
        indexed.sort((a, b) -> Float.compare(
            Float.intBitsToFloat(b[1]),  // Convert back to float for comparison
            Float.intBitsToFloat(a[1])
        ));
        
        // Return top K elements
        // Math.min prevents IndexOutOfBoundsException if k > list size
        return indexed.subList(0, Math.min(k, indexed.size()));
    }
    
    /**
     * Handles GET /health requests.
     * 
     * Health check endpoints should:
     * - Be fast (no heavy computation)
     * - Return consistent status codes (200 for healthy)
     * - Include useful diagnostic information
     * 
     * @param req The HTTP request
     * @param res The HTTP response
     * @return JSON health status
     */
    private String handleHealth(Request req, Response res) {
        // Set content type
        res.type("application/json");
        
        // Create health response object
        HealthResponse health = new HealthResponse();
        health.status = "healthy";
        health.nativeEngineAvailable = nativeLibraryLoaded;
        health.version = "1.0.0";
        
        // System.currentTimeMillis() returns Unix timestamp in milliseconds
        // Useful for checking if response is stale/cached
        health.timestamp = System.currentTimeMillis();
        
        return gson.toJson(health);
    }
    
    /**
     * Handles GET /info requests.
     * 
     * Info endpoints provide metadata about the API:
     * - Version information
     * - Capabilities
     * - Available endpoints
     * 
     * @param req The HTTP request
     * @param res The HTTP response
     * @return JSON with system information
     */
    private String handleInfo(Request req, Response res) {
        res.type("application/json");
        
        // LinkedHashMap preserves insertion order (unlike regular HashMap)
        // This makes JSON output more readable and predictable
        Map<String, Object> info = new LinkedHashMap<>();
        
        // Populate info map
        info.put("name", "Multi-Language AI Image Classification System");
        info.put("version", "1.0.0");
        info.put("model", "ResNet-50 (ImageNet)");
        info.put("numClasses", 1000);
        info.put("inputSize", "224x224x3");
        info.put("nativeEngineLoaded", nativeLibraryLoaded);
        
        // Arrays.asList creates a fixed-size list from varargs
        info.put("endpoints", Arrays.asList(
            "POST /classify - Classify an image",
            "GET /health - Health check",
            "GET /info - This endpoint"
        ));
        
        return gson.toJson(info);
    }
    
    /**
     * Extracts filename from a multipart Part.
     * 
     * The filename is embedded in the Content-Disposition header:
     *   Content-Disposition: form-data; name="image"; filename="dog.jpg"
     * 
     * @param part The multipart part
     * @return The filename, or "unknown" if not found
     */
    private String getFileName(Part part) {
        // Get the Content-Disposition header value
        String header = part.getHeader("content-disposition");
        
        // Split by semicolons: ["form-data", " name=\"image\"", " filename=\"dog.jpg\""]
        for (String cd : header.split(";")) {
            // Check if this segment contains "filename"
            if (cd.trim().startsWith("filename")) {
                // Extract the value after "="
                // Then remove quotes
                return cd.substring(cd.indexOf('=') + 1)  // Everything after =
                         .trim()                          // Remove whitespace
                         .replace("\"", "");              // Remove quotes
            }
        }
        
        // Fallback if filename not found
        return "unknown";
    }
    
    /**
     * Returns HTML for the welcome page.
     * 
     * TEXT BLOCKS (Java 15+):
     * The triple-quote syntax \"\"\" ... \"\"\" allows multi-line strings
     * without escape characters. Great for HTML, JSON, SQL, etc.
     * 
     * @return HTML string for the welcome page
     */
    private String getWelcomePage() {
        // Text block - preserves whitespace and newlines
        return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Image Classification API</title>
                <style>
                    /* CSS styles for the page */
                    /* body: main container styling */
                    body { 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                        max-width: 800px;       /* Limit width for readability */
                        margin: 50px auto;      /* Center horizontally */
                        padding: 20px; 
                        background: #f5f5f5;    /* Light gray background */
                    }
                    /* Container card with shadow */
                    .container { 
                        background: white; 
                        padding: 30px; 
                        border-radius: 10px;                         /* Rounded corners */
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);     /* Subtle shadow */
                    }
                    h1 { color: #333; }
                    /* Endpoint documentation boxes */
                    .endpoint { 
                        background: #f0f0f0; 
                        padding: 15px; 
                        margin: 10px 0; 
                        border-radius: 5px; 
                    }
                    /* HTTP method badges */
                    .method { 
                        color: white; 
                        padding: 3px 8px; 
                        border-radius: 3px; 
                        font-weight: bold; 
                    }
                    .post { background: #49cc90; }    /* Green for POST */
                    .get { background: #61affe; }     /* Blue for GET */
                    /* Inline code styling */
                    code { background: #eee; padding: 2px 6px; border-radius: 3px; }
                    /* Form styling */
                    form { 
                        margin: 20px 0; 
                        padding: 20px; 
                        background: #f9f9f9; 
                        border-radius: 5px; 
                    }
                    input[type="file"] { margin: 10px 0; }
                    /* Submit button */
                    button { 
                        background: #4CAF50;     /* Green */
                        color: white; 
                        padding: 10px 20px; 
                        border: none; 
                        border-radius: 5px; 
                        cursor: pointer; 
                        font-size: 16px; 
                    }
                    button:hover { background: #45a049; }  /* Darker on hover */
                    /* Results display area */
                    #result { 
                        margin-top: 20px; 
                        padding: 15px; 
                        background: #e8f5e9;    /* Light green */
                        border-radius: 5px; 
                        display: none;          /* Hidden by default */
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 AI Image Classification API</h1>
                    <p>Multi-Language AI System - Java REST API + C++ Inference Engine</p>
                    
                    <h2>Try It Out</h2>
                    <!-- File upload form -->
                    <!-- enctype="multipart/form-data" is required for file uploads -->
                    <form id="classifyForm" enctype="multipart/form-data">
                        <!-- accept="image/*" limits file picker to images -->
                        <input type="file" name="image" id="imageInput" accept="image/*" required>
                        <br>
                        <button type="submit">🔍 Classify Image</button>
                    </form>
                    <!-- Results will be displayed here -->
                    <div id="result"></div>
                    
                    <h2>Endpoints</h2>
                    <div class="endpoint">
                        <span class="method post">POST</span> <code>/classify</code>
                        <p>Upload and classify an image. Send as multipart/form-data with field name 'image'.</p>
                    </div>
                    <div class="endpoint">
                        <span class="method get">GET</span> <code>/health</code>
                        <p>Health check endpoint. Returns server status.</p>
                    </div>
                    <div class="endpoint">
                        <span class="method get">GET</span> <code>/info</code>
                        <p>Get model and system information.</p>
                    </div>
                    
                    <h2>Example cURL</h2>
                    <pre><code>curl -X POST -F "image=@your_image.jpg" http://localhost:8080/classify</code></pre>
                </div>
                
                <!-- JavaScript for handling form submission -->
                <script>
                    // Add event listener for form submission
                    document.getElementById('classifyForm').addEventListener('submit', async (e) => {
                        // Prevent default form submission (page reload)
                        e.preventDefault();
                        
                        // Create FormData object from file input
                        const formData = new FormData();
                        formData.append('image', document.getElementById('imageInput').files[0]);
                        
                        // Get reference to result div
                        const resultDiv = document.getElementById('result');
                        resultDiv.style.display = 'block';      // Show the div
                        resultDiv.innerHTML = '⏳ Classifying...'; // Show loading message
                        
                        try {
                            // Send POST request using fetch API
                            const response = await fetch('/classify', { 
                                method: 'POST', 
                                body: formData 
                            });
                            
                            // Parse JSON response
                            const data = await response.json();
                            
                            // Display formatted results
                            resultDiv.innerHTML = '<h3>Results:</h3><pre>' + 
                                JSON.stringify(data, null, 2) + '</pre>';
                        } catch (error) {
                            // Display error message
                            resultDiv.innerHTML = '❌ Error: ' + error.message;
                        }
                    });
                </script>
            </body>
            </html>
            """;
    }
    
    // ========================================================================
    // REINFORCEMENT LEARNING HANDLERS
    // ========================================================================
    
    /**
     * Handles POST /rl/action requests.
     * 
     * RL (Reinforcement Learning) Overview:
     * - An agent interacts with an environment
     * - Environment provides a "state" (observations)
     * - Agent chooses an "action" based on its policy
     * - Environment provides "reward" and new state
     * - Goal: Learn a policy that maximizes total reward
     * 
     * CartPole Problem:
     * - State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
     * - Actions: [push_left, push_right]
     * - Goal: Keep the pole balanced upright
     * 
     * @param req HTTP request with JSON body: {"state": [0.0, 0.0, 0.1, 0.0]}
     * @param res HTTP response
     * @return JSON with policy action and probabilities
     */
    private String handleRLAction(Request req, Response res) {
        res.type("application/json");
        
        try {
            // Parse JSON request body into RLRequest object
            // Gson uses reflection to match JSON keys to class fields
            RLRequest rlReq = gson.fromJson(req.body(), RLRequest.class);
            
            // Validate the request
            if (rlReq == null || rlReq.state == null) {
                res.status(400);  // Bad Request
                return gson.toJson(new ErrorResponse("Missing 'state' array in request body"));
            }
            
            // Get the singleton instance of RLInferenceService
            // Singleton pattern: only one instance exists application-wide
            RLInferenceService rlService = RLInferenceService.getInstance();
            
            // Call the service to get action from policy
            RLInferenceService.PolicyResult result = rlService.getAction(rlReq.state);
            
            return gson.toJson(result);
            
        } catch (Exception e) {
            res.status(500);
            return gson.toJson(new ErrorResponse("RL inference failed: " + e.getMessage()));
        }
    }
    
    /**
     * Handles POST /rl/qvalues requests.
     * 
     * Q-values represent the expected future reward for taking each action
     * in the current state. Higher Q-value = better action.
     * 
     * Q(s, a) = Expected total reward starting from state s, taking action a
     * 
     * @param req HTTP request with state
     * @param res HTTP response
     * @return JSON with Q-values for each action
     */
    private String handleRLQValues(Request req, Response res) {
        res.type("application/json");
        
        try {
            // Parse request body
            RLRequest rlReq = gson.fromJson(req.body(), RLRequest.class);
            
            // Validate
            if (rlReq == null || rlReq.state == null) {
                res.status(400);
                return gson.toJson(new ErrorResponse("Missing 'state' array in request body"));
            }
            
            // Get Q-values from service
            RLInferenceService rlService = RLInferenceService.getInstance();
            RLInferenceService.QValueResult result = rlService.getQValues(rlReq.state);
            
            return gson.toJson(result);
            
        } catch (Exception e) {
            res.status(500);
            return gson.toJson(new ErrorResponse("Q-value query failed: " + e.getMessage()));
        }
    }
    
    /**
     * Handles GET /rl/info requests.
     * 
     * Provides documentation about the RL policy and its interface.
     * 
     * @param req HTTP request
     * @param res HTTP response
     * @return JSON with RL policy information
     */
    private String handleRLInfo(Request req, Response res) {
        res.type("application/json");
        
        // Build info map with LinkedHashMap to preserve order
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("name", "CartPole Policy Engine");
        info.put("algorithm", "REINFORCE (Policy Gradient)");
        info.put("stateDim", 4);
        
        // Document what each state dimension represents
        info.put("stateDescription", Arrays.asList(
            "cart_position",           // Index 0: Position of cart on track
            "cart_velocity",           // Index 1: Velocity of cart
            "pole_angle",              // Index 2: Angle of pole (radians from vertical)
            "pole_angular_velocity"    // Index 3: Angular velocity of pole
        ));
        
        info.put("actionDim", 2);
        info.put("actions", Arrays.asList("push_left", "push_right"));
        
        // Document available endpoints
        info.put("endpoints", Arrays.asList(
            "POST /rl/action - Get action from policy given state",
            "POST /rl/qvalues - Get Q-values for all actions",
            "GET /rl/info - This endpoint"
        ));
        
        // Provide usage example
        // Map.of() creates an immutable map (Java 9+)
        info.put("example", Map.of(
            "request", Map.of("state", Arrays.asList(0.0, 0.0, 0.1, 0.0)),
            "description", "Pole tilted 0.1 rad right"
        ));
        
        return gson.toJson(info);
    }
    
    // ========================================================================
    // RL REQUEST DATA STRUCTURE
    // ========================================================================
    
    /**
     * Request body structure for RL endpoints.
     * 
     * JSON Example: {"state": [0.0, 0.0, 0.1, 0.0]}
     * 
     * Private static inner class - only used within AIController.
     * Gson matches the "state" JSON key to this "state" field.
     */
    private static class RLRequest {
        public float[] state;  // The environment state vector
    }
}

