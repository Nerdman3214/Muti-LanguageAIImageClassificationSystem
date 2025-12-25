package ai.controller;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import spark.Request;
import spark.Response;

import javax.servlet.MultipartConfigElement;
import javax.servlet.http.Part;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.*;

import static spark.Spark.*;

/**
 * REST API Controller for ML Inference
 * 
 * Design Pattern: Controller Pattern
 * - Handles HTTP routing
 * - Validates inputs
 * - Delegates to native inference
 * 
 * OpenAPI/Swagger compatible
 */
public class AIController {
    
    private static final Gson gson = new GsonBuilder().setPrettyPrinting().create();
    private static final int PORT = 8080;
    private static final String UPLOAD_DIR = "uploads";
    
    // JNI native library
    private static boolean nativeLibraryLoaded = false;
    
    static {
        try {
            System.loadLibrary("inference_engine");
            nativeLibraryLoaded = true;
            System.out.println("✅ Native inference engine loaded successfully");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("⚠️ Warning: Could not load native library: " + e.getMessage());
            System.err.println("   Running in simulation mode...");
            nativeLibraryLoaded = false;
        }
    }
    
    // Native methods (JNI)
    public native float[] nativeInfer(String imagePath);
    public native float[] nativeInferState(float[] state);  // RL policy inference
    
    /**
     * Classification result structure
     */
    public static class ClassificationResult {
        public String status;
        public String imageName;
        public List<Prediction> predictions;
        public long inferenceTimeMs;
        public String modelVersion;
        
        public static class Prediction {
            public int classIndex;
            public String label;
            public float confidence;
            
            public Prediction(int classIndex, String label, float confidence) {
                this.classIndex = classIndex;
                this.label = label;
                this.confidence = confidence;
            }
        }
    }
    
    /**
     * Health check response
     */
    public static class HealthResponse {
        public String status;
        public boolean nativeEngineAvailable;
        public String version;
        public long timestamp;
    }
    
    /**
     * Error response
     */
    public static class ErrorResponse {
        public String status = "error";
        public String message;
        
        public ErrorResponse(String message) {
            this.message = message;
        }
    }
    
    // ImageNet labels - loaded from file
    private static String[] IMAGENET_LABELS;
    
    static {
        // Load labels from file
        try {
            java.util.List<String> lines = java.nio.file.Files.readAllLines(
                java.nio.file.Paths.get("models/labels_imagenet.txt"));
            IMAGENET_LABELS = lines.toArray(new String[0]);
            System.out.println("✅ Loaded " + IMAGENET_LABELS.length + " ImageNet labels");
        } catch (Exception e) {
            System.err.println("⚠️ Could not load labels file: " + e.getMessage());
            // Fallback labels
            IMAGENET_LABELS = new String[]{
                "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
                "electric ray", "stingray", "rooster", "hen", "ostrich"
            };
        }
    }
    
    /**
     * Main entry point - starts the REST server
     */
    public static void main(String[] args) {
        AIController controller = new AIController();
        controller.startServer();
    }
    
    /**
     * Start the REST API server
     */
    public void startServer() {
        // Configure Spark
        port(PORT);
        
        // Create upload directory
        new File(UPLOAD_DIR).mkdirs();
        
        // Enable CORS
        before((request, response) -> {
            response.header("Access-Control-Allow-Origin", "*");
            response.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            response.header("Access-Control-Allow-Headers", "Content-Type");
        });
        
        // Handle preflight
        options("/*", (request, response) -> {
            return "OK";
        });
        
        // ============================================
        // REST Endpoints - Image Classification
        // ============================================
        
        /**
         * POST /classify
         * Upload and classify an image
         */
        post("/classify", "multipart/form-data", this::handleClassify);
        
        /**
         * GET /health
         * Health check endpoint
         */
        get("/health", this::handleHealth);
        
        /**
         * GET /info
         * Get model and system information
         */
        get("/info", this::handleInfo);
        
        // ============================================
        // REST Endpoints - Reinforcement Learning
        // ============================================
        
        /**
         * POST /rl/action
         * Get action from RL policy given state
         * Body: {"state": [0.0, 0.0, 0.1, 0.0]}
         */
        post("/rl/action", this::handleRLAction);
        
        /**
         * POST /rl/qvalues
         * Get Q-values for all actions given state
         * Body: {"state": [0.0, 0.0, 0.1, 0.0]}
         */
        post("/rl/qvalues", this::handleRLQValues);
        
        /**
         * GET /rl/info
         * Get RL policy information
         */
        get("/rl/info", this::handleRLInfo);
        
        /**
         * GET /
         * Welcome page
         */
        get("/", (req, res) -> {
            res.type("text/html");
            return getWelcomePage();
        });
        
        // Error handling
        exception(Exception.class, (e, req, res) -> {
            res.status(500);
            res.type("application/json");
            res.body(gson.toJson(new ErrorResponse(e.getMessage())));
        });
        
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
    
    /**
     * Handle image classification request
     */
    private String handleClassify(Request req, Response res) {
        res.type("application/json");
        
        try {
            // Configure multipart
            req.attribute("org.eclipse.jetty.multipartConfig",
                new MultipartConfigElement(UPLOAD_DIR, 10_000_000, 10_000_000, 1024));
            
            Part filePart = req.raw().getPart("image");
            if (filePart == null) {
                res.status(400);
                return gson.toJson(new ErrorResponse("No image file provided. Use 'image' form field."));
            }
            
            String fileName = getFileName(filePart);
            Path tempFile = Files.createTempFile(Path.of(UPLOAD_DIR), "upload_", "_" + fileName);
            
            // Save uploaded file
            try (InputStream is = filePart.getInputStream()) {
                Files.copy(is, tempFile, StandardCopyOption.REPLACE_EXISTING);
            }
            
            long startTime = System.currentTimeMillis();
            
            // Perform classification
            ClassificationResult result = classifyImageFile(tempFile.toString(), fileName);
            
            result.inferenceTimeMs = System.currentTimeMillis() - startTime;
            
            // Clean up
            Files.deleteIfExists(tempFile);
            
            return gson.toJson(result);
            
        } catch (Exception e) {
            res.status(500);
            return gson.toJson(new ErrorResponse("Classification failed: " + e.getMessage()));
        }
    }
    
    /**
     * Classify an image file
     */
    private ClassificationResult classifyImageFile(String imagePath, String fileName) {
        ClassificationResult result = new ClassificationResult();
        result.status = "success";
        result.imageName = fileName;
        result.modelVersion = "1.0.0";
        result.predictions = new ArrayList<>();
        
        if (nativeLibraryLoaded) {
            // Use native C++ inference
            try {
                float[] probs = nativeInfer(imagePath);
                if (probs != null) {
                    // Get top-5 predictions
                    List<int[]> topK = getTopK(probs, 5);
                    for (int[] item : topK) {
                        int idx = item[0];
                        String label = idx < IMAGENET_LABELS.length ? 
                            IMAGENET_LABELS[idx] : "class_" + idx;
                        result.predictions.add(
                            new ClassificationResult.Prediction(idx, label, probs[idx])
                        );
                    }
                }
            } catch (Exception e) {
                System.err.println("Native inference failed: " + e.getMessage());
                // Fall through to simulation
            }
        }
        
        // Simulation mode if native not available or failed
        if (result.predictions.isEmpty()) {
            result.predictions.add(new ClassificationResult.Prediction(1, "goldfish", 0.45f));
            result.predictions.add(new ClassificationResult.Prediction(0, "tench", 0.15f));
            result.predictions.add(new ClassificationResult.Prediction(2, "great white shark", 0.10f));
            result.predictions.add(new ClassificationResult.Prediction(3, "tiger shark", 0.08f));
            result.predictions.add(new ClassificationResult.Prediction(4, "hammerhead shark", 0.05f));
            result.status = "success (simulation mode)";
        }
        
        return result;
    }
    
    /**
     * Get top-K indices from probability array
     */
    private List<int[]> getTopK(float[] probs, int k) {
        List<int[]> indexed = new ArrayList<>();
        for (int i = 0; i < probs.length; i++) {
            indexed.add(new int[]{i, Float.floatToIntBits(probs[i])});
        }
        indexed.sort((a, b) -> Float.compare(
            Float.intBitsToFloat(b[1]), Float.intBitsToFloat(a[1])));
        return indexed.subList(0, Math.min(k, indexed.size()));
    }
    
    /**
     * Handle health check
     */
    private String handleHealth(Request req, Response res) {
        res.type("application/json");
        
        HealthResponse health = new HealthResponse();
        health.status = "healthy";
        health.nativeEngineAvailable = nativeLibraryLoaded;
        health.version = "1.0.0";
        health.timestamp = System.currentTimeMillis();
        
        return gson.toJson(health);
    }
    
    /**
     * Handle info request
     */
    private String handleInfo(Request req, Response res) {
        res.type("application/json");
        
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("name", "Multi-Language AI Image Classification System");
        info.put("version", "1.0.0");
        info.put("model", "ResNet-50 (ImageNet)");
        info.put("numClasses", 1000);
        info.put("inputSize", "224x224x3");
        info.put("nativeEngineLoaded", nativeLibraryLoaded);
        info.put("endpoints", Arrays.asList(
            "POST /classify - Classify an image",
            "GET /health - Health check",
            "GET /info - This endpoint"
        ));
        
        return gson.toJson(info);
    }
    
    /**
     * Get filename from multipart
     */
    private String getFileName(Part part) {
        String header = part.getHeader("content-disposition");
        for (String cd : header.split(";")) {
            if (cd.trim().startsWith("filename")) {
                return cd.substring(cd.indexOf('=') + 1).trim().replace("\"", "");
            }
        }
        return "unknown";
    }
    
    /**
     * Welcome page HTML
     */
    private String getWelcomePage() {
        return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Image Classification API</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                           max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
                    .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    h1 { color: #333; }
                    .endpoint { background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    .method { color: white; padding: 3px 8px; border-radius: 3px; font-weight: bold; }
                    .post { background: #49cc90; }
                    .get { background: #61affe; }
                    code { background: #eee; padding: 2px 6px; border-radius: 3px; }
                    form { margin: 20px 0; padding: 20px; background: #f9f9f9; border-radius: 5px; }
                    input[type="file"] { margin: 10px 0; }
                    button { background: #4CAF50; color: white; padding: 10px 20px; border: none; 
                             border-radius: 5px; cursor: pointer; font-size: 16px; }
                    button:hover { background: #45a049; }
                    #result { margin-top: 20px; padding: 15px; background: #e8f5e9; border-radius: 5px; display: none; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 AI Image Classification API</h1>
                    <p>Multi-Language AI System - Java REST API + C++ Inference Engine</p>
                    
                    <h2>Try It Out</h2>
                    <form id="classifyForm" enctype="multipart/form-data">
                        <input type="file" name="image" id="imageInput" accept="image/*" required>
                        <br>
                        <button type="submit">🔍 Classify Image</button>
                    </form>
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
                
                <script>
                    document.getElementById('classifyForm').addEventListener('submit', async (e) => {
                        e.preventDefault();
                        const formData = new FormData();
                        formData.append('image', document.getElementById('imageInput').files[0]);
                        
                        const resultDiv = document.getElementById('result');
                        resultDiv.style.display = 'block';
                        resultDiv.innerHTML = '⏳ Classifying...';
                        
                        try {
                            const response = await fetch('/classify', { method: 'POST', body: formData });
                            const data = await response.json();
                            resultDiv.innerHTML = '<h3>Results:</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                        } catch (error) {
                            resultDiv.innerHTML = '❌ Error: ' + error.message;
                        }
                    });
                </script>
            </body>
            </html>
            """;
    }
    
    // ============================================
    // Reinforcement Learning Handlers
    // ============================================
    
    /**
     * Handle RL action request
     * POST /rl/action
     * Body: {"state": [0.0, 0.0, 0.1, 0.0]}
     */
    private String handleRLAction(Request req, Response res) {
        res.type("application/json");
        
        try {
            // Parse state from JSON body
            RLRequest rlReq = gson.fromJson(req.body(), RLRequest.class);
            
            if (rlReq == null || rlReq.state == null) {
                res.status(400);
                return gson.toJson(new ErrorResponse("Missing 'state' array in request body"));
            }
            
            // Get action from RL service
            RLInferenceService rlService = RLInferenceService.getInstance();
            RLInferenceService.PolicyResult result = rlService.getAction(rlReq.state);
            
            return gson.toJson(result);
            
        } catch (Exception e) {
            res.status(500);
            return gson.toJson(new ErrorResponse("RL inference failed: " + e.getMessage()));
        }
    }
    
    /**
     * Handle RL Q-values request
     * POST /rl/qvalues
     * Body: {"state": [0.0, 0.0, 0.1, 0.0]}
     */
    private String handleRLQValues(Request req, Response res) {
        res.type("application/json");
        
        try {
            RLRequest rlReq = gson.fromJson(req.body(), RLRequest.class);
            
            if (rlReq == null || rlReq.state == null) {
                res.status(400);
                return gson.toJson(new ErrorResponse("Missing 'state' array in request body"));
            }
            
            RLInferenceService rlService = RLInferenceService.getInstance();
            RLInferenceService.QValueResult result = rlService.getQValues(rlReq.state);
            
            return gson.toJson(result);
            
        } catch (Exception e) {
            res.status(500);
            return gson.toJson(new ErrorResponse("Q-value query failed: " + e.getMessage()));
        }
    }
    
    /**
     * Handle RL info request
     * GET /rl/info
     */
    private String handleRLInfo(Request req, Response res) {
        res.type("application/json");
        
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("name", "CartPole Policy Engine");
        info.put("algorithm", "REINFORCE (Policy Gradient)");
        info.put("stateDim", 4);
        info.put("stateDescription", Arrays.asList(
            "cart_position",
            "cart_velocity", 
            "pole_angle",
            "pole_angular_velocity"
        ));
        info.put("actionDim", 2);
        info.put("actions", Arrays.asList("push_left", "push_right"));
        info.put("endpoints", Arrays.asList(
            "POST /rl/action - Get action from policy given state",
            "POST /rl/qvalues - Get Q-values for all actions",
            "GET /rl/info - This endpoint"
        ));
        info.put("example", Map.of(
            "request", Map.of("state", Arrays.asList(0.0, 0.0, 0.1, 0.0)),
            "description", "Pole tilted 0.1 rad right"
        ));
        
        return gson.toJson(info);
    }
    
    /**
     * Request body for RL endpoints
     */
    private static class RLRequest {
        public float[] state;
    }
}

