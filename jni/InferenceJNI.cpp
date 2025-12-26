/**
 * =============================================================================
 * InferenceJNI.cpp - Java Native Interface Bridge to C++ Inference Engine
 * =============================================================================
 * 
 * JNI (Java Native Interface) OVERVIEW:
 * -------------------------------------
 * JNI is a framework that allows Java code running in the JVM to call (and be
 * called by) native applications and libraries written in C, C++, or assembly.
 * 
 * WHY USE JNI?
 * 1. Performance: C++ can be faster for compute-intensive tasks
 * 2. Access to libraries: Use existing C/C++ libraries (like ONNX Runtime)
 * 3. Hardware access: Direct GPU/CPU optimizations not available in Java
 * 4. Legacy code: Interface with existing native codebases
 * 
 * HOW JNI WORKS:
 * 1. Java declares a method as "native" (no implementation in Java)
 * 2. Java loads a shared library (.so on Linux, .dll on Windows)
 * 3. JVM looks for a C function with a specific naming convention
 * 4. The C function receives a JNIEnv* to interact with the JVM
 * 
 * NAMING CONVENTION:
 * Java method: ai.controller.AIController.nativeInfer(String)
 * C function:  Java_ai_controller_AIController_nativeInfer
 *              ^    ^              ^              ^
 *              |    package        class          method name
 *              prefix (always "Java_")
 * 
 * @file InferenceJNI.cpp
 * @author Multi-Language AI System
 * @version 1.0.0
 */

// ============================================================================
// INCLUDES
// ============================================================================

/**
 * jni.h - The main JNI header file.
 * 
 * This header defines:
 * - JNIEnv*: Pointer to the JNI environment (used to call JVM functions)
 * - jobject, jstring, jfloatArray, etc.: Java type handles
 * - JNIEXPORT and JNICALL: Macros for platform-specific calling conventions
 */
#include <jni.h>

/**
 * Include standard C++ headers BEFORE anything that might include system headers.
 * 
 * ORDER MATTERS!
 * Some system headers or third-party libraries define macros that conflict
 * with our code. Including standard headers first gives us predictable behavior.
 */
#include <memory>     // std::unique_ptr - smart pointer for automatic memory management
#include <stdexcept>  // std::runtime_error - exception class for runtime errors
#include <string>     // std::string - C++ string class (safer than char*)
#include <vector>     // std::vector - dynamic array template

/**
 * MACRO CLEANUP:
 * Some headers might define macros with common names that conflict
 * with our method names. We undefine them to prevent compilation errors.
 * 
 * Example: Some Windows headers define "initialize" as a macro,
 * which would break our InferenceEngine::initialize() method.
 */
#ifdef initialize
#undef initialize
#endif
#ifdef classifyImage  
#undef classifyImage
#endif
#ifdef cleanup
#undef cleanup
#endif

/**
 * Include our custom InferenceEngine header.
 * This declares the C++ class that wraps ONNX Runtime.
 */
#include "InferenceEngine.hpp"

// ============================================================================
// GLOBAL STATE
// ============================================================================

/**
 * Singleton engine instance for image classification.
 * 
 * std::unique_ptr:
 * - Smart pointer that automatically deletes object when it goes out of scope
 * - "Unique" because only one pointer can own the object
 * - Safer than raw pointers - no memory leaks if we forget to delete
 * 
 * WHY GLOBAL SINGLETON?
 * - Loading ONNX models is expensive (reads file, allocates GPU memory, etc.)
 * - We want to load once and reuse across multiple JNI calls
 * - Each JNI call creates a new Java object, but we keep the same C++ engine
 * 
 * THREAD SAFETY WARNING:
 * This simple singleton is NOT thread-safe. If multiple threads call
 * nativeInfer simultaneously, there could be race conditions during
 * initialization. For production, consider using std::call_once.
 */
static std::unique_ptr<InferenceEngine> g_engine;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Throws a Java RuntimeException from C++ code.
 * 
 * WHY NEEDED?
 * - C++ exceptions cannot cross the JNI boundary directly
 * - We must convert C++ exceptions to Java exceptions
 * - If we don't, the JVM might crash or behave unexpectedly
 * 
 * HOW IT WORKS:
 * 1. FindClass() looks up the Java exception class by its fully-qualified name
 * 2. ThrowNew() creates and throws a new instance with our message
 * 3. After this returns, we should return from the JNI function immediately
 *    (the exception will be thrown when control returns to Java)
 * 
 * @param env    Pointer to JNI environment (provides JVM interaction functions)
 * @param message Error message to include in the exception
 */
static void throwRuntime(JNIEnv* env, const std::string& message) {
    // Find the Java RuntimeException class
    // The "/" separators are Java's internal class name format
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    
    if (exClass != nullptr) {
        // Create and throw a new exception with our message
        // c_str() converts std::string to const char* (C-style string)
        env->ThrowNew(exClass, message.c_str());
    }
    // If FindClass failed, we can't throw - JVM is probably in a bad state anyway
}

/**
 * Anonymous namespace for internal helper functions.
 * 
 * ANONYMOUS NAMESPACE:
 * - Functions/variables here have internal linkage (like static)
 * - They're only visible within this translation unit (this .cpp file)
 * - Prevents name conflicts with other files
 * - More idiomatic C++ than static functions
 */
namespace {

/**
 * Ensures the global inference engine is initialized.
 * 
 * LAZY INITIALIZATION:
 * - Engine is only created when first needed
 * - Subsequent calls return the existing engine
 * - This is the "lazy singleton" pattern
 * 
 * @param env JNI environment (used to throw exceptions if init fails)
 * @return Reference to the global inference engine
 * @throws std::runtime_error if initialization fails
 */
InferenceEngine& ensureEngine(JNIEnv* env) {
    // Check if engine doesn't exist yet
    // In boolean context, unique_ptr is false if it's null
    if (!g_engine) {
        // Create a new InferenceEngine on the heap
        // std::make_unique is the safe way to create unique_ptr objects
        g_engine = std::make_unique<InferenceEngine>();
        
        // Initialize with the ResNet50 ImageNet model
        // Returns false if model file not found or ONNX Runtime fails
        if (!g_engine->initialize("models/resnet50_imagenet.onnx")) {
            // First throw a Java exception (for the Java caller)
            throwRuntime(env, "Failed to initialize inference engine");
            // Then throw a C++ exception (stops execution here)
            throw std::runtime_error("Engine init failed");
        }
    }
    // Dereference the unique_ptr to get a reference to the engine
    return *g_engine;
}

} // namespace

// ============================================================================
// JNI EXPORTED FUNCTIONS
// ============================================================================

/**
 * extern "C" block:
 * 
 * By default, C++ compilers "mangle" function names to include type information.
 * For example, "foo(int)" might become "_Z3fooi".
 * 
 * Java's JNI system expects plain C-style names (no mangling).
 * extern "C" tells the compiler not to mangle these function names.
 * 
 * Without extern "C":
 *   Java looks for: Java_ai_controller_AIController_nativeInfer
 *   Compiler exports: _Z46Java_ai_controller_AIController_nativeInferP7...
 *   Result: UnsatisfiedLinkError!
 */
extern "C" {

/**
 * ==========================================================================
 * Native Image Classification Function
 * ==========================================================================
 * 
 * Java signature: public native float[] nativeInfer(String imagePath);
 * 
 * This function:
 * 1. Receives an image path from Java
 * 2. Loads and preprocesses the image using OpenCV (in C++)
 * 3. Runs ResNet50 inference using ONNX Runtime
 * 4. Returns class probabilities back to Java
 * 
 * JNIEXPORT & JNICALL:
 * - JNIEXPORT: Makes function visible for dynamic linking (platform-specific)
 * - JNICALL: Specifies calling convention (how args are passed, usually empty on Linux)
 * 
 * PARAMETERS:
 * @param env       Pointer to JNI function table. NEVER cache across calls!
 * @param thiz      The Java object this method was called on (AIController instance)
 *                  Marked as unused because we don't need the Java object state
 * @param imagePath Java String containing the path to the image file
 * 
 * @return jfloatArray - Java float[] containing 1000 class probabilities
 *         Returns nullptr if an error occurs (exception will be thrown)
 */
JNIEXPORT jfloatArray JNICALL
Java_ai_controller_AIController_nativeInfer(
    JNIEnv* env,
    jobject /* this */,    // Comment with "this" shows it's the object reference
    jstring imagePath
) {
    // Wrap everything in try-catch to convert C++ exceptions to Java exceptions
    try {
        // ====================================================================
        // INPUT VALIDATION
        // ====================================================================
        
        // Check for null input (Java can pass null)
        if (!imagePath) {
            throwRuntime(env, "imagePath is null");
            return nullptr;  // Return immediately after throwing
        }

        // ====================================================================
        // STRING CONVERSION: Java String → C++ std::string
        // ====================================================================
        
        /**
         * GetStringUTFChars() converts Java String to UTF-8 C string.
         * 
         * IMPORTANT: The returned pointer is valid only until ReleaseStringUTFChars()
         * is called. The JVM might give us a direct pointer to internal data,
         * or it might allocate a copy. Either way, we MUST release it.
         * 
         * The second parameter (nullptr) could be a jboolean* to receive
         * whether a copy was made. We don't need that info here.
         */
        const char* pathChars = env->GetStringUTFChars(imagePath, nullptr);
        
        // Copy to std::string immediately so we can release the JNI string
        // Ternary operator handles the (unlikely) case of null return
        std::string path(pathChars ? pathChars : "");
        
        // ALWAYS release JNI strings when done - failure to do so leaks memory!
        env->ReleaseStringUTFChars(imagePath, pathChars);

        // Additional validation after conversion
        if (path.empty()) {
            throwRuntime(env, "imagePath is empty");
            return nullptr;
        }

        // ====================================================================
        // INFERENCE
        // ====================================================================
        
        // Get or create the inference engine (lazy initialization)
        InferenceEngine& engine = ensureEngine(env);
        
        // Run inference - this does all the heavy lifting:
        // 1. Loads image with OpenCV (or fallback)
        // 2. Preprocesses: resize to 224x224, normalize, convert to tensor
        // 3. Runs ONNX Runtime inference
        // 4. Applies softmax to get probabilities
        std::vector<float> probabilities = engine.classifyImage(path);

        // ====================================================================
        // RESULT CONVERSION: C++ std::vector → Java float[]
        // ====================================================================
        
        /**
         * NewFloatArray() allocates a new Java float[] in the JVM heap.
         * 
         * static_cast<jsize>: jsize is JNI's integer type for array sizes.
         * We cast from size_t to jsize to avoid compiler warnings.
         * 
         * The returned array is managed by the JVM's garbage collector -
         * we don't need to free it manually.
         */
        jfloatArray output = env->NewFloatArray(static_cast<jsize>(probabilities.size()));
        
        if (!output) {
            // Allocation failed - JVM is out of memory
            throwRuntime(env, "Failed to allocate float array for output");
            return nullptr;
        }

        /**
         * SetFloatArrayRegion() copies data from C++ array into Java array.
         * 
         * Parameters:
         * - output: destination Java array
         * - 0: starting index in Java array
         * - size: number of elements to copy
         * - data(): pointer to first element of C++ vector
         */
        env->SetFloatArrayRegion(
            output,                                           // destination
            0,                                                // start index
            static_cast<jsize>(probabilities.size()),         // count
            probabilities.data()                              // source pointer
        );
        
        // Return the Java array - JVM takes ownership
        return output;

    } catch (const std::exception& e) {
        // Catch any C++ exception and convert to Java exception
        // e.what() returns the error message string
        throwRuntime(env, e.what());
        return nullptr;
    }
    // Note: Uncaught exceptions of unknown type will crash the JVM
    // In production, consider adding: catch (...) { throwRuntime(env, "Unknown error"); }
}

// ============================================================================
// REINFORCEMENT LEARNING POLICY INFERENCE
// ============================================================================

/**
 * Separate engine instance for RL policy inference.
 * 
 * WHY SEPARATE?
 * - Different model (CartPole policy vs ResNet50)
 * - Different input format (state vector vs image)
 * - Independent lifecycle (might want one without the other)
 */
static std::unique_ptr<InferenceEngine> g_rlEngine;

/**
 * Ensures the RL policy engine is initialized.
 * Same pattern as ensureEngine() above, but for the RL model.
 * 
 * @param env JNI environment for throwing exceptions
 * @return Reference to the RL inference engine
 */
static InferenceEngine& ensureRLEngine(JNIEnv* env) {
    if (!g_rlEngine) {
        g_rlEngine = std::make_unique<InferenceEngine>();
        
        // Initialize with CartPole policy model
        // initializeRL() is different from initialize() - it sets up for state inference
        if (!g_rlEngine->initializeRL("models/cartpole_policy.onnx")) {
            throwRuntime(env, "Failed to initialize RL policy engine");
            throw std::runtime_error("RL engine init failed");
        }
    }
    return *g_rlEngine;
}

/**
 * ==========================================================================
 * Native RL State Inference Function
 * ==========================================================================
 * 
 * Java signature: public native float[] nativeInferState(float[] state);
 * 
 * This function:
 * 1. Receives an environment state vector from Java (e.g., CartPole's 4D state)
 * 2. Runs the ONNX policy model to get action logits
 * 3. Returns raw logits back to Java (softmax applied in Java)
 * 
 * CARTPOLE STATE VECTOR:
 * - state[0]: cart position (meters from center, range ~[-4.8, 4.8])
 * - state[1]: cart velocity (m/s)
 * - state[2]: pole angle (radians from vertical, range ~[-0.42, 0.42])
 * - state[3]: pole angular velocity (rad/s)
 * 
 * @param env        JNI environment
 * @param thiz       The AIController Java object (unused)
 * @param stateArray Java float[] containing the state vector
 * @return jfloatArray containing action logits [left_logit, right_logit]
 */
JNIEXPORT jfloatArray JNICALL
Java_ai_controller_AIController_nativeInferState(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray stateArray
) {
    try {
        // ====================================================================
        // INPUT VALIDATION
        // ====================================================================
        
        if (!stateArray) {
            throwRuntime(env, "state array is null");
            return nullptr;
        }

        // Get array length
        // GetArrayLength works for any Java array type
        jsize stateLen = env->GetArrayLength(stateArray);
        
        if (stateLen <= 0) {
            throwRuntime(env, "state array is empty");
            return nullptr;
        }

        // ====================================================================
        // ARRAY CONVERSION: Java float[] → C++ std::vector
        // ====================================================================
        
        /**
         * Pre-allocate vector with exact size for efficiency.
         * This avoids resizing during element insertion.
         */
        std::vector<float> state(stateLen);
        
        /**
         * GetFloatArrayRegion() copies data from Java array to C++ buffer.
         * 
         * Unlike GetFloatArrayElements(), this always makes a copy.
         * GetFloatArrayElements() might return a direct pointer (faster but riskier).
         * For small arrays like our 4-element state, copying is fine.
         * 
         * Parameters:
         * - stateArray: source Java array
         * - 0: starting index
         * - stateLen: number of elements to copy
         * - state.data(): destination buffer (pointer to vector's internal array)
         */
        env->GetFloatArrayRegion(stateArray, 0, stateLen, state.data());

        // ====================================================================
        // RL INFERENCE
        // ====================================================================
        
        // Get or create the RL engine
        InferenceEngine& engine = ensureRLEngine(env);
        
        /**
         * inferState() runs the policy network forward pass.
         * 
         * Input: 4D state vector
         * Output: 2D action logits (raw, before softmax)
         * 
         * The Java layer applies softmax to convert logits to probabilities,
         * then selects the action with highest probability.
         */
        std::vector<float> actionLogits = engine.inferState(state);

        // ====================================================================
        // RESULT CONVERSION: C++ std::vector → Java float[]
        // ====================================================================
        
        // Allocate Java array for output
        jfloatArray output = env->NewFloatArray(static_cast<jsize>(actionLogits.size()));
        
        if (!output) {
            throwRuntime(env, "Failed to allocate float array for action logits");
            return nullptr;
        }

        // Copy data from C++ to Java
        env->SetFloatArrayRegion(
            output,
            0,
            static_cast<jsize>(actionLogits.size()),
            actionLogits.data()
        );
        
        return output;

    } catch (const std::exception& e) {
        // Convert C++ exception to Java exception
        throwRuntime(env, e.what());
        return nullptr;
    }
}

} // extern "C"

/**
 * =============================================================================
 * APPENDIX: JNI DATA TYPE MAPPINGS
 * =============================================================================
 * 
 * Java Type     | JNI Type      | C++ Equivalent   | Size
 * --------------|---------------|------------------|-------
 * boolean       | jboolean      | unsigned char    | 8 bit
 * byte          | jbyte         | signed char      | 8 bit
 * char          | jchar         | unsigned short   | 16 bit
 * short         | jshort        | short            | 16 bit
 * int           | jint          | int              | 32 bit
 * long          | jlong         | long long        | 64 bit
 * float         | jfloat        | float            | 32 bit
 * double        | jdouble       | double           | 64 bit
 * Object        | jobject       | void*            | pointer
 * String        | jstring       | void*            | pointer
 * array         | jarray        | void*            | pointer
 * float[]       | jfloatArray   | void*            | pointer
 * 
 * MEMORY MANAGEMENT RULES:
 * 1. Local references (most jobject/jarray) are automatically freed when
 *    the native method returns
 * 2. GetStringUTFChars() → must call ReleaseStringUTFChars()
 * 3. GetFloatArrayElements() → must call ReleaseFloatArrayElements()
 * 4. NewObject/NewFloatArray → managed by GC, don't free manually
 * 5. FindClass() returns local ref → freed automatically on return
 * 
 * THREAD SAFETY:
 * - JNIEnv* is thread-local - NEVER cache it or pass between threads
 * - To use JNI from a native thread, call AttachCurrentThread() first
 * - Global references (NewGlobalRef) can be shared between threads
 */
